import numpy as np
import random
import time
import torch
import asyncio
import matplotlib.pyplot as plt
import evaluate

from math import floor
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
from datasets import load_dataset
from simulation_env import create_nodes

from utils import Request
from avl import AVLTree  # replace with actual AVLTree import
from algo1_ontime import paper_1_sol  
from fast_scheduler_ontime import opt_sol

import matplotlib.pyplot as plt
from collections import defaultdict


alpaca_data = load_dataset("tatsu-lab/alpaca", split="train")
bertscore = evaluate.load("bertscore")
bertscore_model = "roberta-large"

request_id_map = {}

def s_to_latency(s):
    return 42_500*s 

def get_prompt_and_reference():
    item = alpaca_data[random.randint(0, len(alpaca_data) - 1)]
    instruction = item['instruction'].strip()
    output = item['output'].strip()
    prompt = f"Instruction: {instruction}\nAnswer:"
    return prompt, output

prompt_lengths = [128, 256, 512]
max_batch_size = 6
max_total_tokens_per_batch = 2048

def simulate_poisson_arrivals(rate_lambda, duration_sec):
    arrivals = []
    current_time = 0.0
    while current_time < duration_sec:
        inter_arrival = np.random.exponential(1.0 / rate_lambda)
        current_time += inter_arrival
        if current_time < duration_sec:
            arrivals.append(current_time)
    return arrivals

def dbm_to_watts(dbm):
    return 10 ** (dbm / 10) / 1000

def calculate_shannon_delay(bits, bandwidth_hz, tx_dbm, pathloss_gain=1e-3, noise_dbm_hz=-174):
    tx_power = dbm_to_watts(tx_dbm)
    noise_power = dbm_to_watts(noise_dbm_hz) * bandwidth_hz
    snr = (tx_power * pathloss_gain) / noise_power
    capacity = bandwidth_hz * np.log2(1 + snr)
    return bits / capacity

async def run_request(model, tokenizer, arrival_times, prompts, references, input_lens, gen_lens, required_accs, device, nodes, request_ids=None):
    raw_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max(input_lens),
    )
    raw_inputs = {k: v.to(device) for k, v in raw_inputs.items()}

    #node = random.choice(nodes)
    node = nodes[1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    batch_start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **raw_inputs,
            max_new_tokens=max(gen_lens),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    batch_end_time = time.time()
    real_latency = batch_end_time - batch_start_time
    total_tokens_batch = sum(input_lens[i] + gen_lens[i] for i in range(len(outputs)))

    flops_per_token = 1e9
    results = []

    for i, output in enumerate(outputs):
        bits_in = input_lens[i] * 16
        bits_out = gen_lens[i] * 16

        uplink_delay = calculate_shannon_delay(bits_in, node.bandwidth_ul_hz, node.tx_power_ul_dbm)
        downlink_delay = calculate_shannon_delay(bits_out, node.bandwidth_dl_hz, node.tx_power_dl_dbm)
        transmission_delay = uplink_delay + downlink_delay

        total_tokens = input_lens[i] + gen_lens[i]
        compute_latency = total_tokens * flops_per_token / node.flops
        estimated_latency = transmission_delay + compute_latency

        real_latency_i = real_latency * (total_tokens / total_tokens_batch)
        total_latency = estimated_latency + real_latency_i

        input_ids = raw_inputs["input_ids"][i]
        output_ids = output.tolist()

        generated_ids = output_ids[len(input_ids):]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        reference_text = references[i]

        bertscore_result = bertscore.compute(
            predictions=[generated_text],
            references=[reference_text],
            lang="en",
            model_type=bertscore_model,
            rescale_with_baseline=False
        )
        accuracy = bertscore_result["f1"][0]

        results.append({
            "id": request_ids[i] if request_ids else -1,
            "arrival_time": arrival_times[i],
            "input_length": input_lens[i],
            "generation_length": gen_lens[i],
            "required_accuracy": required_accs[i],
            "prompt_tokens": raw_inputs["input_ids"][i],
            "generated_text": generated_text,
            "reference_text": reference_text,
            "accuracy": accuracy,
            "latency": total_latency,
            "estimated_latency": estimated_latency,
            "real_latency": real_latency_i,
            "node": node.name
        })

    return results

def group_by_epoch(arrival_times, input_lens, gen_lens, prompts, required_accs, references, epoch_duration=2.0):
    epoch_buckets = defaultdict(list)
    for i in range(len(prompts)):
        epoch_index = floor(arrival_times[i] / epoch_duration)
        epoch_buckets[epoch_index].append((
            arrival_times[i], prompts[i], input_lens[i], gen_lens[i], required_accs[i], references[i]
        ))
    return [epoch_buckets[k] for k in sorted(epoch_buckets.keys())]

async def simulate_requests(rate_lambda, duration_sec, model, tokenizer):
    start_time = time.time()
    arrivals = simulate_poisson_arrivals(rate_lambda, duration_sec)
    device = next(model.parameters()).device
    edge_server, uav, vehicle = await create_nodes()
    nodes = [edge_server, uav, vehicle]

    baseline_successful_requests = []
    baseline_dropped_requests = 0
    completions_by_second = defaultdict(int)
    estimated_latencies = []

    input_lens = [random.choice(prompt_lengths) for _ in arrivals]
    gen_lens = [random.choice(prompt_lengths) for _ in arrivals]
    required_accs = [random.uniform(0, 1) for _ in arrivals]

    prompt_refs = [get_prompt_and_reference() for _ in arrivals]
    prompts = [p for p, r in prompt_refs]
    references = [r for p, r in prompt_refs]

    batches = group_by_epoch(arrivals, input_lens, gen_lens, prompts, required_accs, references)
    result_counter = 1
    request_objects = []
    req_id = 0

    start_e_time = time.time()

    for batch in batches:
        i = 0
        while i < len(batch):
            
            start_time = time.time()

            trimmed_batch = []
            token_sum = 0
            j = i
            while j < len(batch):
                in_len, gen_len = batch[j][2], batch[j][3]
                if len(trimmed_batch) >= max_batch_size:
                    break
                if token_sum + in_len + gen_len > max_total_tokens_per_batch:
                    break
                trimmed_batch.append(batch[j])
                token_sum += in_len + gen_len
                j += 1

            if not trimmed_batch:
                i += 1
                continue

            batch_arrival_times = [item[0] for item in trimmed_batch]
            batch_prompts = [item[1] for item in trimmed_batch]
            batch_input_lens = [item[2] for item in trimmed_batch]
            batch_gen_lens = [item[3] for item in trimmed_batch]
            batch_required_accs = [item[4] for item in trimmed_batch]
            batch_references = [item[5] for item in trimmed_batch]

            try:
                batch_results = await run_request(
                    model, tokenizer,
                    batch_arrival_times, batch_prompts, batch_references,
                    batch_input_lens, batch_gen_lens,
                    batch_required_accs,
                    device, nodes
                )
            except Exception as e:
                print(f"Batch failed: {e}")
                baseline_dropped_requests += len(trimmed_batch)
                i = j
                continue

            end_time = time.time()
            batch_time = end_time - start_time
            dropped_this_batch = 0
            batch_indiv_time = 0.0

            for k, result in enumerate(batch_results):
                batch_indiv_time += result["latency"]

            for k, result in enumerate(batch_results):
                print(f"Request {result_counter}: BERTScore Accuracy = {result['accuracy']:.3f}")
                # print(f"=== Result {result_counter} ===")
                # print(f"Node: {result['node']}")
                # print(f"Latency: {result['latency']:.3f} seconds")
                # print(f"BERTScore Accuracy: {result['accuracy']:.3f}\n")
                # print("Prompt:")
                # print(batch_prompts[k].strip())
                # print("\nGenerated Response:")
                # print(result['generated_text'].strip())
                # print("\nReference Response:")
                # print(result['reference_text'].strip())
                # print("=" * 40)

                estimated_latencies.append(result["latency"] ** 2)

                req = Request(
                    id=req_id,
                    prompt_length=result["input_length"],
                    output_length=result["generation_length"],
                    latency=int(random.random() * 375_000 + 200_000),
                    accuracy=result["accuracy"],
                    time_taken = result["latency"]
                )

                # print(result["prompt_tokens"])
                # print(result["input_length"], result["generation_length"], result["latency"], s_to_latency(result["latency"]), req.latency, result["accuracy"])
                req_id += 1
                request_objects.append(req)

                print(batch_indiv_time, s_to_latency(batch_indiv_time), req.latency, s_to_latency(batch_indiv_time) <= req.latency)
                if s_to_latency(batch_indiv_time) <= req.latency:
                    second = int(result["arrival_time"] + result["latency"])
                    completions_by_second[second] += 1
                    baseline_successful_requests.append(result)
                else:
                    baseline_dropped_requests += 1
                    dropped_this_batch += 1

                result_counter += 1

            # baseline_dropped_requests += len(trimmed_batch) - len(batch_results)
            i = j

            print("Batch completed. Time: ", batch_time, " size: ", len(trimmed_batch), " dropped: ", dropped_this_batch)

    print(f"\nTotal requests: {len(arrivals)}")
    print(f"Baseline: {len(baseline_successful_requests)} success")
    print(f"Baseline dropped: {baseline_dropped_requests}")
    print(f"Total simulation time: {time.time() - start_e_time:.2f} seconds")

    print("\nPreparing to evaluate optimized schedulers...")

    prompts_map = {i: p for i, p in enumerate(prompts)}
    references_map = {i: r for i, r in enumerate(references)}

    for i, req in enumerate(request_objects):
        req.id = i
    with open("requests.txt", "w") as f:
        for req in request_objects:
            f.write(repr(req) + "\n")

    paper_dropped = await evaluate_scheduler("PAPER_1_SOL", paper_1_sol, request_objects, prompts_map, references_map, model, tokenizer, nodes, device)
    opt_dropped = await evaluate_scheduler("OPT_SOL", opt_sol, request_objects, prompts_map, references_map, model, tokenizer, nodes, device)

    labels = ['Baseline', 'OPT_SOL', 'PAPER_1_SOL']
    dropped_counts = [baseline_dropped_requests, opt_dropped, paper_dropped]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, dropped_counts, color=['red', 'green', 'blue'])
    plt.ylabel("Dropped Requests")
    plt.title("Dropped Requests by Scheduler")
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('multiple_algos.png')
    # plt.show()
    # plt.save

    return {
        "arrivals": arrivals,
        "completions_by_second": completions_by_second,
        "baseline_dropped": baseline_dropped_requests,
        "opt_dropped": opt_dropped,
        "paper_dropped": paper_dropped,
        "successful_requests": baseline_successful_requests,
        "estimated_latencies": estimated_latencies,
    }

async def evaluate_scheduler(name, scheduler_fn, requests, prompts, references, model, tokenizer, nodes, device):
    print(f"\nEvaluating: {name}")
    scheduled_reqs = scheduler_fn(requests.copy())

    if scheduled_reqs is None:
        print(f"{name} — Scheduler returned None.")
        return len(requests)

    ttime = 0.0
    successful = 0
    dropped = 0
    batched = []
    batches = {}

    for req in scheduled_reqs:
        request_id_map[req.id] = req
        epoch_id = req.epoch
        batches.setdefault(epoch_id, []).append(req)

    for epoch_id in batches:
        batched.append(batches[epoch_id])

    for batch in batched:
        prompts_batch = [prompts[r.id] for r in batch]
        references_batch = [references[r.id] for r in batch]
        input_lens = [r.prompt_length for r in batch]
        output_lens = [r.output_length for r in batch]
        required_accs = [r.accuracy for r in batch]
        arrival_times = [r.creation_time for r in batch]
        request_ids = [r.id for r in batch]

        try:
            results = await run_request(
                model, tokenizer,
                arrival_times, prompts_batch, references_batch,
                input_lens, output_lens, required_accs,
                device, nodes, request_ids
            )
        except Exception as e:
            dropped += len(batch)
            continue

        for r in results:
            ttime += r["latency"]
            if s_to_latency(r["latency"]) <= request_id_map[r["id"]].latency:
                successful += 1
            else:
                dropped += 1

    total = len(scheduled_reqs)
    accuracy = successful / total if total > 0 else 0.0
    print(f"{name} — Accuracy: {accuracy * 100:.2f}% ({successful}/{total})")
    return dropped

