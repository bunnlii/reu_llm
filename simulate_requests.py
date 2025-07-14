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

orca_data = load_dataset("Open-Orca/OpenOrca", split="train")
#bertscore = evaluate.load("bertscore")
#bertscore_model = "microsoft/deberta-xlarge-mnli"


def get_prompt_and_reference():
    item = orca_data[random.randint(0, len(orca_data) - 1)]
    return item["question"], item["response"]

prompt_lengths = [128, 256, 512]
max_batch_size = 8
max_total_tokens_per_batch = 4096

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

async def run_request(model, tokenizer, arrival_times, prompts, references, input_lens, gen_lens, required_accs, device, nodes):
    raw_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max(input_lens),
    )
    raw_inputs = {k: v.to(device) for k, v in raw_inputs.items()}

    node = random.choice(nodes)

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

        input_ids = raw_inputs["input_ids"][i]
        output_ids = output.tolist()

        generated_ids = output_ids[len(input_ids):]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        reference_text = references[i]

        # bertscore_result = bertscore.compute(
        #     predictions=[generated_text],
        #     references=[reference_text],
        #     lang="en",
        #     model_type=bertscore_model,
        #     rescale_with_baseline=False
        # )
        # accuracy = bertscore_result["f1"][0]
        accuracy = 0

        results.append({
            "arrival_time": arrival_times[i],
            "input_length": input_lens[i],
            "generation_length": gen_lens[i],
            "required_accuracy": required_accs[i],
            "prompt_tokens": raw_inputs["input_ids"][i],
            "generated_text": generated_text,
            "reference_text": reference_text,
            "accuracy": accuracy,
            "latency": estimated_latency,
            "node": node.name,
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

    successful_requests = []
    dropped_requests = 0
    completions_by_second = defaultdict(int)
    estimated_latencies = []  # New: collect estimated latency per request

    input_lens = [random.choice(prompt_lengths) for _ in arrivals]
    gen_lens = [random.choice(prompt_lengths) for _ in arrivals]
    required_accs = [random.uniform(0, 1) for _ in arrivals]

    prompt_refs = [get_prompt_and_reference() for _ in arrivals]
    prompts = [p for p, r in prompt_refs]
    references = [r for p, r in prompt_refs]

    batches = group_by_epoch(arrivals, input_lens, gen_lens, prompts, required_accs, references)
    result_counter = 1

    for batch in batches:
        i = 0
        while i < len(batch):
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
                dropped_requests += len(trimmed_batch)
                i = j
                continue

            for k, result in enumerate(batch_results):
                print(f"=== Result {result_counter} ===")
                print(f"Node: {result['node']}")
                print(f"Latency: {result['latency']:.3f} seconds")
                print(f"BERTScore Accuracy: {result['accuracy']:.3f}\n")

                print("Prompt:")
                print(batch_prompts[k].strip())
                print("\nGenerated Response:")
                print(result['generated_text'].strip())
                print("\nReference Response:")
                print(result['reference_text'].strip())

                print("=" * 40)

                estimated_latencies.append(result["latency"])

                if 0 <= result["latency"] <= 2.0:
                    second = int(result["arrival_time"] + result["latency"])
                    completions_by_second[second] += 1  
                    successful_requests.append(result)
                else:
                    dropped_requests += 1

                result_counter += 1

            dropped_requests += len(trimmed_batch) - len(batch_results)
            i = j

    print(f"\nTotal requests: {len(arrivals)}")
    print(f"Successful (1.5s–2.0s deadline): {len(successful_requests)}")
    print(f"Dropped: {dropped_requests}")
    print(f"Total simulation time: {time.time() - start_time:.2f} seconds")

    return {
        "arrivals": arrivals,
        "completions_by_second": completions_by_second,
        "successful_requests": successful_requests,
        "estimated_latencies": estimated_latencies
    }

