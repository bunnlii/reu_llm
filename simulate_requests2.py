import numpy as np
import random
import time
import torch
import asyncio
import matplotlib.pyplot as plt
import evaluate

from utils import Request
from math import floor
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
from datasets import load_dataset
from simulation_env import create_nodes

#orca_data = load_dataset("Open-Orca/OpenOrca", split="train")
alpaca_data = load_dataset("tatsu-lab/alpaca", split="train")
bertscore = evaluate.load("bertscore")
bertscore_model = "roberta-large"

# THIS IS FOR ORCA DATASET
# def get_prompt_and_reference():
#     item = orca_data[random.randint(0, len(orca_data) - 1)]
#     prompt = f"Instruction: Answer the following question clearly.\nQuestion: {item['question']}\nAnswer:"
#     return prompt, item["response"]

def is_valid(requests):
    t_min = min(req.latency for req in requests)
    tau_min = t_min
    total_bandwidth = 0
    for req in requests:
        total_bandwidth += req.get_bandwidth()
        if total_bandwidth > tau_min:
            return 0
    return total_bandwidth

# O(n^3 log n) algorithm from paper
def paper_1_sol(reqs):
    reqs.sort(key=lambda x: x.latency, reverse=True)
    for z in range(len(reqs), 0, -1):
        for d in range(z, len(reqs) + 1):
            f_d = reqs[:d]
            f_d.sort(key=lambda x: x.output_length, reverse=True)
            s = f_d[:z]
            bandwidth = is_valid(s)
            if bandwidth:
                return s

def get_prompt_and_reference():
    item = alpaca_data[random.randint(0, len(alpaca_data) - 1)]
    instruction = item['instruction'].strip()
    output = item['output'].strip()
    prompt = f"Instruction: {instruction}\nAnswer:"
    return prompt, output

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


def estimate_latency(prompt_len, output_len):
    total_tokens = prompt_len + output_len
    min_tokens = 256  # 128+128 minimum tokens
    max_tokens = 1024  # 512+512 max tokens
    min_latency = 1.5
    max_latency = 2.0
    
    # Clip tokens within range
    clipped_tokens = max(min_tokens, min(total_tokens, max_tokens))
    
    # Linear interpolation between min_latency and max_latency
    latency = min_latency + (clipped_tokens - min_tokens) * (max_latency - min_latency) / (max_tokens - min_tokens)
    return latency

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
        # accuracy = 0 

        results.append({
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
    estimated_latencies = [] 

    input_lens = [random.choice(prompt_lengths) for _ in arrivals]
    gen_lens = [random.choice(prompt_lengths) for _ in arrivals]
    required_accs = [random.uniform(0, 1) for _ in arrivals]

    prompt_refs = [get_prompt_and_reference() for _ in arrivals]
    prompts = [p for p, r in prompt_refs]
    references = [r for p, r in prompt_refs]

    batches = group_by_epoch(arrivals, input_lens, gen_lens, prompts, required_accs, references)
    result_counter = 1


    for batch in batches:
        # Wrap items as Request objects for paper_1_sol
        request_objs = [
            Request(
                id=k,
                prompt_length=item[2],         # input_length
                output_length=item[3],         # generation_length
                latency=estimate_latency(item[2], item[3]),  
                accuracy=0,
                required_accuracy=item[4],
                prompt=item[1],
                reference=item[5],
                arrival_time=item[0],          # arrival_time
                input_length=item[2]           # input_length again, if needed separately
            )
            for k, item in enumerate(batch)
        ]

        selected = paper_1_sol(request_objs)
        if not selected:
            continue

        # Extract selected requests back into batch format
        batch_arrival_times = [req.arrival_time for req in selected]
        batch_prompts = [req.prompt for req in selected]
        batch_input_lens = [req.input_length for req in selected]
        batch_gen_lens = [req.output_length for req in selected]
        batch_required_accs = [req.required_accuracy for req in selected]
        batch_references = [req.reference for req in selected]

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
            dropped_requests += len(selected)
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

            estimated_latencies.append(result["latency"] ** 2)

            if 0 <= result["latency"] <= 2.0:
                second = int(result["arrival_time"] + result["latency"])
                completions_by_second[second] += 1
                successful_requests.append(result)
            else:
                dropped_requests += 1

            result_counter += 1

    print(f"\nTotal requests: {len(arrivals)}")
    print(f"Successful (1.5s–2.0s deadline): {len(successful_requests)}")
    print(f"Dropped: {dropped_requests}")
    print(f"Total simulation time: {time.time() - start_time:.2f} seconds")
    
    total_squared_latency = sum(estimated_latencies)
    print(f"Total squared latency: {total_squared_latency:.3f} seconds^2")

    return {
        "arrivals": arrivals,
        "completions_by_second": completions_by_second,
        "successful_requests": successful_requests,
        "estimated_latencies": estimated_latencies
    }


