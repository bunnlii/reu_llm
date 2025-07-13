import numpy as np
import random
import time
import torch
import asyncio
import matplotlib.pyplot as plt

from math import floor
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
from datasets import load_dataset
from simulation_env import create_nodes

orca_data = load_dataset("Open-Orca/OpenOrca", split="train")

def get_prompt():
    return orca_data[random.randint(0, len(orca_data) - 1)]["question"]

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

async def run_request(model, tokenizer, arrival_times, prompts, input_lens, gen_lens, required_accs, device, nodes):
    # Tokenize the batch
    raw_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max(input_lens),
    )
    raw_inputs = {k: v.to(device) for k, v in raw_inputs.items()}

    # Pick a random node
    node = random.choice(nodes)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **raw_inputs,
            max_new_tokens=max(gen_lens),
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    wall_clock_latency = end_time - start_time

    results = []
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        results.append({
            "arrival_time": arrival_times[i],
            "input_length": input_lens[i],
            "generation_length": gen_lens[i],
            "required_accuracy": required_accs[i],
            "prompt_tokens": raw_inputs["input_ids"][i],
            "generated_text": generated_text,
            "latency": wall_clock_latency,
            "node": node.name,
        })

    return results

def group_by_epoch(arrival_times, input_lens, gen_lens, prompts, required_accs, epoch_duration=2.0):
    epoch_buckets = defaultdict(list)
    for i in range(len(prompts)):
        epoch_index = floor(arrival_times[i] / epoch_duration)
        epoch_buckets[epoch_index].append((
            arrival_times[i], prompts[i], input_lens[i], gen_lens[i], required_accs[i]
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

    input_lens = [random.choice(prompt_lengths) for _ in arrivals]
    gen_lens = [random.choice(prompt_lengths) for _ in arrivals]
    required_accs = [random.uniform(0, 1) for _ in arrivals]
    prompts = [get_prompt() for _ in arrivals]

    # Batch requests by epoch
    batches = group_by_epoch(arrivals, input_lens, gen_lens, prompts, required_accs, epoch_duration=2.0)

    for batch in batches:
        trimmed_batch = []
        token_sum = 0

        for item in batch:
            in_len, gen_len = item[2], item[3]
            if len(trimmed_batch) >= max_batch_size:
                break
            if token_sum + in_len + gen_len > max_total_tokens_per_batch:
                break
            trimmed_batch.append(item)
            token_sum += in_len + gen_len

        if not trimmed_batch:
            continue

        batch_arrival_times = [item[0] for item in trimmed_batch]
        batch_prompts = [item[1] for item in trimmed_batch]
        batch_input_lens = [item[2] for item in trimmed_batch]
        batch_gen_lens = [item[3] for item in trimmed_batch]
        batch_required_accs = [item[4] for item in trimmed_batch]

        try:
            batch_results = await run_request(
                model, tokenizer,
                batch_arrival_times, batch_prompts,
                batch_input_lens, batch_gen_lens,
                batch_required_accs,
                device, nodes
            )
        except Exception as e:
            print(f"Batch failed: {e}")
            dropped_requests += len(trimmed_batch)
            continue

        # Print generated texts here:
        for i, result in enumerate(batch_results):
            print(f"Prompt {i+1}: {batch_prompts[i]}")
            print(f"Generated {i+1}: {result['generated_text']}")
            print(f"Latency: {result['latency']:.3f}s, Node: {result['node']}")
            print("-" * 50)

            if 1.5 <= result["latency"] <= 2.0:
                second = int(result["arrival_time"] + result["latency"])
                completions_by_second[second] += 1
                successful_requests.append(result)
            else:
                dropped_requests += 1

        dropped_requests += len(trimmed_batch) - len(batch_results)

    print(f"\nTotal requests: {len(arrivals)}")
    print(f"Successful (1.5s–2.0s deadline): {len(successful_requests)}")
    print(f"Dropped: {dropped_requests}")
    print(f"Total simulation time: {time.time() - start_time:.2f} seconds")

    return {
        "arrivals": arrivals,
        "completions_by_second": completions_by_second,
        "successful_requests": successful_requests
    }

