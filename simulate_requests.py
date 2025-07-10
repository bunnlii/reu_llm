import numpy as np
import random
import time
import torch
import asyncio
import matplotlib.pyplot as plt

from collections import Counter
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
from datasets import load_dataset
from simulation_env import create_nodes

orca_data = load_dataset("Open-Orca/OpenOrca", split="train")

def get_prompt():
    return orca_data[random.randint(0, len(orca_data) - 1)]["question"]

prompt_lengths = [128, 256, 512]

def simulate_poisson_arrivals(rate_lambda, duration_sec):
    arrivals = []
    current_time = 0.0
    while current_time < duration_sec:
        inter_arrival = np.random.exponential(1.0 / rate_lambda)
        current_time += inter_arrival
        if current_time < duration_sec:
            arrivals.append(current_time)
    return arrivals

def estimate_latency(node, input_len, gen_len):
    N0_dbm_per_hz = -174
    N0_watt = 10 ** ((N0_dbm_per_hz - 30) / 10)

    if node.pathloss_model == "rayleigh":
        pathloss = 1e-3 * np.random.rayleigh(scale=1.0) ** 2
    elif node.pathloss_model == "free_space": 
        distance = 100
        frequency = 2.4e9
        wavelength = 3e8 / frequency
        pathloss = (wavelength / (4 * np.pi * distance)) ** 2
    elif node.pathloss_model == "manhattan":
        pathloss = np.random.uniform(1e-4, 5e-4)
    else:
        pathloss = 1e-3

    tx_power_watt = 10 ** ((node.tx_power_ul_dbm - 30) / 10)
    snr = tx_power_watt * pathloss / (N0_watt * node.bandwidth_ul_hz)
    rate_bps = node.bandwidth_ul_hz * np.log2(1 + snr)

    upload_size = input_len * 2
    download_size = gen_len * 2

    upload_time = upload_size * 8 / rate_bps
    download_time = download_size * 8 / rate_bps

    total_flops_needed = (input_len + gen_len) * 1e9
    total_available_flops = node.num_gpus * node.flops_per_gpu
    inference_time = total_flops_needed / total_available_flops

    return upload_time, inference_time, download_time

async def run_request(model, tokenizer, arrival_times, prompts, input_lens, gen_lens, required_accs, device, nodes):
    # Tokenize the entire batch of prompts together
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

    results = []
    for i, output in enumerate(outputs):
        upload_time, inference_time, download_time = estimate_latency(node, input_lens[i], gen_lens[i])
        simulated_latency = upload_time + inference_time + download_time

        if 0 <= simulated_latency <= 15:
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            results.append({
                "arrival_time": arrival_times[i],
                "input_length": input_lens[i],
                "generation_length": gen_lens[i],
                "required_accuracy": required_accs[i],
                "prompt_tokens": raw_inputs["input_ids"][i],
                "generated_text": generated_text,
                "latency": simulated_latency,
                "node": node.name,
            })

    return results


def bucket_and_batch(arrival_times, input_lens, gen_lens, prompts, required_accs, token_limit=2048):
    batches = []
    current_batch = []
    current_token_sum = 0

    for i in range(len(prompts)):
        total_len = input_lens[i] + gen_lens[i]

        if current_token_sum + total_len > token_limit and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_token_sum = 0

        current_batch.append((arrival_times[i], prompts[i], input_lens[i], gen_lens[i], required_accs[i]))
        current_token_sum += total_len

    if current_batch:
        batches.append(current_batch)

    return batches

async def simulate_requests(rate_lambda, duration_sec, model, tokenizer):
    start_time = time.time()
    arrivals = simulate_poisson_arrivals(rate_lambda, duration_sec)
    device = next(model.parameters()).device
    edge_server, uav, vehicle = await create_nodes()
    nodes = [edge_server, uav, vehicle]

    successful_requests = []
    dropped_requests = 0
    completions_by_second = defaultdict(int)

    # Generate all request parameters
    input_lens = [random.choice(prompt_lengths) for _ in arrivals]
    gen_lens = [random.choice(prompt_lengths) for _ in arrivals]
    required_accs = [random.uniform(0, 1) for _ in arrivals]
    prompts = [get_prompt() for _ in arrivals]

    # Batch requests
    batches = bucket_and_batch(arrivals, input_lens, gen_lens, prompts, required_accs, token_limit=2048)

    for batch in batches:
        batch_arrival_times = [item[0] for item in batch]
        batch_prompts = [item[1] for item in batch]
        batch_input_lens = [item[2] for item in batch]
        batch_gen_lens = [item[3] for item in batch]
        batch_required_accs = [item[4] for item in batch]

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
            dropped_requests += len(batch)
            continue

        for result in batch_results:
            complete_time = result["arrival_time"] + result["latency"]
            second = int(complete_time)
            completions_by_second[second] += 1

        successful_requests.extend(batch_results)
        dropped_requests += len(batch) - len(batch_results)

    print(f"\nTotal requests: {len(arrivals)}")
    print(f"Successful (0s–60s): {len(successful_requests)}")
    print(f"Dropped: {dropped_requests}")
    print(f"Total simulation time: {time.time() - start_time:.2f} seconds")

    return {
    "arrivals": arrivals,
    "completions_by_second": completions_by_second,
    "successful_requests": successful_requests
}



