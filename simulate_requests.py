import numpy as np
import random
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
from datasets import load_dataset
from simulation_env import create_nodes

orca_data = load_dataset("Open-Orca/OpenOrca", split="train")

def get_prompt():
    return orca_data[random.randint(0, len(orca_data) - 1)]["question"]

prompt_lengths = [64]

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

async def run_request(model, tokenizer, arrival_time, device, nodes, executor):
    input_len = np.random.choice(prompt_lengths)
    gen_len = np.random.choice(prompt_lengths)
    required_acc = np.random.uniform(0, 1)

    prompt = get_prompt()
    raw_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=input_len)
    if 'attention_mask' in raw_inputs:
        raw_inputs['attention_mask'] = raw_inputs['attention_mask'].float()

    node = random.choice(nodes)
    upload_time, inference_time, download_time = estimate_latency(node, input_len, gen_len)
    simulated_latency = upload_time + inference_time + download_time

    loop = asyncio.get_event_loop()
    try:
        output = await loop.run_in_executor(
            executor,
            lambda: model.generate(
                **{k: v.to(device) for k, v in raw_inputs.items()},
                max_new_tokens=gen_len,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        )
    except Exception as e:
        print(f"Request failed: {e}")
        return None

    if 1.5 <= simulated_latency <= 60:
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Successful Request")
        print(f"Node: {node.name}")
        print(f"Prompt: {prompt}")
        print(f"Answer: {generated_text}")
        print(f"Latency: {simulated_latency:.2f} seconds")
        print(f"Required Accuracy: {required_acc:.2f}")

        return {
            "arrival_time": arrival_time,
            "input_length": input_len,
            "generation_length": gen_len,
            "required_accuracy": required_acc,
            "prompt_tokens": raw_inputs["input_ids"].to(device),
            "generated_text": generated_text,
            "latency": simulated_latency,
            "node": node.name
        }
    else:
        return None

async def simulate_requests(rate_lambda, duration_sec, model, tokenizer):
    start_time = time.time()
    arrivals = simulate_poisson_arrivals(rate_lambda, duration_sec)
    device = next(model.parameters()).device
    edge_server, uav, vehicle = await create_nodes()
    nodes = [edge_server, uav, vehicle]

    successful_requests = []
    dropped_requests = 0

    with ThreadPoolExecutor() as executor:
        tasks = [
            run_request(model, tokenizer, arrival, device, nodes, executor)
            for arrival in arrivals
        ]
        results = await asyncio.gather(*tasks)

    for result in results:
        if result is not None:
            successful_requests.append(result)
        else:
            dropped_requests += 1

    print(f"\nTotal requests: {len(arrivals)}")
    print(f"Successful (1.5s–60s): {len(successful_requests)}")
    print(f"Dropped: {dropped_requests}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal simulation time: {total_time:.2f} seconds")
    print(f"Total simulation time: {time.time() - start_time:.2f} seconds")

    return successful_requests

