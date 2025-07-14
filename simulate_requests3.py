import numpy as np
import random
import time
import torch
import asyncio
import matplotlib.pyplot as plt

from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from simulation_env import create_nodes
from avl import AVLTree
from utils import Request

orca_data = load_dataset("Open-Orca/OpenOrca", split="train")

prompt_lengths = [128, 256, 512]


def get_prompt():
    return orca_data[random.randint(0, len(orca_data) - 1)]["question"]


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


def opt_sol(reqs):
    ret = []
    last_reqs = len(reqs) + 1
    i = 0

    while reqs:
        i += 1
        if last_reqs - len(reqs) == 0:
            for request in reqs:
                request.epoch = i + 1
                ret.append(request)
            break
        last_reqs = len(reqs)

        best_solution = []
        highest_output_length = 0
        output_len_tree = AVLTree()

        reqs.sort(key=lambda x: x.latency, reverse=True)

        for request in reqs:
            tau_min = request.latency
            highest_output_length = max(highest_output_length, request.output_length)
            output_len_tree.insert(request)

            lo, hi = 0, highest_output_length + 1
            while lo < hi:
                mid = (lo + hi) // 2
                if output_len_tree.get_bandwidth_sum_total_less_than(Request.get_bandwidth_from_output_length(mid)) < tau_min:
                    lo = mid + 1
                else:
                    hi = mid

            total_bandwidth = output_len_tree.get_bandwidth_sum_total_less_than(Request.get_bandwidth_from_output_length(mid - 1))
            output_set = output_len_tree.get_all_less_than(mid)

            if total_bandwidth <= tau_min and len(output_set) > len(best_solution):
                best_solution = output_set

        for request in best_solution:
            request.epoch = i
            ret.append(request)
        reqs = [r for r in reqs if r not in set(best_solution)]

    return ret


def assign_requests_to_nodes(all_requests, nodes):
    scheduled = opt_sol(all_requests[:])
    node_allocations = [[] for _ in nodes]

    for r in scheduled:
        node_index = (r.epoch - 1) % len(nodes)
        node_allocations[node_index].append(r)

    return node_allocations


async def run_node_requests(model, tokenizer, node_requests, node, device):
    successful = []
    for request in node_requests:
        raw_input = tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=request.prompt_length
        )
        raw_input = {k: v.to(device) for k, v in raw_input.items()}

        try:
            with torch.no_grad():
                output = model.generate(
                    **raw_input,
                    max_new_tokens=request.output_length,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id
                )
            upload_time, inference_time, download_time = estimate_latency(
                node, request.prompt_length, request.output_length)
            latency = upload_time + inference_time + download_time

            if latency <= 15:
                request.generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                request.latency = latency
                request.node = node.name
                successful.append(request)
        except Exception as e:
            print(f"Error processing request: {e}")
    return successful


async def simulate_requests(rate_lambda, duration_sec, model, tokenizer):
    start_time = time.time()
    arrivals = simulate_poisson_arrivals(rate_lambda, duration_sec)
    device = next(model.parameters()).device
    edge_server, uav, vehicle = await create_nodes()
    nodes = [edge_server, uav, vehicle]

    input_lens = [random.choice(prompt_lengths) for _ in arrivals]
    gen_lens = [random.choice(prompt_lengths) for _ in arrivals]
    required_accs = [random.uniform(0, 1) for _ in arrivals]
    prompts = [get_prompt() for _ in arrivals]

    all_requests = [Request(i, input_lens[i], gen_lens[i], 0, required_accs[i], prompts[i], arrivals[i])
                    for i in range(len(arrivals))]

    node_allocations = assign_requests_to_nodes(all_requests, nodes)

    all_successful = []
    tasks = [run_node_requests(model, tokenizer, node_allocations[i], nodes[i], device) for i in range(len(nodes))]
    results = await asyncio.gather(*tasks)
    for node_results in results:
        all_successful.extend(node_results)

    completions_by_second = defaultdict(int)
    for r in all_successful:
        second = int(r.arrival_time + r.latency)
        completions_by_second[second] += 1

    print(f"\nTotal requests: {len(arrivals)}")
    print(f"Successful (0s–60s): {len(all_successful)}")
    print(f"Dropped: {len(arrivals) - len(all_successful)}")
    print(f"Total simulation time: {time.time() - start_time:.2f} seconds")
    print(f"\nSimulation completed in {total_runtime:.2f} seconds")

    return {
        "arrivals": arrivals,
        "completions_by_second": completions_by_second,
        "successful_requests": all_successful
        "total_runtime": total_runtime
    }
