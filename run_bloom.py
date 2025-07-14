import os
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from pathlib import Path
from simulate_requests import simulate_requests
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import asyncio
from collections import Counter
import matplotlib.pyplot as plt

def main():
    device = "cuda:0"
    # quantized_model_path = str(Path("./quantized_bloom_3b").resolve())
    quantized_model_path = str(Path("./quantized_bloom_7b1").resolve())
    #quantized_model_path = str(Path("./quantized_opt_13b").resolve())

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_path, local_files_only=True)

    print("Loading model with GPTQ quantization config")
    gptq_config = GPTQConfig(bits=8, group_size=128, tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        quantized_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=gptq_config,
        device_map={"": 0}
    )

    print("Simulating requests")
    results = asyncio.run(simulate_requests(rate_lambda=50, duration_sec=10, model=model, tokenizer=tokenizer))

    # each generated request (may remove later)
    for i, req in enumerate(results["successful_requests"]):
        print(f"Request #{i+1} - Arrival time: {req['arrival_time']:.2f}s, Input length: {req['input_length']}, Required accuracy: {req['required_accuracy']:.2f}")
        print(f"Generated Text:\n{req['generated_text']}\n{'-'*40}")

    return results

if __name__ == "__main__":
    results = main()

    # Extract arrivals and completions data from results
    arrivals = results["arrivals"] 
    completions_by_second = results["completions_by_second"] 

    # Compute arrival counts by integer second
    arrival_counts = Counter(int(t) for t in arrivals)

    all_seconds = sorted(set(arrival_counts.keys()) | set(completions_by_second.keys()))

    # arrival and completion counts per second
    arrival_rates = [arrival_counts.get(sec, 0) for sec in all_seconds]
    completion_rates = [completions_by_second.get(sec, 0) for sec in all_seconds]

    #Request Completions Over Time
    plt.figure(figsize=(10, 5))
    plt.plot(all_seconds, completion_rates, marker="o")
    plt.title("Request Completions Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of Completed Requests")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("completion_over_time.png")
    plt.show()

    #Completion Rate vs Arrival Rate
    plt.figure(figsize=(10, 6))
    plt.plot(arrival_rates, completion_rates, marker="x", linestyle="-", color="blue")
    plt.title("Completion Rate vs Arrival Rate")
    plt.xlabel("Arrival Rate (requests/sec)")
    plt.ylabel("Completion Rate (requests/sec)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("completion_vs_arrival_rate.png")
    plt.show()

