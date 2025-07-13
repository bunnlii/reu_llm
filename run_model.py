import os
import torch
import time
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from pathlib import Path
from simulate_requests3 import simulate_requests
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import asyncio
from collections import Counter
import matplotlib.pyplot as plt

def main():
    start_time = time.time()
    device = "cuda:0"
    quantized_model_path = str(Path("./quantized_bloom_3b").resolve())
    # quantized_model_path = str(Path("./quantized_bloom_7b1").resolve())
    # quantized_model_path = str(Path("./quantized_opt_13b").resolve())

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
        print(f"Request #{i+1} - Arrival time: {req.arrival_time:.2f}s, Input length: {req.prompt_length}, Required accuracy: {req.accuracy:.2f}")
        print(f"Generated Text:\n{req.generated_text}\n{'-'*40}")

    total_time = time.time() - start_time
    print(f"\n Total simulation runtime: {total_time:.2f} seconds")

    return results

if __name__ == "__main__":
    results = main()
