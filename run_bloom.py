import os
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from data import plot_metrics 
from pathlib import Path
from simulate_requests import simulate_requests
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import asyncio

def main():
    device = "cuda:0"
    quantized_model_path = str(Path("./quantized_bloom_3b").resolve())

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
    results = asyncio.run(simulate_requests(rate_lambda=200, duration_sec=10, model=model, tokenizer=tokenizer))

    return results

if __name__ == "__main__":
    results = main()
    plot_metrics(
        latencies=[req["latency"] for req in results["successful_requests"]],
        successful_requests=results["successful_requests"]
    )

