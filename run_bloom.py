import os
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from pathlib import Path
from simulate_requests4 import simulate_requests
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import asyncio

from data import plot_combined_metrics

async def run_model(quantized_model_path, rate_lambda=50, duration_sec=10):
    print(f"Loading tokenizer from {quantized_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model with GPTQ quantization config from {quantized_model_path}")
    gptq_config = GPTQConfig(bits=8, group_size=128, tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        quantized_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=gptq_config,
        device_map={"": 0}
    )

    print(f"Simulating requests for model at {quantized_model_path}")
    results = await simulate_requests(rate_lambda=rate_lambda, duration_sec=duration_sec, model=model, tokenizer=tokenizer)

    return results

async def main():
    model_paths = [
        str(Path("./quantized_bloom_3b").resolve()),
        str(Path("./quantized_bloom_7b1").resolve()),
        str(Path("./quantized_opt_13b").resolve())
    ]

    all_latencies = []
    all_successful_requests = []
    model_names = []

    for path in model_paths:
        results = await run_model(path)
        latencies = [req["latency"] for req in results["successful_requests"]]
        all_latencies.append(latencies)
        all_successful_requests.append(results["successful_requests"])
        model_names.append(Path(path).name)

    plot_combined_metrics(all_latencies, all_successful_requests, model_names=model_names)

if __name__ == "__main__":
    asyncio.run(main())

