import os
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from data import plot_latencies
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
    results = asyncio.run(simulate_requests(rate_lambda=50, duration_sec=10, model=model, tokenizer=tokenizer))

    #for i, req in enumerate(results["successful_requests"]):
        #print(f"Request #{i+1} - Arrival time: {req['arrival_time']:.2f}s, Input length: {req['input_length']}, Required accuracy: {req['required_accuracy']:.2f}")
        #print(f"Generated Text:\n{req['generated_text']}\n{'-'*40}")

    return results

if __name__ == "__main__":
    results = main()
    plot_latencies([req["latency"] for req in results["successful_requests"]])
