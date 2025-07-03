import os
import torch
from pathlib import Path
from simulate_requests import simulate_requests
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    quantized_model_path = str(Path("./quantized_bloom_3b").resolve())
    # quantized_model_path = str(Path("./quantized_bloom_7b1").resolve())
    # quantized_model_path = str(Path("./quantized_opt_13b").resolve())

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_path, local_files_only=True)

    print("Loading model with GPTQ quantization config")
    gptq_config = GPTQConfig(bits=8, group_size=128, dataset="c4", tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        quantized_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=gptq_config,
    )
    model.to(device)

    print("Simulating requests")
    requests = simulate_requests(rate_lambda=50, duration_sec=10, model=model, tokenizer=tokenizer)

    for i, req in enumerate(requests):
        print(f"Request #{i+1} - Arrival time: {req['arrival_time']:.2f}s, Input length: {req['input_length']}, Required accuracy: {req['required_accuracy']:.2f}")
        print(f"Generated Text:\n{req['generated_text']}\n{'-'*40}")

if __name__ == "__main__":
    main()
