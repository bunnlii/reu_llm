import numpy as np
import random
import time
import concurrent.futures
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
from datasets import load_dataset

orca_data = load_dataset("Open-Orca/OpenOrca", split="train")
prompt_lengths = [128]

def simulate_requests(rate_lambda, duration_sec, model, tokenizer):
    arrivals = []
    current_time = 0.0
    while current_time < duration_sec:
        inter_arrival = np.random.exponential(1.0 / rate_lambda)
        current_time += inter_arrival
        if current_time < duration_sec:
            arrivals.append(current_time)

    device = next(model.parameters()).device
    successful_requests, dropped_requests = [], 0

    for arrival_time in arrivals:
        input_len = np.random.choice(prompt_lengths)
        gen_len = np.random.choice(prompt_lengths)
        required_acc = np.random.uniform(0, 1)

        prompt = orca_data[random.randint(0, len(orca_data) - 1)]["question"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=input_len)
        if 'attention_mask' in inputs:
            inputs['attention_mask'] = inputs['attention_mask'].float()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        start_time = time.time()
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    model.generate,
                    **inputs,
                    max_new_tokens=gen_len,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id
                )
                output = future.result(timeout=10)
            latency = time.time() - start_time
        except concurrent.futures.TimeoutError:
            print(f"\nRequest timed out (>{10} seconds).")
            dropped_requests += 1
            continue

        if 1.5 <= latency <= 10:
            input_ids = inputs["input_ids"][0]
            generated_ids = output[0][len(input_ids):]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            print("\n====== Successful Request ======")
            print(f"Prompt:\n{prompt.strip()}")
            print("\nAnswer:\n" + generated_text)
            print(f"\nLatency: {latency:.2f} seconds")
            print(f"Accuracy: {required_acc:.2f}")
            print("================================")

            successful_requests.append({
                "arrival_time": arrival_time,
                "input_length": input_len,
                "generation_length": gen_len,
                "required_accuracy": required_acc,
                "prompt_tokens": inputs["input_ids"],
                "generated_text": generated_text,
                "latency": latency
            })
        else:
            dropped_requests += 1

    print(f"\nTotal requests: {len(arrivals)}")
    print(f"Successful (within latency): {len(successful_requests)}")
    print(f"Dropped (outside 1.5s–10s): {dropped_requests}")
    return successful_requests
