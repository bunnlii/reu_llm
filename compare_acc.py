import asyncio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig

from simulate_requests4 import simulate_requests       # baseline simulation function
from algo1_ontime import paper_1_sol                  # paper_1 scheduler algorithm
from utils import Request                              # Request class
from simulation_env import create_nodes               # function to create nodes
from simulate_requests4 import evaluate_scheduler     # scheduler evaluation function


async def run_algorithms_on_model(model_path, rate_lambda=100, duration_sec=10):
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model with GPTQ quantization config from {model_path}")
    gptq_config = GPTQConfig(bits=8, group_size=128, tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=gptq_config,
        device_map={"": 0}
    )

    print("\n--- Running baseline simulation ---")
    baseline_results = await simulate_requests(rate_lambda, duration_sec, model, tokenizer)

    # Prepare Request objects, prompts, references for scheduler evaluation
    requests = []
    prompts_map = {}
    references_map = {}

    # Create Request objects for each successful baseline request
    for i, res in enumerate(baseline_results["successful_requests"]):
        # You may need to adjust fields based on your Request class definition
        req = Request(
            id=i,
            prompt_length=res["input_length"],
            output_length=res["generation_length"],
            latency=int(np.random.uniform(200_000, 575_000)),  # Randomized example latency (use your actual logic if available)
            accuracy=res["accuracy"],
            time_taken=res["latency"]
        )
        req.creation_time = res["arrival_time"]  # Store arrival time in Request object
        req.epoch = int(req.creation_time // 2)  # Example epoch grouping every 2 seconds (adjust if needed)
        requests.append(req)
        prompts_map[i] = res["prompt_tokens"]
        references_map[i] = res["reference_text"]

    device = next(model.parameters()).device
    nodes = await create_nodes()

    print("\n--- Evaluating paper_1 scheduler ---")
    paper_1_results = await evaluate_scheduler(
        "PAPER_1_SOL",
        paper_1_sol,
        requests,
        prompts_map,
        references_map,
        model,
        tokenizer,
        nodes,
        device
    )

    # Return results for both baseline and paper_1
    return {
        "baseline": baseline_results,
        "paper_1_sol": paper_1_results
    }


def plot_accuracy_comparison(all_results, model_name, save_path=None):
    plt.figure(figsize=(12, 6))

    markers = {'baseline': 'o', 'paper_1_sol': 's'}
    colors = {'baseline': 'blue', 'paper_1_sol': 'green'}

    for algo_name, results in all_results.items():
        successful_requests = results["successful_requests"]
        accuracies = [r["accuracy"] for r in successful_requests]

        batch_size = 100
        num_batches = len(accuracies) // batch_size
        avg_accuracies = [np.mean(accuracies[i * batch_size:(i + 1) * batch_size]) for i in range(num_batches)]

        remainder = len(accuracies) % batch_size
        if remainder > 0:
            avg_accuracies.append(np.mean(accuracies[-remainder:]))

        x_vals = np.arange(1, len(avg_accuracies) + 1) * batch_size

        plt.plot(x_vals, avg_accuracies,
                 marker=markers.get(algo_name, 'o'),
                 linestyle='-',
                 color=colors.get(algo_name, 'black'),
                 alpha=0.8,
                 label=algo_name)

    plt.xlabel("Number of Requests Processed")
    plt.ylabel("Average Accuracy (BERTScore F1)")
    plt.title(f"Accuracy Comparison on {model_name}")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


async def main():
    model_paths = [
        str(Path("./quantized_bloom_3b").resolve()),
        str(Path("./quantized_bloom_7b1").resolve())
    ]

    for path in model_paths:
        model_name = Path(path).name
        print(f"\n===== Running algorithms on model: {model_name} =====")

        results = await run_algorithms_on_model(path, rate_lambda=200, duration_sec=10)
        plot_accuracy_comparison(results, model_name=model_name, save_path=f"accuracy_comparison_{model_name}.png")


if __name__ == "__main__":
    asyncio.run(main())


