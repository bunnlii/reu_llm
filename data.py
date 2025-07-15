import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size):
    """Compute simple moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_combined_metrics(all_latencies, all_successful_requests, model_names=None, ma_window=20):
    """
    all_latencies: list of lists of latencies (one list per model)
    all_successful_requests: list of lists of successful requests (one list per model)
    model_names: list of strings for legend labels, optional
    ma_window: moving average window size
    """
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'cyan']

    # 1) Latency line plots (was histogram)
    plt.figure(figsize=(12, 5))
    bins = 20
    for i, latencies in enumerate(all_latencies):
        label = model_names[i] if model_names else f"Model {i+1}"
        color = colors[i % len(colors)]

        counts, bin_edges = np.histogram(latencies, bins=bins)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        plt.plot(bin_centers, counts, marker='o', linestyle='-', color=color, label=label)

    plt.title("Latency Distribution (Line Plot)")
    plt.xlabel("Latency (seconds²)")
    plt.ylabel("Number of Requests")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_latency_line_plot.png")
    plt.close()
    print("Saved combined latency line plot to combined_latency_line_plot.png")

    # 2) Accuracy (moving average over time)
    avg_accuracies = []
    plt.figure(figsize=(12, 5))
    for i, successful_requests in enumerate(all_successful_requests):
        accuracies = [req["accuracy"] for req in successful_requests]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        avg_accuracies.append(avg_accuracy)

        smoothed = moving_average(accuracies, ma_window)
        x_vals = list(range(len(smoothed)))

        label = (model_names[i] if model_names else f"Model {i+1}") + f" (Avg: {avg_accuracy:.3f})"
        color = colors[i % len(colors)]
        plt.plot(x_vals, smoothed, linestyle='-', color=color, alpha=0.8, label=label)

    plt.title(f"Smoothed BERTScore Accuracy Over Time (Window = {ma_window})")
    plt.xlabel("Request Index")
    plt.ylabel("Moving Avg Accuracy (F1 Score)")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_smoothed_accuracy.png")
    plt.close()
    print("Saved smoothed accuracy plot to combined_smoothed_accuracy.png")

    # 3) Print average accuracies
    print("\n=== Average Accuracies ===")
    for i, avg in enumerate(avg_accuracies):
        label = model_names[i] if model_names else f"Model {i+1}"
        print(f"{label}: {avg:.4f}")

