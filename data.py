import matplotlib.pyplot as plt

def plot_combined_metrics(all_latencies, all_successful_requests, model_names=None):
    """
    all_latencies: list of lists of latencies (one list per model)
    all_successful_requests: list of lists of successful requests (one list per model)
    model_names: list of strings for legend labels, optional
    """

    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'cyan']

    # 1) Latency histogram (overlayed histograms)
    plt.figure(figsize=(12, 5))
    bins = 20
    for i, latencies in enumerate(all_latencies):
        label = model_names[i] if model_names else f"Model {i+1}"
        color = colors[i % len(colors)]
        plt.hist(latencies, bins=bins, color=color, alpha=0.4, edgecolor='black', label=label)
    plt.title("Latency Distribution (Histogram)")
    plt.xlabel("Latency (seconds²)")
    plt.ylabel("Number of Requests")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_latency_histogram.png")
    plt.close()
    print("Saved combined latency histogram to combined_latency_histogram.png")

    # 2) Accuracy per request (line plot)
    plt.figure(figsize=(12, 5))
    for i, successful_requests in enumerate(all_successful_requests):
        accuracies = [req["accuracy"] for req in successful_requests]
        label = model_names[i] if model_names else f"Model {i+1}"
        color = colors[i % len(colors)]
        plt.plot(accuracies, marker='o', linestyle='-', color=color, alpha=0.7, label=label)
    plt.title("BERTScore Accuracy per Request")
    plt.xlabel("Request Index")
    plt.ylabel("Accuracy (F1 Score)")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_accuracy_per_request.png")
    plt.close()
    print("Saved combined accuracy plot to combined_accuracy_per_request.png")

