import matplotlib.pyplot as plt

def plot_metrics(latencies, successful_requests):
    if not successful_requests:
        print("No successful requests to plot.")
        return

    accuracies = [req["accuracy"] for req in successful_requests]

    # Plot 1: Latency per request
    # plt.figure(figsize=(10, 4))
    # plt.plot(latencies, marker='o', linestyle='-', color='blue', alpha=0.7)
    # plt.title("Estimated Latency per Request")
    # plt.xlabel("Request Index")
    # plt.ylabel("Latency (seconds²)")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("latency_per_request.png")
    # plt.close()
    # print("Saved latency line plot to latency_per_request.png")

    #Plot 2: Latency histogram
    # plt.figure(figsize=(10, 4))
    # plt.hist(latencies, bins=20, color='green', edgecolor='black', alpha=0.7)
    # plt.title("Latency Distribution (Histogram)")
    # plt.xlabel("Latency (seconds²)")
    # plt.ylabel("Number of Requests")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("latency_histogram.png")
    # plt.close()
    # print("Saved latency histogram to latency_histogram.png")

    #Plot 3: Accuracy per request
    plt.figure(figsize=(10, 4))
    plt.plot(accuracies, marker='o', linestyle='-', color='orange', alpha=0.7)
    plt.title("BERTScore Accuracy per Request")
    plt.xlabel("Request Index")
    plt.ylabel("Accuracy (F1 Score)")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_per_request.png")
    plt.close()
    print("Saved accuracy plot to accuracy_per_request.png")

