import matplotlib.pyplot as plt

def plot_latencies(latencies, save_path="latency_plot.png"):
    if not latencies:
        print("No latencies to plot.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))  # 1 row, 2 cols

    # graph 1: latency per request (line plot)
    axs[0].plot(latencies, marker='o', linestyle='-', color='blue', alpha=0.7)
    axs[0].set_title("Estimated Latency per Request")
    axs[0].set_xlabel("Request Index")
    axs[0].set_ylabel("Latency (seconds)")
    axs[0].grid(True)

    # graph 2: histogram of latency frequency
    axs[1].hist(latencies, bins=20, color='green', edgecolor='black', alpha=0.7)
    axs[1].set_title("Latency Distribution (Histogram)")
    axs[1].set_xlabel("Latency (seconds)")
    axs[1].set_ylabel("Number of Requests")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

    print(f"Latency plots saved to {save_path}")

