import matplotlib.pyplot as plt

def plot_latencies(latencies, save_path="latency_plot.png"):
    if not latencies:
        print("No latencies to plot.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(latencies, marker='o', linestyle='-', color='blue', alpha=0.7)
    plt.title("Estimated Latency per Request")
    plt.xlabel("Request Index")
    plt.ylabel("Latency (seconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()  # Comment out if running non-interactively
    plt.close()
    print(f"Latency plot saved to {save_path}")
