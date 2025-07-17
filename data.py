import matplotlib.pyplot as plt
import numpy as np

def plot_combined_metrics(all_latencies, all_successful_requests, model_names=None, ma_window=20):
    """
    all_latencies: list of lists of latencies (one list per model)
    all_successful_requests: list of lists of successful requests (one list per model)
    model_names: list of strings for legend labels, optional
    ma_window: moving average window size
    """
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'cyan']

def plot_multiple_accuracies(successful_requests_list, model_names, batch_size=100, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 5))

    for successful_requests, model_name in zip(successful_requests_list, model_names):
        accuracies = [req["accuracy"] for req in successful_requests]
        num_batches = len(accuracies) // batch_size
        avg_accuracies = [np.mean(accuracies[i*batch_size:(i+1)*batch_size]) for i in range(num_batches)]
        remainder = len(accuracies) % batch_size
        if remainder > 0:
            avg_accuracies.append(np.mean(accuracies[-remainder:]))

        x_vals = np.arange(1, len(avg_accuracies) + 1) * batch_size
        plt.plot(x_vals, avg_accuracies, marker='o', linestyle='-', label=model_name)

    plt.xlabel("Number of Requests Processed")
    plt.ylabel("Average BERTScore Accuracy (F1)")
    plt.title(f"Average Accuracy per {batch_size} Requests")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved combined accuracy plot to {save_path}")
        plt.close()
    else:
        plt.show()


# def moving_average(data, window_size):
#     """Compute simple moving average."""
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

#     # 1) Latency line plots (was histogram)
#     plt.figure(figsize=(12, 5))
#     bins = 20
#     for i, latencies in enumerate(all_latencies):
#         label = model_names[i] if model_names else f"Model {i+1}"
#         color = colors[i % len(colors)]

#         counts, bin_edges = np.histogram(latencies, bins=bins)
#         bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

#         plt.plot(bin_centers, counts, marker='o', linestyle='-', color=color, label=label)

#     plt.title("Latency Distribution (Line Plot)")
#     plt.xlabel("Latency (seconds²)")
#     plt.ylabel("Number of Requests")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("combined_latency_line_plot.png")
#     plt.close()
#     print("Saved combined latency line plot to combined_latency_line_plot.png")

#     # 2) Accuracy (moving average over time) — COMMENTED OUT
#     # avg_accuracies = []
#     # plt.figure(figsize=(12, 5))
#     # for i, successful_requests in enumerate(all_successful_requests):
#     #     accuracies = [req["accuracy"] for req in successful_requests]
#     #     avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
#     #     avg_accuracies.append(avg_accuracy)

#     #     smoothed = moving_average(accuracies, ma_window)
#     #     x_vals = list(range(len(smoothed)))

#     #     label = (model_names[i] if model_names else f"Model {i+1}") + f" (Avg: {avg_accuracy:.3f})"
#     #     color = colors[i % len(colors)]
#     #     plt.plot(x_vals, smoothed, linestyle='-', color=color, alpha=0.8, label=label)

#     # plt.title(f"Smoothed BERTScore Accuracy Over Time (Window = {ma_window})")
#     # plt.xlabel("Request Index")
#     # plt.ylabel("Moving Avg Accuracy (F1 Score)")
#     # plt.ylim(0, 1.0)
#     # plt.grid(True)
#     # plt.legend()
#     # plt.tight_layout()
#     # plt.savefig("combined_smoothed_accuracy.png")
#     # plt.close()
#     # print("Saved smoothed accuracy plot to combined_smoothed_accuracy.png")

#     # 3) Print average accuracies — COMMENTED OUT
#     # print("\n=== Average Accuracies ===")
#     # for i, avg in enumerate(avg_accuracies):
#     #     label = model_names[i] if model_names else f"Model {i+1}"
#     #     print(f"{label}: {avg:.4f}")

# # def plot_static_batching_epochs(bloom3b_epochs, bloom7b1_epochs):
# #     # Ensure both lists have the same length (pad with zeros if needed)
# #     max_len = max(len(bloom3b_epochs), len(bloom7b1_epochs))
    
# #     bloom3b_epochs = list(bloom3b_epochs) + [0] * (max_len - len(bloom3b_epochs))
# #     bloom7b1_epochs = list(bloom7b1_epochs) + [0] * (max_len - len(bloom7b1_epochs))
    
# #     x_epochs = np.arange(max_len)  # epoch indices starting at 0

# #     plt.figure(figsize=(8, 5))
# #     plt.plot(x_epochs, bloom3b_epochs, 'o-', label='Static Batching, BLOOM-3B', color='orange')
# #     plt.plot(x_epochs, bloom7b1_epochs, 's--', label='Static Batching, BLOOM-7.1B', color='blue')

# #     plt.xlabel("Epoch index")
# #     plt.ylabel("Number of completed requests")
# #     plt.title("Static Batching: BLOOM-3B vs BLOOM-7.1B")
# #     plt.xticks(x_epochs)  # show every epoch tick explicitly
# #     plt.grid(True)
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.savefig("static_batching_bloom3b_vs_7b1pt1.png")
# #     plt.show()


