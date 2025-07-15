from utils import *
from avl import AVLTree
import random
import matplotlib.pyplot as plt

def plot_current(reqs):

    sizes = [a.output_length + a.output_length*a.output_length for a in reqs]
    paper_2_times = [a.latency for a in reqs]
    epochs = [a.epoch for a in reqs]  

    fig = plt.figure(figsize=(12, 6))

    scatter = plt.scatter(
        sizes,
        paper_2_times,
        c=epochs,
        cmap='viridis',
        s=50,
        edgecolor='k'
    )

    plt.colorbar(scatter, label='Depth')
    plt.title("Prompt Scheduling split into epochs")
    plt.xlabel("f(s, n)")
    plt.ylabel("t")
    plt.grid(True)
    plt.show()


def opt_sol(reqs: list[Request]):

    ret = []
    last_reqs = len(reqs) + 1
    i = 0

    while reqs:
        i += 1

        print(f"Remaining requests: {len(reqs)}", last_reqs - len(reqs))
        if last_reqs - len(reqs) == 0:
            print("No more requests can be removed, breaking out of the loop.")
            for request in reqs:
                request.epoch = i+1
                ret.append(request)
            break
        last_reqs = len(reqs)

        best_solution = []
        best_solution_bandwidth = 0

        highest_output_length = 0
        output_len_tree = AVLTree()
        reqs.sort(key=lambda x: x.latency, reverse=True)

        smallest_output = float('inf')
        for request in reqs:
            tau_min = request.latency
            highest_output_length = max(highest_output_length, request.output_length)
            output_len_tree.insert(request)

            # if request.output_length < smallest_output or request == reqs[-1]:
            smallest_output = request.output_length
            lo = 0
            hi = highest_output_length + 1
            while lo < hi:
                mid = (lo + hi) // 2
                if output_len_tree.get_bandwidth_sum_total_less_than(Request.get_bandwidth_from_output_length(mid)) < tau_min:
                    lo = mid + 1
                else:
                    hi = mid

            total_bandwidth = output_len_tree.get_bandwidth_sum_total_less_than(Request.get_bandwidth_from_output_length(mid-1))
            output_set = output_len_tree.get_all_less_than(mid)
            if total_bandwidth <= tau_min and len(output_set) > len(best_solution):
                best_solution_bandwidth = total_bandwidth
                best_solution = output_set
                # print(best_solution)

        for request in best_solution:
            request.epoch = i
            ret.append(request)
        best_solution_set = set(best_solution)
        reqs = [r for r in reqs if r not in best_solution_set]
        
    return ret


if __name__ == "__main__":
    random.seed(87)  # For reproducibility
    plot = []
    for i in range(1):
        size = 1600
        requests = [
            Request(i, random.randint(1, 150), random.randint(1, 150), random.randint(20000, 90000), random.randint(1, 150))
            for i in range(size)
        ]
        # requests = [
        #     Request(0, 0, 100, 90000, 1),
        # ]
        plot.extend(opt_sol(requests))
    plot_current(plot)
