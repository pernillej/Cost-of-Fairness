import matplotlib.pyplot as plt
import numpy as np


def plot_result(result, label, show=False):
    scores = sort_results_for_plotting(list(result.values()))
    scores = np.array(scores)
    x = scores[:, 0]
    y = scores[:, 1]
    line, = plt.plot(x, y, '-o', label=label)
    if show:
        plt.show()
    return line


def plot_results(result_summaries):
    x_label = result_summaries[0]["accuracy_metric"]
    y_label = "1 - " + result_summaries[0]["fairness_metric"]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    lines = []
    for result_summary in result_summaries:
        lines.append(plot_result(result_summary["result"], result_summary["name"]))
    plt.legend(handles=lines)
    plt.show()


def sort_results_for_plotting(scores):
    return sorted(scores, key=lambda x: x[1])
