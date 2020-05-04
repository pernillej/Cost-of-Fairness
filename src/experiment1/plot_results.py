from src.util.filehandler import read_result_from_file
from src.util.plotter import plot_results


def plot(filenames):
    if len(filenames) <= 0:
        return
    results = []
    for filename in filenames:
        try:
            result = read_result_from_file(filename)
            results.append(result)
        except FileNotFoundError:
            print('Invalid filename: ' + filename)
    if len(results) > 0:
        plot_results(results)


plot([])
