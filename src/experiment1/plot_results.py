from src.util.filehandler import read_result_from_file
from src.util.plotter import plot_results

# The best front from each algorithm using the statistical parity difference metric
STATISTICAL_PARITY_RESULTS = ['svm_24-04-2020_10-08.txt', 'svm_reweighing_24-04-2020_23-07.txt',
                              'svm_dir_25-04-2020_23-06.txt', 'svm_optimpreproc_26-04-2020_20-49.txt']

# The best front from each algorithm using the theil index metric
THEIL_RESULTS = ['svm_29-04-2020_18-02.txt', 'svm_reweighing_29-04-2020_20-35.txt', 'svm_dir_29-04-2020_22-40.txt',
                 'svm_optimpreproc_29-04-2020_08-29.txt']


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


plot(THEIL_RESULTS)
