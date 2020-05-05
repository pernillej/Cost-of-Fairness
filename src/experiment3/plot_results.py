from src.util.filehandler import read_result_from_file
from src.util.plotter import plot_results

# The best front from each algorithm using the statistical parity difference metric on the fair classifiers
FAIR_STATISTICAL_PARITY_RESULTS = ['svm_fair_05-05-2020_11-41.txt', 'svm_reweighing_fair_05-05-2020_?-?.txt',
                                   'svm_dir_fair_05-05-2020_?-?.txt', 'svm_optimpreproc_fair_05-05-2020_?-?.txt']
# The best front from each algorithm using the statistical parity difference metric on the accurate classifiers
ACC_STATISTICAL_PARITY_RESULTS = ['svm_acc_05-05-2020_11-44.txt', 'svm_reweighing_acc_05-05-2020_?-?.txt',
                                  'svm_dir_acc_05-05-2020_?-?.txt', 'svm_optimpreproc_acc_05-05-2020_?-?.txt']

# The best front from each algorithm using the theil index metric on the fair classifiers
FAIR_THEIL_RESULTS = ['svm_fair_05-05-2020_?-?.txt', 'svm_reweighing_fair_05-05-2020_?-?.txt',
                      'svm_dir_fair_05-05-2020_?-?.txt', 'svm_optimpreproc_fair_05-05-2020_?-?.txt']
# The best front from each algorithm using the theil index metric on the accurate classifiers
ACC_THEIL_RESULTS = ['svm_acc_05-05-2020_?-?.txt', 'svm_reweighing_acc_05-05-2020_?-?.txt',
                     'svm_dir_acc_05-05-2020_?-?.txt', 'svm_optimpreproc_acc_05-05-2020_?-?.txt']


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


plot(FAIR_STATISTICAL_PARITY_RESULTS)
