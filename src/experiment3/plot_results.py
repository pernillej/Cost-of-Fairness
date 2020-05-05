from src.util.filehandler import read_result_from_file
from src.util.plotter import plot_results

# The best front from each algorithm using the statistical parity difference metric on the fair classifiers
FAIR_STATISTICAL_PARITY_RESULTS = ['svm_fair_05-05-2020_11-41.txt', 'svm_reweighing_fair_05-05-2020_11-49.txt',
                                   'svm_dir_fair_05-05-2020_11-56.txt', 'svm_optimpreproc_fair_05-05-2020_12-03.txt']
# The best front from each algorithm using the statistical parity difference metric on the accurate classifiers
ACC_STATISTICAL_PARITY_RESULTS = ['svm_acc_05-05-2020_11-44.txt', 'svm_reweighing_acc_05-05-2020_11-53.txt',
                                  'svm_dir_acc_05-05-2020_11-59.txt', 'svm_optimpreproc_acc_05-05-2020_12-07.txt']

# The best front from each algorithm using the theil index metric on the fair classifiers
FAIR_THEIL_RESULTS = ['svm_fair_05-05-2020_12-10.txt', 'svm_reweighing_fair_05-05-2020_12-16.txt',
                      'svm_dir_fair_05-05-2020_12-20.txt', 'svm_optimpreproc_fair_05-05-2020_12-27.txt']
# The best front from each algorithm using the theil index metric on the accurate classifiers
ACC_THEIL_RESULTS = ['svm_acc_05-05-2020_12-14.txt', 'svm_reweighing_acc_05-05-2020_12-20.txt',
                     'svm_dir_acc_05-05-2020_12-25.txt', 'svm_optimpreproc_acc_05-05-2020_12-30.txt']


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
plot(ACC_STATISTICAL_PARITY_RESULTS)
plot(FAIR_THEIL_RESULTS)
plot(ACC_THEIL_RESULTS)
