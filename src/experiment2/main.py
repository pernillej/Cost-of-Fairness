from src.util.plotter import plot_results
from src.util.filehandler import read_result_from_file
from src.data import load_german_dataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_german
from src.metrics import statistical_parity_difference, theil_index

"""
TODO: Expand experiment 1 to add a classification threshold
"""

"""
Configuration
"""

NUM_GENERATIONS = 10
POPULATION_SIZE = 10
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.7
CHROMOSOME_LENGTH = 30 + 57  # 15 each for C and gamma, + 57 for the number of features in german data set
OPTIM_PREPROC_CHROMOSOME_LENGTH = 30 + 11  # 15 each for C and gamma, + 11 for the num features in preproc german data
DATA_SET = load_german_dataset()
OPTIM_PREPROC_DATA_SET = load_preproc_data_german()
PRIVILEGED_GROUPS = [{'age': 1}]
UNPRIVILEGED_GROUPS = [{'age': 0}]
FAIRNESS_METRIC = statistical_parity_difference


def run_experiments():
    return


def plot():
    svm_results = read_result_from_file('svm_18-04-2020_17-17.txt')
    reweighing_results = read_result_from_file('svm_reweighing_18-04-2020_17-25.txt')
    dir_results = read_result_from_file('svm_dir_18-04-2020_19-43.txt')
    optimpreproc_results = read_result_from_file('svm_optimpreproc_20-04-2020_11-16.txt')

    plot_results([svm_results, reweighing_results, dir_results, optimpreproc_results])