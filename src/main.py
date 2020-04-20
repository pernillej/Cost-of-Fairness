from src.experiments.reweighing import svm_reweighing_experiment
from src.experiments.roc import svm_roc_experiment
from src.experiments.baseline import svm_experiment
from src.experiments.calibrated_eq_odds import svm_caleqodds_experiment
from src.experiments.disparate_impact_remover import svm_dir_experiment
from src.experiments.optimpreproc import svm_optimpreproc_experiment
from src.util.plotter import plot_results
from src.util.filehandler import read_result_from_file
from src.data import load_german_dataset
from src.metrics import statistical_parity_difference, theil_index, disparate_impact, average_odds_difference, \
    equal_opportunity_difference

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_german

"""
Done: 
1. Collect the dataset(s)
2. Setup SVM with 5-fold testing.
3. Add basic metrics
4. Setup NSGA2
5. Add saving and plotting of results
6. Add out mitigation methods to svm

TODO:
Customize...
"""

"""
Configuration
"""

NUM_GENERATIONS = 100
POPULATION_SIZE = 50
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.7
CHROMOSOME_LENGTH = 30 + 57  # 15 each for C and gamma, + 57 for the number of features in german data set
OPTIM_PREPROC_CHROMOSOME_LENGTH = 30 + 11  # 15 each for C and gamma, + 11 for the num features in preproc german data
DATA_SET = load_german_dataset()
OPTIM_PREPROC_DATA_SET = load_preproc_data_german()
PRIVILEGED_GROUPS = [{'age': 1}]
UNPRIVILEGED_GROUPS = [{'age': 0}]
FAIRNESS_METRIC = statistical_parity_difference


def run_experiment():
    """
    print("Running SVM")
    result = svm_experiment(num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE,
                            mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                            chromosome_length=CHROMOSOME_LENGTH, fairness_metric=FAIRNESS_METRIC,
                            data_set=DATA_SET, privileged_groups=PRIVILEGED_GROUPS,
                            unprivileged_groups=UNPRIVILEGED_GROUPS)
    print('Results: ' + str(result))
    print("Running SVM with Reweighing")
    result = svm_reweighing_experiment(num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE,
                                       mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                                       chromosome_length=CHROMOSOME_LENGTH, fairness_metric=FAIRNESS_METRIC,
                                       data_set=DATA_SET, privileged_groups=PRIVILEGED_GROUPS,
                                       unprivileged_groups=UNPRIVILEGED_GROUPS)
    print('Results: ' + str(result))
    print("Running SVM with DisparateImpactRemover")
    result = svm_dir_experiment(num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE,
                                mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                                chromosome_length=CHROMOSOME_LENGTH, fairness_metric=FAIRNESS_METRIC,
                                data_set=DATA_SET, privileged_groups=PRIVILEGED_GROUPS,
                                unprivileged_groups=UNPRIVILEGED_GROUPS)
    print('Results: ' + str(result))
    """
    print("Running SVM with Optimized Preprocessing")
    result = svm_optimpreproc_experiment(num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE,
                                         mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                                         chromosome_length=OPTIM_PREPROC_CHROMOSOME_LENGTH,
                                         fairness_metric=FAIRNESS_METRIC,
                                         data_set=OPTIM_PREPROC_DATA_SET, privileged_groups=PRIVILEGED_GROUPS,
                                         unprivileged_groups=UNPRIVILEGED_GROUPS)
    print('Results: ' + str(result))


def plot():
    svm_results = read_result_from_file('svm_18-04-2020_17-17.txt')
    reweighing_results = read_result_from_file('svm_reweighing_18-04-2020_17-25.txt')
    dir_results = read_result_from_file('svm_dir_18-04-2020_19-43.txt')
    caleqodds_results = read_result_from_file('svm_caleqodds_19-04-2020_06-38.txt')

    plot_results([svm_results, reweighing_results, dir_results, caleqodds_results])


run_experiment()
# plot()

