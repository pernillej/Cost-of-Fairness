from src.experiment4.reweighing import svm_reweighing_experiment
from src.experiment4.baseline import svm_experiment
from src.experiment4.disparate_impact_remover import svm_dir_experiment
from src.experiment4.optimpreproc import svm_optimpreproc_experiment
from src.util.plotter import plot_results
from src.util.filehandler import read_result_from_file
from src.data import load_compas_dataset, load_optimpreproc_compas_dataset
from src.metrics import statistical_parity_difference, theil_index, binary_accuracy

"""
Configuration
"""

NUM_GENERATIONS = 100
POPULATION_SIZE = 50
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.7
CHROMOSOME_LENGTH = 30 + 12  # 15 each for C and gamma, + 12 for the number of features in compas data set
OPTIM_PREPROC_CHROMOSOME_LENGTH = 30 + 10  # 15 each for C and gamma, + 10 for the num features in preproc compas data
TRAINING_DATA, TEST_DATA = load_compas_dataset()
OPTIM_PREPROC_TRAINING_DATA, OPTIM_PREPROC_TEST_DATA = load_optimpreproc_compas_dataset()
SVM_MAX_ITER = 10000
SVM_SEED = 0
PRIVILEGED_GROUPS = [{'race': 1}]
UNPRIVILEGED_GROUPS = [{'race': 0}]
FAIRNESS_METRIC = statistical_parity_difference
ACCURACY_METRIC = binary_accuracy


def run_experiments():
    print("Running SVM")
    result = svm_experiment(num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE,
                            mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                            chromosome_length=CHROMOSOME_LENGTH, fairness_metric=FAIRNESS_METRIC,
                            accuracy_metric=ACCURACY_METRIC, training_data=TRAINING_DATA, test_data=TEST_DATA,
                            privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
                            max_iter=SVM_MAX_ITER, svm_seed=SVM_SEED)
    print('Results: ' + str(result))

    print("Running SVM with Reweighing")
    result = svm_reweighing_experiment(num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE,
                                       mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                                       chromosome_length=CHROMOSOME_LENGTH, fairness_metric=FAIRNESS_METRIC,
                                       accuracy_metric=ACCURACY_METRIC, training_data=TRAINING_DATA,
                                       test_data=TEST_DATA, privileged_groups=PRIVILEGED_GROUPS,
                                       unprivileged_groups=UNPRIVILEGED_GROUPS, max_iter=SVM_MAX_ITER,
                                       svm_seed=SVM_SEED)
    print('Results: ' + str(result))
    print("Running SVM with DisparateImpactRemover")
    result = svm_dir_experiment(num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE,
                                mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                                chromosome_length=CHROMOSOME_LENGTH, fairness_metric=FAIRNESS_METRIC,
                                accuracy_metric=ACCURACY_METRIC, training_data=TRAINING_DATA, test_data=TEST_DATA,
                                privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
                                max_iter=SVM_MAX_ITER, svm_seed=SVM_SEED)
    print('Results: ' + str(result))
    print("Running SVM with Optimized Preprocessing")
    result = svm_optimpreproc_experiment(num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE,
                                         mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                                         chromosome_length=OPTIM_PREPROC_CHROMOSOME_LENGTH,
                                         fairness_metric=FAIRNESS_METRIC, accuracy_metric=ACCURACY_METRIC,
                                         training_data=OPTIM_PREPROC_TRAINING_DATA, test_data=OPTIM_PREPROC_TEST_DATA,
                                         privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
                                         max_iter=SVM_MAX_ITER, svm_seed=SVM_SEED)
    print('Results: ' + str(result))


def plot():
    svm_results = read_result_from_file('svm_22-04-2020_15-59.txt')
    # reweighing_results = read_result_from_file('svm_reweighing_21-04-2020_11-07.txt')
    # dir_results = read_result_from_file('svm_dir_21-04-2020_11-27.txt')
    # optimpreproc_results = read_result_from_file('svm_optimpreproc_21-04-2020_11-29.txt')

    # plot_results([svm_results, reweighing_results, dir_results, optimpreproc_results])
    plot_results([svm_results])


run_experiments()
# plot()

