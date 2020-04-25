from src.util.plotter import plot_results
from src.util.filehandler import read_result_from_file
from src.data import load_compas_dataset, load_optimpreproc_compas_dataset
from src.metrics import statistical_parity_difference, binary_accuracy
from src.experiment2.baseline import svm_experiment
from src.experiment2.disparate_impact_remover import svm_dir_experiment
from src.experiment2.optimpreproc import svm_optimpreproc_experiment
from src.experiment2.reweighing import svm_reweighing_experiment

"""
Configuration
"""
"""
TODO: Replace with proper values
"""
SVM_C = 0.1
SVM_GAMMA = 0.1
SVM_SELECTED_FEATURES = [0, 1, 2, 3, 4, 5]
SVM_REWEIGHING_C = 0.1
SVM_REWEIGHING_GAMMA = 0.1
SVM_REWEIGHING_SELECTED_FEATURES = [0, 1, 2, 3, 4, 5]
SVM_DIR_C = 0.1
SVM_DIR_GAMMA = 0.1
SVM_DIR_SELECTED_FEATURES = [0, 1, 2, 3, 4, 5]
SVM_OPTIMPREPROC_C = 0.1
SVM_OPTIMPREPROC_GAMMA = 0.1
SVM_OPTIMPREPROC_SELECTED_FEATURES = [0, 1, 2, 3, 4, 5]

NUM_GENERATIONS = 100
POPULATION_SIZE = 50
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.7
CHROMOSOME_LENGTH = 10  # 15 for the classification threshold
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
    result = svm_experiment(C=SVM_C, gamma=SVM_GAMMA, selected_features=SVM_SELECTED_FEATURES,
                            num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE,
                            mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                            chromosome_length=CHROMOSOME_LENGTH, fairness_metric=FAIRNESS_METRIC,
                            accuracy_metric=ACCURACY_METRIC, training_data=TRAINING_DATA, test_data=TEST_DATA,
                            privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
                            max_iter=SVM_MAX_ITER, svm_seed=SVM_SEED)
    print('Results: ' + str(result))
    print("Running SVM with Reweighing")
    result = svm_reweighing_experiment(C=SVM_REWEIGHING_C, gamma=SVM_REWEIGHING_GAMMA,
                                       selected_features=SVM_REWEIGHING_SELECTED_FEATURES,
                                       num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE,
                                       mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                                       chromosome_length=CHROMOSOME_LENGTH, fairness_metric=FAIRNESS_METRIC,
                                       accuracy_metric=ACCURACY_METRIC, training_data=TRAINING_DATA,
                                       test_data=TEST_DATA, privileged_groups=PRIVILEGED_GROUPS,
                                       unprivileged_groups=UNPRIVILEGED_GROUPS, max_iter=SVM_MAX_ITER, svm_seed=SVM_SEED)
    print('Results: ' + str(result))
    print("Running SVM with DisparateImpactRemover")
    result = svm_dir_experiment(C=SVM_DIR_C, gamma=SVM_DIR_GAMMA, selected_features=SVM_DIR_SELECTED_FEATURES,
                                num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE,
                                mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                                chromosome_length=CHROMOSOME_LENGTH, fairness_metric=FAIRNESS_METRIC,
                                accuracy_metric=ACCURACY_METRIC, training_data=TRAINING_DATA, test_data=TEST_DATA,
                                privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
                                max_iter=SVM_MAX_ITER, svm_seed=SVM_SEED)
    print('Results: ' + str(result))
    print("Running SVM with Optimized Preprocessing")
    result = svm_optimpreproc_experiment(C=SVM_OPTIMPREPROC_C, gamma=SVM_OPTIMPREPROC_GAMMA,
                                         selected_features=SVM_OPTIMPREPROC_SELECTED_FEATURES,
                                         num_generations=NUM_GENERATIONS, population_size=POPULATION_SIZE,
                                         mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
                                         chromosome_length=CHROMOSOME_LENGTH, fairness_metric=FAIRNESS_METRIC,
                                         accuracy_metric=ACCURACY_METRIC, training_data=TRAINING_DATA,
                                         test_data=TEST_DATA, privileged_groups=PRIVILEGED_GROUPS,
                                         unprivileged_groups=UNPRIVILEGED_GROUPS, max_iter=SVM_MAX_ITER,
                                         svm_seed=SVM_SEED)
    print('Results: ' + str(result))


def plot():
    svm_results = read_result_from_file('svm_25-04-2020_12-18.txt')
    reweighing_results = read_result_from_file('svm_reweighing_25-04-2020_12-19.txt')
    dir_results = read_result_from_file('svm_dir_25-04-2020_12-20.txt')
    optimpreproc_results = read_result_from_file('svm_optimpreproc_25-04-2020_12-21.txt')

    plot_results([svm_results, reweighing_results, dir_results, optimpreproc_results])


run_experiments()
# plot()
