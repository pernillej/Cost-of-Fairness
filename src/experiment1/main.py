from src.experiment1.reweighing import svm_reweighing_experiment
from src.experiment1.baseline import svm_experiment
from src.experiment1.disparate_impact_remover import svm_dir_experiment
from src.experiment1.optimpreproc import svm_optimpreproc_experiment
from src.data import load_compas_dataset, load_optimpreproc_compas_dataset
from src.experiment1.config import theil_config, statistical_parity_config

"""
Default Configuration.
"""
CHROMOSOME_LENGTH = 30 + 12  # 15 each for C and gamma, + 12 for the number of features in compas data set
OPTIM_PREPROC_CHROMOSOME_LENGTH = 30 + 10  # 15 each for C and gamma, + 10 for the num features in preproc compas data
TRAINING_DATA, TEST_DATA = load_compas_dataset()
OPTIM_PREPROC_TRAINING_DATA, OPTIM_PREPROC_TEST_DATA = load_optimpreproc_compas_dataset()
PRIVILEGED_GROUPS = [{'race': 1}]
UNPRIVILEGED_GROUPS = [{'race': 0}]


def run_experiments(config):
    num_generations = config["num_generations"]
    pop_size = config["pop_size"]
    mutation_rate = config["mutation_rate"]
    crossover_rate = config["crossover_rate"]
    svm_max_iter = config["svm_max_iter"]
    svm_seed = config["svm_seed"]
    fairness_metric = config["fairness_metric"]
    accuracy_metric = config["accuracy_metric"]

    print("Running SVM")
    result = svm_experiment(num_generations=num_generations, population_size=pop_size,
                            mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                            chromosome_length=CHROMOSOME_LENGTH, fairness_metric=fairness_metric,
                            accuracy_metric=accuracy_metric, training_data=TRAINING_DATA, test_data=TEST_DATA,
                            privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
                            max_iter=svm_max_iter, svm_seed=svm_seed)
    print('Results: ' + str(result))
    print("Running SVM with Reweighing")
    result = svm_reweighing_experiment(num_generations=num_generations, population_size=pop_size,
                                       mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                                       chromosome_length=CHROMOSOME_LENGTH, fairness_metric=fairness_metric,
                                       accuracy_metric=accuracy_metric, training_data=TRAINING_DATA,
                                       test_data=TEST_DATA, privileged_groups=PRIVILEGED_GROUPS,
                                       unprivileged_groups=UNPRIVILEGED_GROUPS, max_iter=svm_max_iter,
                                       svm_seed=svm_seed)
    print('Results: ' + str(result))
    print("Running SVM with DisparateImpactRemover")
    result = svm_dir_experiment(num_generations=num_generations, population_size=pop_size,
                                mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                                chromosome_length=CHROMOSOME_LENGTH, fairness_metric=fairness_metric,
                                accuracy_metric=accuracy_metric, training_data=TRAINING_DATA, test_data=TEST_DATA,
                                privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
                                max_iter=svm_max_iter, svm_seed=svm_seed)
    print('Results: ' + str(result))
    print("Running SVM with Optimized Preprocessing")
    result = svm_optimpreproc_experiment(num_generations=num_generations, population_size=pop_size,
                                         mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                                         chromosome_length=OPTIM_PREPROC_CHROMOSOME_LENGTH,
                                         fairness_metric=fairness_metric, accuracy_metric=accuracy_metric,
                                         training_data=OPTIM_PREPROC_TRAINING_DATA, test_data=OPTIM_PREPROC_TEST_DATA,
                                         privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
                                         max_iter=svm_max_iter, svm_seed=svm_seed)
    print('Results: ' + str(result))


run_experiments(statistical_parity_config)
