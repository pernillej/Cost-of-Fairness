from src.data import load_compas_dataset, load_optimpreproc_compas_dataset
from src.experiment3.baseline import svm_experiment
from src.experiment3.reweighing import svm_reweighing_experiment
from src.experiment3.disparate_impact_remover import svm_dir_experiment
from src.experiment3.optimpreproc import svm_optimpreproc_experiment
from src.experiment3.config import theil_config, statistical_parity_config

"""
Configuration
"""
CHROMOSOME_LENGTH = 20  # 10 for each of the classification thresholds
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
    svm_fair = config['classifiers']["svm_fair"]
    svm_acc = config['classifiers']["svm_acc"]
    svm_reweighing_fair = config['classifiers']["svm_rw_fair"]
    svm_reweighing_acc = config['classifiers']["svm_rw_acc"]
    svm_dir_fair = config['classifiers']["svm_dir_fair"]
    svm_dir_acc = config['classifiers']["svm_dir_acc"]
    svm_optimpreproc_fair = config['classifiers']["svm_opp_fair"]
    svm_optimpreproc_acc = config['classifiers']["svm_opp_acc"]

    print("Running the fair SVM classifier")
    result = svm_experiment(classifier_chromosome=svm_fair, num_generations=num_generations, population_size=pop_size,
                            mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                            chromosome_length=CHROMOSOME_LENGTH, fairness_metric=fairness_metric,
                            accuracy_metric=accuracy_metric, training_data=TRAINING_DATA, test_data=TEST_DATA,
                            privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
                            max_iter=svm_max_iter, svm_seed=svm_seed, name_postfix='fair')
    print('Results: ' + str(result))
    print("Running the accurate SVM classifier")
    result = svm_experiment(classifier_chromosome=svm_acc, num_generations=num_generations, population_size=pop_size,
                            mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                            chromosome_length=CHROMOSOME_LENGTH, fairness_metric=fairness_metric,
                            accuracy_metric=accuracy_metric, training_data=TRAINING_DATA, test_data=TEST_DATA,
                            privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
                            max_iter=svm_max_iter, svm_seed=svm_seed, name_postfix='acc')
    print('Results: ' + str(result))
    print("Running the fair SVM with Reweighing")
    result = svm_reweighing_experiment(classifier_chromosome=svm_reweighing_fair, num_generations=num_generations,
                                       population_size=pop_size, mutation_rate=mutation_rate,
                                       crossover_rate=crossover_rate, chromosome_length=CHROMOSOME_LENGTH,
                                       fairness_metric=fairness_metric, accuracy_metric=accuracy_metric,
                                       training_data=TRAINING_DATA, test_data=TEST_DATA,
                                       privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
                                       max_iter=svm_max_iter, svm_seed=svm_seed, name_postfix='fair')
    print('Results: ' + str(result))
    print("Running the accurate SVM with Reweighing")
    result = svm_reweighing_experiment(classifier_chromosome=svm_reweighing_acc, num_generations=num_generations,
                                       population_size=pop_size, mutation_rate=mutation_rate,
                                       crossover_rate=crossover_rate, chromosome_length=CHROMOSOME_LENGTH,
                                       fairness_metric=fairness_metric, accuracy_metric=accuracy_metric,
                                       training_data=TRAINING_DATA, test_data=TEST_DATA,
                                       privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
                                       max_iter=svm_max_iter, svm_seed=svm_seed, name_postfix='acc')
    print('Results: ' + str(result))
    print("Running the fair SVM with DisparateImpactRemover")
    result = svm_dir_experiment(classifier_chromosome=svm_dir_fair, num_generations=num_generations,
                                population_size=pop_size, mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                                chromosome_length=CHROMOSOME_LENGTH, fairness_metric=fairness_metric,
                                accuracy_metric=accuracy_metric, training_data=TRAINING_DATA, test_data=TEST_DATA,
                                privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
                                max_iter=svm_max_iter, svm_seed=svm_seed, name_postfix='fair')
    print('Results: ' + str(result))
    print("Running the accurate SVM with DisparateImpactRemover")
    result = svm_dir_experiment(classifier_chromosome=svm_dir_acc, num_generations=num_generations,
                                population_size=pop_size, mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                                chromosome_length=CHROMOSOME_LENGTH, fairness_metric=fairness_metric,
                                accuracy_metric=accuracy_metric, training_data=TRAINING_DATA, test_data=TEST_DATA,
                                privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
                                max_iter=svm_max_iter, svm_seed=svm_seed, name_postfix='acc')
    print('Results: ' + str(result))
    print("Running the fair SVM with Optimized Preprocessing")
    result = svm_optimpreproc_experiment(classifier_chromosome=svm_optimpreproc_fair, num_generations=num_generations,
                                         population_size=pop_size, mutation_rate=mutation_rate,
                                         crossover_rate=crossover_rate, chromosome_length=CHROMOSOME_LENGTH,
                                         fairness_metric=fairness_metric, accuracy_metric=accuracy_metric,
                                         training_data=TRAINING_DATA, test_data=TEST_DATA,
                                         privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
                                         max_iter=svm_max_iter, svm_seed=svm_seed, name_postfix='fair')
    print('Results: ' + str(result))
    print("Running the accurate SVM with Optimized Preprocessing")
    result = svm_optimpreproc_experiment(classifier_chromosome=svm_optimpreproc_acc, num_generations=num_generations,
                                         population_size=pop_size, mutation_rate=mutation_rate,
                                         crossover_rate=crossover_rate, chromosome_length=CHROMOSOME_LENGTH,
                                         fairness_metric=fairness_metric, accuracy_metric=accuracy_metric,
                                         training_data=TRAINING_DATA, test_data=TEST_DATA,
                                         privileged_groups=PRIVILEGED_GROUPS, unprivileged_groups=UNPRIVILEGED_GROUPS,
                                         max_iter=svm_max_iter, svm_seed=svm_seed, name_postfix='acc')
    print('Results: ' + str(result))


run_experiments(statistical_parity_config)
run_experiments(theil_config)
