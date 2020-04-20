from src.nsga2.nsga2 import nsga2
from src.nsga2.population import get_C, get_gamma, get_selected_features
from src.metrics import statistical_parity_difference, function_name_to_string
from src.data import load_german_dataset
from src.exploring.full_algorithms import svm
from src.util.filehandler import write_result_to_file

NUM_GENERATIONS = 10
POPULATION_SIZE = 20
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.7
CHROMOSOME_LENGTH = 30 + 57  # 15 each for C and gamma, plus 57 for the number of features in german data set
FAIRNESS_METRIC = statistical_parity_difference
DATA_SET = load_german_dataset()
PRIVILEGED_GROUPS = [{'age': 1}]
UNPRIVILEGED_GROUPS = [{'age': 0}]


def testing_experiment():
    """
    Baseline SVM experiment.

    :return: Resulting Pareto front
    """
    result = nsga2(pop_size=POPULATION_SIZE,
                   num_generations=NUM_GENERATIONS,
                   chromosome_length=CHROMOSOME_LENGTH,
                   crossover_rate=CROSSOVER_RATE,
                   mutation_rate=MUTATION_RATE,
                   evaluation_algorithm=evaluation_function)

    # Summary to write to file
    result_summary = {'name': 'SVM',
                      'result': result,
                      'fairness_metric': function_name_to_string(FAIRNESS_METRIC),
                      'nsga2_parameters': {'num_generations': NUM_GENERATIONS, 'population_size': POPULATION_SIZE,
                                           'crossover_rate': CROSSOVER_RATE, 'mutation_rate': MUTATION_RATE,
                                           'chromosome_length': CHROMOSOME_LENGTH}}
    write_result_to_file(result_summary, "testing")
    # Return only the result, not the summary
    return result


def evaluation_function(chromosome):
    """
    Function the be used to evaluate the NSGA-II chromosomes and return fitness scores.
    Contains the baseline svm algorithm, that returns the fitness scores for the specified metrics.

    :param chromosome: Chromosome to specify parameters for SVM
    :return: The fitness scores in a list: [accuracy_score, fairness_score]
    """
    # Check if scores already calculated for identical chromosome, in which case return those scores
    if str(chromosome) in FITNESS_SCORES:
        return FITNESS_SCORES[str(chromosome)]
    else:
        C = get_C(chromosome)
        gamma = get_gamma(chromosome)
        selected_features = get_selected_features(chromosome, 30)
        accuracy_score, fairness_score = svm(dataset=DATA_SET, fairness_metric=FAIRNESS_METRIC,
                                             C=C, gamma=gamma, keep_features=selected_features,
                                             privileged_groups=PRIVILEGED_GROUPS,
                                             unprivileged_groups=UNPRIVILEGED_GROUPS)
        FITNESS_SCORES[str(chromosome)] = [accuracy_score, fairness_score]
        return [accuracy_score, fairness_score]


""" 
Collects scores already calculated to remove unnecessary burden of recalculating for identical chromosomes
"""
FITNESS_SCORES = {}

