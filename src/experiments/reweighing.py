from src.nsga2.nsga2 import nsga2
from src.nsga2.population import get_C, get_gamma, get_selected_features
from src.metrics import auc, statistical_parity, theil_index, function_name_to_string
from src.data import load_german_dataframe, get_drop_features
from src.algorithms import svm_reweighing
from src.util.filehandler import write_result_to_file

NUM_GENERATIONS = 2
POPULATION_SIZE = 40
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.7
CHROMOSOME_LENGTH = 30 + 57  # 15 each for C and gamma, plus 57 for the number of features in german data set
METRICS = {  # Statistical parity seems to keep getting 1 as fairness score always?
    'accuracy': auc,
    'fairness': statistical_parity
}
# Dataframe and attributes
DF, DF_ATTRIBUTES = load_german_dataframe()


def reweighing_experiment():
    """
    Reweighing with SVM experiment.

    :return: Resulting Pareto front
    """
    result = nsga2(pop_size=POPULATION_SIZE,
                   num_generations=NUM_GENERATIONS,
                   chromosome_length=CHROMOSOME_LENGTH,
                   crossover_rate=CROSSOVER_RATE,
                   mutation_rate=MUTATION_RATE,
                   evaluation_algorithm=evaluation_function)

    # Summary to write to file
    result_summary = {'name': 'SVM_Reweighing',
                      'result': result,
                      'metrics': {'accuracy': function_name_to_string(METRICS['accuracy']),
                                  'fairness': function_name_to_string(METRICS['fairness'])},
                      'nsga2_parameters': {'num_generations': NUM_GENERATIONS, 'population_size': POPULATION_SIZE,
                                           'crossover_rate': CROSSOVER_RATE, 'mutation_rate': MUTATION_RATE,
                                           'chromosome_length': CHROMOSOME_LENGTH}}
    write_result_to_file(result_summary, "svm_reweighing")
    # Return only the result, not the summary
    return result


def evaluation_function(chromosome):
    """
    Function the be used to evaluate the NSGA-II chromosomes and return fitness scores.
    Contains the svm_reweighing algorithm, that returns the fitness scores for the specified metrics.

    :param chromosome: Chromosome to specify parameters for SVM
    :return: The fitness scores in a list: [accuracy_score, fairness_score]
    """
    # Check if scores already calculated for identical chromosome, in which case return those scores
    if str(chromosome) in FITNESS_SCORES:
        return FITNESS_SCORES[str(chromosome)]
    else:
        C = get_C(chromosome)
        gamma = get_gamma(chromosome)
        selected_features = get_selected_features(chromosome)
        drop_features = get_drop_features(DF_ATTRIBUTES['feature_names'], selected_features)
        accuracy_score, fairness_score = svm_reweighing(DF, METRICS, DF_ATTRIBUTES, C=C, gamma=gamma,
                                                        drop_features=drop_features)
        FITNESS_SCORES[str(chromosome)] = [accuracy_score, fairness_score]
        return [accuracy_score, fairness_score]


""" 
Collects scores already calculated to remove unnecessary burden of recalculating for identical chromosomes
"""
FITNESS_SCORES = {}
