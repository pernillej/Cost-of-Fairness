from src.nsga2.nsga2 import nsga2
from src.nsga2.population import get_tau, get_selected_features
from src.metrics import auc, statistical_parity, theil_index, function_name_to_string
from src.data import load_german_dataframe, get_drop_features
from src.algorithms import meta_fair
from src.util.filehandler import write_result_to_file

NUM_GENERATIONS = 2
POPULATION_SIZE = 2
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.7
CHROMOSOME_LENGTH = 15 + 57  # 15 each for tau, plus 57 for the number of features in german data set
METRICS = {
    'accuracy': auc,
    'fairness': statistical_parity
}
# Dataframe and attributes
DF, DF_ATTRIBUTES = load_german_dataframe()


def metafair_experiment():
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
    result_summary = {'name': 'Meta_Fair',
                      'result': result,
                      'metrics': {'accuracy': function_name_to_string(METRICS['accuracy']),
                                  'fairness': function_name_to_string(METRICS['fairness'])},
                      'nsga2_parameters': {'num_generations': NUM_GENERATIONS, 'population_size': POPULATION_SIZE,
                                           'crossover_rate': CROSSOVER_RATE, 'mutation_rate': MUTATION_RATE,
                                           'chromosome_length': CHROMOSOME_LENGTH}}
    write_result_to_file(result_summary, "meta_fair")
    # Return only the result, not the summary
    return result


def evaluation_function(chromosome):
    """
    Function the be used to evaluate the NSGA-II chromosomes and return fitness scores.
    Contains the meta_fair algorithm, that returns the fitness scores for the specified metrics.

    :param chromosome: Chromosome to specify parameters for MetaFair
    :return: The fitness scores in a list: [accuracy_score, fairness_score]
    """
    # Check if scores already calculated for identical chromosome, in which case return those scores
    if str(chromosome) in FITNESS_SCORES:
        return FITNESS_SCORES[str(chromosome)]
    else:
        tau = get_tau(chromosome)
        selected_features = get_selected_features(chromosome, 15)
        drop_features = get_drop_features(DF_ATTRIBUTES['feature_names'], selected_features)
        if "age" in drop_features:  # Age must be in the features for MetaFair to work
            drop_features.remove('age')
        accuracy_score, fairness_score = meta_fair(DF, METRICS, DF_ATTRIBUTES, tau=tau, drop_features=drop_features)
        FITNESS_SCORES[str(chromosome)] = [accuracy_score, fairness_score]
        return [accuracy_score, fairness_score]


""" 
Collects scores already calculated to remove unnecessary burden of recalculating for identical chromosomes
"""
FITNESS_SCORES = {}
