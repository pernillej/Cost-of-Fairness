from src.nsga2.nsga2 import nsga2
from src.nsga2.population import get_C, get_gamma
from src.metrics import function_name_to_string
from src.experiment1.algorithms import svm_optimpreproc
from src.util.filehandler import write_result_to_file

""" 
Collects scores already calculated to remove unnecessary burden of recalculating for identical chromosomes
"""
FITNESS_SCORES = {}


def svm_optimpreproc_experiment(num_generations, population_size, mutation_rate, crossover_rate, chromosome_length,
                                fairness_metric, data_set, privileged_groups, unprivileged_groups):
    """
    SVM with Optimized Preprocessing experiment.

    :param num_generations: Number of generations for NSGA-II
    :param population_size: Population size for NSGA-II
    :param mutation_rate: Mutation rate for NSGA-II
    :param crossover_rate: Crossover probability for NSGA-II
    :param chromosome_length: Length of chromosome used in NSGA-II
    :param fairness_metric: Fairness metric used to calculate the fairness score
    :param data_set: The data set to run experiment on
    :param privileged_groups: The privileged groups in the data set
    :param unprivileged_groups: The unprivileged groups in the data set
    :return: Resulting Pareto front
    """

    # Defines the evaluation function
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
            accuracy_score, fairness_score = svm_optimpreproc(dataset=data_set, fairness_metric=fairness_metric,
                                                              C=C, gamma=gamma, privileged_groups=privileged_groups,
                                                              unprivileged_groups=unprivileged_groups)
            FITNESS_SCORES[str(chromosome)] = [accuracy_score, fairness_score]
            return [accuracy_score, fairness_score]

    # Runs NSGA-II
    result = nsga2(pop_size=population_size,
                   num_generations=num_generations,
                   chromosome_length=chromosome_length,
                   crossover_rate=crossover_rate,
                   mutation_rate=mutation_rate,
                   evaluation_algorithm=evaluation_function)

    # Writes summary to file
    result_summary = {'name': 'SVM_OptimPreProc',
                      'result': result,
                      'fairness_metric': function_name_to_string(fairness_metric),
                      'nsga2_parameters': {'num_generations': num_generations, 'population_size': population_size,
                                           'crossover_rate': crossover_rate, 'mutation_rate': mutation_rate,
                                           'chromosome_length': chromosome_length}}
    write_result_to_file(result_summary, "svm_optimpreproc")
    # Return only the result, not the summary
    return result
