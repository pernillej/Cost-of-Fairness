from src.nsga2.population import create_population, generate_children, get_population_fitness
import numpy as np
import random


def nsga2(pop_size, num_generations, chromosome_length, mutation_rate, crossover_rate, evaluation_algorithm):
    """
    Run the NSGA-II algorithm with the specified parameters.

    :param pop_size: Number of chromosomes in the population
    :param num_generations: Number of generations to run
    :param chromosome_length: The length of the chromosomes (amount of bits)
    :param mutation_rate: The rate of mutation
    :param crossover_rate: The probability of crossover
    :param evaluation_algorithm: The evaluation algorithm to be used to determine the fairness metrics
    :return: The final best Pareto front after NSGA-II is finished
    """
    population = create_population(pop_size, chromosome_length)

    for gen in range(num_generations):
        if gen % 5 == 0:
            print("Generation " + str(gen))

        # Combine parents and offspring
        population = np.vstack((population, generate_children(population, crossover_rate, mutation_rate)))
        population = np.unique(population, axis=0)  # Only unique chromosomes

        # Score population
        population_fitness = get_population_fitness(population, evaluation_algorithm)

        # Reduce population using Fast-Non-Dominated-Sort and Crowding Distance
        population = select_population(population, scores=population_fitness, population_size=pop_size)

    # Return final Pareto front
    return get_final_front(population, evaluation_algorithm)


"""
Code below from (with updates, adaptations and fixes): 
https://pythonhealthcare.org/2019/01/17/117-genetic-algorithms-2-a-multiple-objective-genetic-algorithm-nsga-ii/
"""


def get_final_front(population, evaluation_algorithm):
    """
    Get the final Pareto front from the final population

    :param population: The final population of chromosomes
    :param evaluation_algorithm: Evaluation algorithm to be used to determine the fairness metrics
    :return: The final Pareto front as a dictionary, with the chromosomes ase keys
    """
    population_fitness = get_population_fitness(population, evaluation_algorithm)
    population_ids = np.arange(population.shape[0]).astype(int)
    pareto_front = identify_pareto(population_fitness, population_ids)
    result = {}
    for i in pareto_front:
        result[str(population[i].tolist()).replace('\n', '')] = population_fitness[i].tolist()
    return result


def calculate_crowding_distances(scores):
    """
    Crowding is based on a vector for each individual
    All scores are normalised between low and high. For any one score, all
    chromosomes are sorted in order low to high. Crowding for chromosome x
    for that score is the difference between the next highest and next
    lowest score. Total crowding value sums all crowding for all scores

    :param scores: The fitness scores the calculate crowding distances from
    :return: The crowding distances
    """

    population_size = len(scores[:, 0])
    number_of_scores = len(scores[0, :])

    # create crowding matrix of population (row) and score (column)
    crowding_matrix = np.zeros((population_size, number_of_scores))

    # normalise scores (ptp is max-min)
    normed_scores = (scores - scores.min(0)) / scores.ptp(0)

    # calculate crowding distance for each score in turn
    for col in range(number_of_scores):
        crowding = np.zeros(population_size)

        # end points have maximum crowding
        crowding[0] = 1
        crowding[population_size - 1] = 1

        # Sort each score (to calculate crowding between adjacent scores)
        sorted_scores = np.sort(normed_scores[:, col])
        sorted_scores_index = np.argsort(
            normed_scores[:, col])

        # Calculate crowding distance for each individual
        crowding[1:population_size - 1] = \
            (sorted_scores[2:population_size] -
             sorted_scores[0:population_size - 2])

        # resort to original order (two steps)
        re_sort_order = np.argsort(sorted_scores_index)
        sorted_crowding = crowding[re_sort_order]

        # Record crowding distances
        crowding_matrix[:, col] = sorted_crowding

    # Sum crowding distances of each score
    crowding_distances = np.sum(crowding_matrix, axis=1)

    return crowding_distances


def tournament_selection(scores, number_to_select):
    """
    This function selects a number of chromosomes based on tournament of
    crowding distances. Two members of the population are picked at
    random. The one with the higher crowding distance is always picked

    :param scores: The fitness scores for the population
    :param number_to_select: Number of chromosomes to select using tournament selection
    :return: Ids indicating which chromosomes are picked
    """
    population_ids = np.arange(scores.shape[0])

    crowding_distances = calculate_crowding_distances(scores)

    picked_population_ids = np.zeros(number_to_select)

    picked_scores = np.zeros((number_to_select, len(scores[0, :])))

    for i in range(number_to_select):

        population_size = population_ids.shape[0]

        fighter1_id = random.randint(0, population_size - 1)

        fighter2_id = random.randint(0, population_size - 1)

        # If fighter # 1 is better
        if crowding_distances[fighter1_id] >= crowding_distances[fighter2_id]:

            # add solution to picked solutions array
            picked_population_ids[i] = population_ids[fighter1_id]

            # Add score to picked scores array
            picked_scores[i, :] = scores[fighter1_id, :]

            # remove selected solution from available solutions
            population_ids = np.delete(population_ids, fighter1_id, axis=0)

            scores = np.delete(scores, fighter1_id, axis=0)

            crowding_distances = np.delete(crowding_distances, fighter1_id, axis=0)
        else:
            picked_population_ids[i] = population_ids[fighter2_id]

            picked_scores[i, :] = scores[fighter2_id, :]

            population_ids = np.delete(population_ids, fighter2_id, axis=0)

            scores = np.delete(scores, fighter2_id, axis=0)

            crowding_distances = np.delete(crowding_distances, fighter2_id, axis=0)

    # Convert to integer
    picked_population_ids = np.asarray(picked_population_ids, dtype=int)

    return picked_population_ids


def identify_pareto(scores, population_ids):
    """
    Identifies a single Pareto front, and returns the population IDs of
    the selected chromosomes.

    :param scores: Fitness scores to find a Pareto front from
    :param population_ids: Ids for the population
    :return: Ids for the selected chromosomes
    """
    population_size = scores.shape[0]
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Pareto front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


def select_population(population, scores, population_size):
    """
    As necessary repeats Pareto front selection to build a population within
    defined size limit. Will reduce a Pareto front by applying crowding
    selection as necessary.

    :param population: The population to reduce
    :param scores: Fitness scores for the population
    :param population_size: The desired size of the reduced population
    :return: The reduced population of chromosomes
    """
    unselected_population_ids = np.arange(population.shape[0])
    all_population_ids = np.arange(population.shape[0])
    pareto_front = []
    while len(pareto_front) < population_size:
        temp_pareto_front = identify_pareto(
            scores[unselected_population_ids, :], unselected_population_ids)
        # Check size of total pareto front.
        # If larger than maximum size reduce new pareto front by crowding
        combined_pareto_size = len(pareto_front) + len(temp_pareto_front)
        if combined_pareto_size > population_size:
            number_to_select = population_size - len(pareto_front)
            selected_individuals = (tournament_selection(
                scores[temp_pareto_front], number_to_select))
            temp_pareto_front = temp_pareto_front[selected_individuals]

        # Add latest pareto front to full Pareto front
        pareto_front = np.hstack((pareto_front, temp_pareto_front))

        # Update unselected population ID by using sets to find IDs in all
        # ids that are not in the selected front
        unselected_set = set(all_population_ids) - set(pareto_front)
        unselected_population_ids = np.array(list(unselected_set))

    population = population[pareto_front.astype(int)]
    return population
