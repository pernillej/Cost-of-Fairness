import numpy as np
import random
from src.util.binary_conversion import from_binary_to_float_in_range


def create_population(pop_size, chromosome_length):
    """
    Creates a population of chromosomes, where the chromosomes are arrays of bits of a desired length

    :param pop_size: Amount of chromosomes to create
    :param chromosome_length: The length of the chromosome bit arrays
    :return: The population (a list) of chromosomes.
    """
    population = []
    for i in range(pop_size):
        chromosome = np.zeros(chromosome_length)
        n_ones = random.randint(0, chromosome_length)
        chromosome[0: n_ones] = 1
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def crossover(parent1, parent2):
    """
    Perform crossover of 2 parent chromosomes.
    Picks a random point to slice and combine the parents, and returns 2 children.
    Each child contains opposite parts of the parent

    :param parent1: The first parent to perform crossover with
    :param parent2: The second parent to perform crossover with
    :return: 2 child chromosomes
    """
    chromosome_length = len(parent1)
    crossover_point = random.randint(1, chromosome_length - 1)
    child1 = np.hstack((parent1[0:crossover_point], parent2[crossover_point:]))
    child2 = np.hstack((parent2[0:crossover_point], parent1[crossover_point:]))
    return child1, child2


def mutate(chromosome):
    """
    Perform mutation on a chromosome. Flips one random bit in the chromosome.

    :param chromosome: The chromosome to mutate
    :return: The mutated chromosome
    """
    mutated_chromosome = chromosome
    bit_index = random.randint(0, len(chromosome) - 1)
    mutated_chromosome[bit_index] = 1 - chromosome[bit_index]
    return mutated_chromosome


def generate_children(population, crossover_rate, mutation_rate):
    """
    Generate a children population from the parent population.

    :param population: The parent population to generate children from
    :param crossover_rate: The probability of crossover
    :param mutation_rate: The rate of mutation
    :return: List of children chromosomes
    """
    children = []
    pop_size = len(population)

    for i in range(int(pop_size/2)):
        c1 = population[random.randint(0, pop_size-1)]
        c2 = population[random.randint(0, pop_size-1)]
        if random.random() <= crossover_rate:
            c1, c2 = crossover(c1, c2)
        if random.random() <= mutation_rate:
            mutate(c1)
        if random.random() <= mutation_rate:
            mutate(c2)
        children.append(c1)
        children.append(c2)
    return np.array(children)


def get_population_fitness(population, evaluation_algorithm):
    """
    Calculate fitness scores for the population.

    :param population: The population to calculate fitness scores on
    :param evaluation_algorithm: The evaluation algorithm to be used to calculate fitness cores
    :return: The fitness scores
    """
    fitness_scores = np.zeros((population.shape[0], 2))
    for i in range(population.shape[0]):
        fitness = evaluation_algorithm(population[i])
        fitness_scores[i] = fitness
    return fitness_scores


def get_C(chromosome):
    """
    Get the C value from the chromosome. Aka, the first 15 bits of the chromosome, decoded to a float

    :param chromosome: Chromosome to get C value from
    :return: C value as a float in range [1/2^16, 2^16]
    """
    C = from_binary_to_float_in_range(chromosome[:15], 5, [-16, 16])
    return C


def get_gamma(chromosome):
    """
    Get the gamma value from the chromosome. Aka, the second 15 bits of the chromosome, decoded to a float

    :param chromosome: Chromosome to get gamma value from
    :return: gamma value as a float in range [1/2^10, 2^3]
    """
    gamma = from_binary_to_float_in_range(chromosome[15:30], 4, [-10, 3])
    return gamma


def get_selected_features(chromosome, start_index):
    """
    Get selected features bits from the chromosome. Aka the final bits of the chromosome, after the C and gamma bits

    :param chromosome: Chromosome to get selected features from
    :param start_index: Index where selected features bits start
    :return: List of selected features, as a list of bits indicating if a
             feature corresponding to the index is selected or not
    """
    return np.where(chromosome[start_index:] == 1)[0]

def get_classification_threshold(chromosome):
    thresh = from_binary_to_float_in_range(chromosome[30:35], 2, [-4, 0])
    return thresh
