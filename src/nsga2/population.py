import numpy as np
import random
from src.util.binary_conversion import from_binary_to_float_in_range


def create_population(pop_size, chromosome_length):
    population = []
    for i in range(pop_size):
        chromosome = np.zeros(chromosome_length)
        n_ones = random.randint(0, chromosome_length)
        chromosome[0: n_ones] = 1
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def crossover(parent1, parent2):
    # Slice one parent into another at a random point, to generate 2 children
    chromosome_length = len(parent1)
    crossover_point = random.randint(1, chromosome_length - 1)
    child1 = np.hstack((parent1[0:crossover_point], parent2[crossover_point:]))
    child2 = np.hstack((parent2[0:crossover_point], parent1[crossover_point:]))
    return child1, child2


def mutate(chromosome):
    mutated_chromosome = chromosome
    bit_index = random.randint(0, len(chromosome) - 1)
    mutated_chromosome[bit_index] = 1 - chromosome[bit_index]
    return mutated_chromosome


def generate_children(population, crossover_rate, mutation_rate):
    new_population = []
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
        new_population.append(c1)
        new_population.append(c2)
    return np.array(new_population)


def get_population_fitness(population, evaluation_algorithm):
    fitness_scores = np.zeros((population.shape[0], 2))
    for i in range(population.shape[0]):
        fitness = evaluation_algorithm(population[i])
        fitness_scores[i] = fitness
    return fitness_scores


def get_C(chromosome):
    # Make sure never 0 (aka all bits 0), in which case, make it the lowest possible number
    C = from_binary_to_float_in_range(chromosome[:14], 5, [-16, 16])
    return C


def get_gamma(chromosome):
    # Make sure never 0 (aka all bits 0), in which case, make it the lowest possible number
    gamma = from_binary_to_float_in_range(chromosome[15:29], 4, [-10, 3])
    return gamma


def get_selected_features(chromosome):
    # Return selected features part of chromosome
    return chromosome[30:]
