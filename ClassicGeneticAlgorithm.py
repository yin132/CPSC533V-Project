import numpy as np

import matplotlib.pyplot as plt
import random as r

# Using binary encoding for the chromosome
class ClassicGeneticAlgorithm:
    def __init__(self, chromosome_length, compute_chromosome_fitness):
        self.chromosome_length = chromosome_length
        # function which takes in a binary encoded chromosome and returns a fitness score
        self.compute_chromosome_fitness = compute_chromosome_fitness

        self.best_values = np.array([])
        self.average_values = np.array([])
        self.medians = np.array([])
        self.min_values = np.array([])
        self.lower_quartiles = np.array([])
        self.upper_quartiles = np.array([])

    # Inputs:
    # - Population size n
    # - Maximum number of iterations MAX
    # Returns:
    # - Global best solution Y_bt
    def run(self, n, max, C_P = 1.0, M_P = 1.0):
        # Generate initial population of n chromosomes Y_i (i = 1,2, ..., n)
        Y = self.intialize_chromosomes(n)

        # Set iteration counter t = 0
        t = 0

        # Compute the fitness value of each chromosomes
        fitness_values = self.compute_fitness(Y)

        print(f"Generation {t}: Best Value: {np.max(fitness_values)} Average Value: {np.average(fitness_values)} Fitness Values: {fitness_values}")

        while t < max:
            # New population
            Y_new = np.zeros(Y.shape)

            for i in range(Y.shape[0]):
                # Select a pair of chromosomes from intial population based on fitness
                C1, C2 = self.selection(Y, fitness_values)

                # Apply crossover operation on selected pair with crossover probability
                O = self.crossover(C1, C2, C_P)

                # Apply mutation on the offspring with mutation probability
                O_1 = self.mutation(O, M_P)

                Y_new[i] = O_1

            # Replace old population with newly generated population
            Y = Y_new

            # Increment the current iteration t by 1.
            t += 1

            # Compute fitness values of new population
            fitness_values = self.compute_fitness(Y)
            print(f"Generation {t}: Best Value: {np.max(fitness_values)} Average Value: {np.average(fitness_values)} Fitness Values: {fitness_values}")
            self.best_values = np.append(self.best_values, np.max(fitness_values))
            self.average_values = np.append(self.average_values, np.average(fitness_values))
            self.min_values = np.append(self.min_values, np.min(fitness_values))
            self.upper_quartiles = np.append(self.upper_quartiles, np.quantile(fitness_values, 0.75))
            self.lower_quartiles = np.append(self.lower_quartiles, np.quantile(fitness_values, 0.25))
            self.medians = np.append(self.medians, np.median(fitness_values))
        self.plot(max)

    # Returns a set of n chromosomes Y_i (i = 1, 2, ..., n)
    def intialize_chromosomes(self, n):
        return np.random.randint(2, size=(n, self.chromosome_length), dtype=np.int32)
    
    def compute_fitness(self, Y):
        return np.apply_along_axis(self.compute_chromosome_fitness, axis=1, arr=Y)

    # Returns a pair of chromosomes C1, C2 from intial population Y based on fitness values.
    def selection(self, Y, fitness_values):
        # ranked selection: Weight by rank not score

        # # Get rank of each chromosome
        # ranks = len(fitness_values) - np.argsort(fitness_values)

        # # Get probabilities
        # weights = ranks / np.sum(ranks)

        # Use fitness value as weight
        weights = fitness_values / np.sum(fitness_values)

        # Selected indices
        indices = np.random.choice(range(len(fitness_values)), size=2, p=weights, replace=False)

        return Y[indices[0]], Y[indices[1]]

    # Apply crossover operation on selected pair C1, C2 with crossover probability C_P.
    # Returns the resulting offspring O.
    # Types of crossover
    # Uniform: swap points
    # 1-Point: swap segments past 1 point
    # N-Point: swap segments past 1 point n times
    def crossover(self, C1, C2, C_P):
        if r.random() < C_P:
            # 1-point:
            # point = r.randint(0, len(C1) - 1)
            # O = np.concatenate((C1[:point], C2[point:]))

            # uniform:
            # random array of true false
            cross = np.random.choice([True, False], size=len(C2))

            # Combine them
            O = np.where(cross, C1, C2)

            return O
        else:
            return C1

    # Apply mutation on the offspring O with mutation probability M_P
    def mutation(self, O, M_P):
        if r.random() < M_P:
            # Inversion: reverse a string between two random points
            point1 = r.randint(0, len(O) - 1)
            point2 = r.randint(0, len(O) - 1)

            start = min(point1, point2)
            end = max(point1, point2)

            # Invert the subsequence
            O[start:end + 1] = O[start:end + 1][::-1]
            return O
        else:
            return O

    def plot(self, max_generations):
        plt.plot(np.array(list(range(max_generations))), self.average_values, label='average')
        plt.plot(np.array(list(range(max_generations))), self.best_values, label='max')
        plt.plot(np.array(list(range(max_generations))), self.upper_quartiles, label='Q3')
        plt.plot(np.array(list(range(max_generations))), self.medians, label='median')
        plt.plot(np.array(list(range(max_generations))), self.lower_quartiles, label='Q1')
        # plt.plot(np.array(list(range(max_generations))), self.min_values, label='min')
        plt.legend()
        plt.show()
