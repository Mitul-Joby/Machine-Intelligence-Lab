#########################################################################################################

# Build and Train a neural network using Genetic Algorithm to realise the functionality of XOR gate.
# To start evolving the GA, the run() method is called. This method applies the pipeline of the genetic
# algorithm by calculating the fitness values of the solutions, selecting the parents, mating the parents
# by applying the mutation and crossover operations, and producing a new population.

# This process lasts for the 50 generations.
# Assume the following:
# 1) Each chromosome represents all the weights in the network.
# 2) Initial random population of 10 members
# 3) Accuracy measure is used as fitness function.
# 4) Initial biases
# 5) Activation functions as either Sigmoid or Relu for input and hidden layers. You may choose softmax 
#    for output layer.
# 6) Use appropriate GA Operators: Crossover, Mutation

#########################################################################################################

'''
UE20CS302 (D Section)
Machine Intelligence
Genetic Algorithm - XOR Gate

Mitul Joby
PES2UG20CS199
'''

import numpy
import pygad
import pygad.nn
import pygad.gann

def fitness_func(solution, sol_idx):
    global GANN, inputs, outputs
    predictions = pygad.nn.predict(last_layer = GANN.population_networks[sol_idx], data_inputs = inputs)
    correct_predictions = numpy.where(predictions == outputs)[0].size
    solution_fitness = (correct_predictions / outputs.size) * 100
    return solution_fitness

def on_generation(ga):
    global GANN
    population_matrices = pygad.gann.population_as_matrices(population_networks = GANN.population_networks, population_vectors = ga.population)
    GANN.update_population_trained_weights(population_trained_weights=population_matrices)

    print(f"Generation = {ga.generations_completed}")
    print(f"Fitness    = {ga.best_solution()[1]}")

inputs  = numpy.array([[1, 1], [1, 0], [0, 1], [0, 0]])
outputs = numpy.array([0, 1, 1, 0])

GANN = pygad.gann.GANN(num_solutions = 10, 
                        num_neurons_input = 2,
                        num_neurons_hidden_layers = [2],
                        num_neurons_output = 2,
                        hidden_activations = ["relu"],
                        output_activation = "softmax")

population_vectors = pygad.gann.population_as_vectors(population_networks=GANN.population_networks)

ga = pygad.GA(num_generations = 50, 
                num_parents_mating = 3, 
                initial_population = population_vectors.copy(),
                fitness_func = fitness_func,
                mutation_percent_genes = 10,
                mutation_num_genes = 2,
                on_generation = on_generation)

ga.run()
print()

solution, solution_fitness, solution_idx = ga.best_solution()
print(f"Best solution        : {solution}")
print(f"Best solution fitness: {solution_fitness}")
print(f"Best solution idx    : {solution_idx}")

predictions = pygad.nn.predict(last_layer = GANN.population_networks[solution_idx], data_inputs = inputs)
print(f"\nInputs:\n{inputs}")
print(f"Predictions : {predictions}")
