from ClassicGeneticAlgorithm import ClassicGeneticAlgorithm
from Cartpole import Cartpole

import math

num_bins = 10

cartpole = Cartpole(num_bins)
action_state_size = cartpole.env.action_space.n
observation_space_size = cartpole.env.observation_space.shape[0]

chromosome_length = pow(num_bins, observation_space_size) * math.ceil(math.log2(action_state_size))

# takes in a binary nd array C of length chromosome_length and returns
# the result of using this as a policy for running cartpole
def compute_chromosome_fitness(C):
    policy_table = C.reshape(tuple([num_bins]*observation_space_size))
    # takes in a binned observation and returns an action
    def chromosome_policy(binned_observation): 
        action = policy_table[tuple(binned_observation.astype(int))]
        return int(action)

    return cartpole.run(chromosome_policy)

ga = ClassicGeneticAlgorithm(chromosome_length, compute_chromosome_fitness)
ga.run(20, 200, 0.5, 0.01)