import math
import torch

from ClassicGeneticAlgorithm import ClassicGeneticAlgorithm
from Cartpole import Cartpole
from CartpoleNN import CartpoleNN
from LunarLander import LunarLander
from LunarLanderNN import LunarLanderNN

# num_bins = 0
# cartpole = Cartpole(num_bins)

# # Tabular if num_bin != 0, else Neural Net
# if num_bins == 0:
#     model = CartpoleNN()
#     chromosome_length = 4*32 + 32 + 32*32 + 32 + 32*2 + 2
# else:   
#     action_state_size = cartpole.env.action_space.n
#     observation_space_size = cartpole.env.observation_space.shape[0]
#     chromosome_length = pow(num_bins, observation_space_size) * math.ceil(math.log2(action_state_size))

# # takes in a binary nd array C of length chromosome_length and returns
# # the result of using this as a policy for running cartpole
# def compute_chromosome_fitness(C):
#     if num_bins == 0:
#         # Neural Network Policy
#         model.load_chromosome(C)

#         def chromosome_policy(observation):
#             observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
#             action = model.select_action(observation).item()
#             return int(action)
#     else:
#         # Tabular Policy
#         policy_table = C.reshape(tuple([num_bins]*observation_space_size))
#         # takes in a binned observation and returns an action
#         def chromosome_policy(binned_observation): 
#             action = policy_table[tuple(binned_observation.astype(int))]
#             return int(action)
    
#     return cartpole.run(chromosome_policy)

lunarlander = LunarLander()

model = LunarLanderNN()
chromosome_length = 8*32 + 32 + 32*32 + 32 + 32*4 + 4

# takes in a binary nd array C of length chromosome_length and returns
# the result of using this as a policy for running cartpole
def compute_chromosome_fitness(C):
    # Neural Network Policy
    model.load_chromosome(C)

    def chromosome_policy(observation):
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        action = model.select_action(observation).item()
        return int(action)

    return lunarlander.run(chromosome_policy)

ga = ClassicGeneticAlgorithm(chromosome_length, compute_chromosome_fitness)
ga.run(200, 10, 0.3, 0.01)