import numpy as np
import gym
import os
import yaml
from agent_class import GeneralController
from diversity_class import DivGeneralController
from utilities import get_obs_sizes
import lbforaging
import coopreaching
import gym_cooking


# Parameters 
VERBOSE = False
SEED = 25551
n_episode = 10
render_env = False

# Config environment 
env_name = "Foraging-6x6-2p-4f-v2"
# env_name = "Foraging-6x6-2p-3f-v2"
#env_name = "MARL-CooperativeReaching-5-40-v0"
#env_name = "cookingZooEnv-tiny-250-v0"

env = gym.make(env_name)
env.seed(SEED)
if VERBOSE:
    print(env.reset())
# Get environment parameters to define agent
n_agents = env.n_agents
state_sizes, agent_o_size = get_obs_sizes(env.new_observation_space, n_agents, env_name)
act_sizes = env.action_space.nvec[0]
# print them if necessary 
if VERBOSE:
    print("args.n_agents", n_agents)
    print('act_sizes', act_sizes)
    print("state", env.observation_space)
    print("state2", env.new_observation_space)
    print('state_sizes', state_sizes)
    print('agent_o_size', agent_o_size)

################## Important bit ########################
# Define agent. As of now you have 6 to choose from, with 3 seeds each 
agent = GeneralController("qmix", "seed_3", act_sizes, state_sizes, agent_o_size, n_agents, env_name)
# agent = GeneralController("iql", "seed_2", act_sizes, state_sizes, agent_o_size, n_agents, env_name)
# agent = GeneralController("ippo", "seed_3", act_sizes, state_sizes, agent_o_size, n_agents, env_name)
# agent = GeneralController("mappo", "seed_1", act_sizes, state_sizes, agent_o_size, n_agents, env_name)
# agent = GeneralController("maddpg", "seed_3", act_sizes, state_sizes, agent_o_size, n_agents, env_name)
# agent = GeneralController("maa2c", "seed_3", act_sizes, state_sizes, agent_o_size, n_agents, env_name)




# Run for number of episodes 
for i in range(n_episode):
    obs = env.reset()
    avgs = []
    dones = [False]*2

    # And until it is not done 
    while not any(dones):
        # Select actions
        actions = agent.act(obs, verbose=VERBOSE)
        # Take action 
        n_obs, rews, dones, infos = env.step(actions)

        # Not important
        if render_env:
            env.render()
        avgs.append(rews[0])
        obs = n_obs
        # input()
    print("rewards", np.sum(avgs), "in " +  str(len(avgs)) + " steps" )

