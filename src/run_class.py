import numpy as np
import gym
import os
import yaml
from agent_class import GeneralController
from utilities import get_obs_sizes
import lbforaging


# Config environment 
env_name = "Foraging-6x6-2p-4f-v2"
env = gym.make(env_name)
env.seed(2555)
# Get environment parameters to define agent
n_agents = env.n_agents
state_sizes, agent_o_size = get_obs_sizes(env.new_observation_space, n_agents, env_name)
act_sizes = env.action_space.nvec[0]

# Define agent. As of now you have 4 to choose from, with 3 seeds each 

agent = GeneralController("qmix", "seed_1", act_sizes, state_sizes, agent_o_size, n_agents)
# agent = GeneralController("iql", "seed_2", act_sizes, state_sizes, agent_o_size, n_agents)
# agent = GeneralController("ippo", "seed_3", act_sizes, state_sizes, agent_o_size, n_agents)
# agent = GeneralController("mappo", "seed_3", act_sizes, state_sizes, agent_o_size, n_agents)



n_episode = 10
render_env = False
# Run for number of episodes 
for i in range(n_episode):
    obs = env.reset()
    avgs = []
    dones = [False]*2

    # And until it is not done 
    while not any(dones):
        # Select actions
        actions = agent.act(obs, verbose=False)
        # Take action 
        n_obs, rews, dones, infos = env.step(actions)

        # Not important
        if render_env:
            env.render()
        avgs.append(rews[0])
        obs = n_obs

    print("rewards", np.sum(avgs), "in " +  str(len(avgs)) + " steps" )

