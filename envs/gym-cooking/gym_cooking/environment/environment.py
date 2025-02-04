from gym_cooking.environment import cooking_zoo
from gym.utils import seeding

import gym


class GymCookingEnvironment(gym.Env):
    """Environment object for Overcooked."""

    metadata = {'render.modes': ['human'], 'name': "cooking_zoo"}

    def __init__(self, level, record, max_steps, recipe, obs_spaces=None, action_scheme="full_action_scheme", ghost_agents=0):
        super().__init__()
        self.num_agents = 1
        self.zoo_env = cooking_zoo.CookingEnvironment(level=level, num_agents=self.num_agents, record=record,
                                                max_steps=max_steps, recipes=[recipe], obs_spaces=obs_spaces,
                                                action_scheme=action_scheme, ghost_agents=ghost_agents)
        self.observation_space = self.zoo_env.observation_spaces["player_0"]
        self.action_space = self.zoo_env.action_spaces["player_0"]
        print(self.num_agents, obs_spaces, action_scheme)

    def step(self, action):
        converted_action = {"player_0": action}
        obs, reward, done, info = self.zoo_env.step(converted_action)
        return obs["player_0"], reward["player_0"], done["player_0"], info["player_0"]

    def reset(self):
        return self.zoo_env.reset()["player_0"]

    def render(self, mode='human'):
        pass

