from gym.envs.registration import registry, register, make, spec
from itertools import product

corridor_lengths = range(5, 10)
max_timesteps = range(15, 50)

for s, f in product(corridor_lengths, max_timesteps):
    register(
        id="MARL-CooperativeReaching-{0}-{1}-v0".format(s, f),
        entry_point="coopreaching.coopreaching:MARLCooperativeReachingEnv",
        kwargs={
            "world_length": s,
            "world_height": s,
            "max_episode_steps": f,
            "seed": 1234,
            "mode": "MARL"
        }
    )

for s, f in product(corridor_lengths, max_timesteps):
    register(
        id="MARL-CooperativeReaching-{0}-{1}-adhoc-v0".format(s, f),
        entry_point="coopreaching.coopreaching:MARLCooperativeReachingEnv",
        kwargs={
            "world_length": s,
            "world_height": s,
            "max_episode_steps": f,
            "seed": 1234,
            "mode": "adhoc-eval"
        }
    )
