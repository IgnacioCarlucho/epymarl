from gym.envs.registration import register
from grid_generalization.static_layouts import layouts_2p_4w_gen, layouts_2p_4w_train


for i in range(15):
    register(
        id=f'grid-generalization-2p-{i}w-v0',
        entry_point='grid_generalization.environment:GridGeneralization',
        kwargs={
            'grid_size': (9, 9),
            'goal_xys': [[6, 2], [2, 2]],
            'agent_start_xys': [[[6, 6], [7, 7], [5, 7], [7, 5], [5, 8], [8, 5]],
                                [[2, 6], [3, 6], [4, 6], [2, 7], [3, 7], [4, 7]]],
            'n_walls': i,
            'time_limit': 100
        }
    )

# register(
#     id=f'grid-generalization-2p-4w-v0',
#     entry_point='grid_generalization.environment:GridGeneralization',
#     kwargs={
#         'grid_size': (9, 9),
#         'goal_xys': [[6, 2], [2, 2]],
#         'agent_start_xys': [[[6, 6], [7, 7], [5, 7], [7, 5], [5, 8], [8, 5]],
#                             [[2, 6], [3, 6], [4, 6], [2, 7], [3, 7], [4, 7]]],
#         'n_walls': 0,
#         'time_limit': 100,
#         'static_layouts': layouts_2p_4w_train
#     }
# )

register(
    id=f'grid-generalization-2p-4w-static-v0',
    entry_point='grid_generalization.environment:GridGeneralization',
    kwargs={
        'grid_size': (9, 9),
        'goal_xys': [[6, 2], [2, 2]],
        'agent_start_xys': [[[6, 6]],
                            [[2, 6]]],
        'n_walls': 4,
        'time_limit': 100,
        'static_layouts': layouts_2p_4w_gen
    }
)
