import numpy as np
import gym
from gym import spaces
from random import shuffle


_LAYER_AGENTS = 0
_LAYER_GOALS = 1
_LAYER_WALLS = 2


class Entity:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y


class Agent(Entity):
    def __init__(self, num, x, y, goal_x, goal_y):
        super().__init__(f'agent_{num}', x, y)
        self.goal_x = goal_x
        self.goal_y = goal_y


class Goal(Entity):
    def __init__(self, num, x, y):
        super().__init__(f'goal_{num}', x, y)


class Wall(Entity):
    def __init__(self, num, x, y):
        super().__init__(f'wall_{num}', x, y)


class GridGeneralization(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, grid_size, goal_xys, agent_start_xys, n_walls, time_limit, static_layouts=None):
        super().__init__()

        # assert len(goal_xys) == len(agent_start_xys), 'Number of goals != number of agents'

        self.grid = np.zeros((3, *grid_size))
        self.grid_size = grid_size
        self.goal_xys = goal_xys
        self.agent_start_xys = agent_start_xys
        self.n_walls = n_walls
        self.time_limit = time_limit
        self.static_layouts = static_layouts
        self.n_agents = len(agent_start_xys)

        self.action_space = spaces.Tuple(tuple(len(self.agent_start_xys) * [spaces.Discrete(5)]))

        self.agents = []
        self.goals = []
        self.walls = []

        self.current_step = 0
        self.reset()

        self.observation_space = spaces.Tuple(tuple(
            [spaces.Box(
                low=0.0, high=1.0, shape=(self._make_obs()[0].shape), dtype=np.float32
            ) for _ in range(self.n_agents)]
        ))

        self._rendering_initialized = False

    def reset(self):
        self.grid = np.zeros((3, *self.grid_size))
        self.current_step = 0

        taken_xy = []

        # Place agents
        self.agents = []
        for i in range(self.n_agents):
            loc = np.random.choice(len(self.agent_start_xys[i]))
            self.agents.append(Agent(
                i, self.agent_start_xys[i][loc][0], self.agent_start_xys[i][loc][1], self.goal_xys[i][0], self.goal_xys[i][1]
            ))

            taken_xy.append(self.agent_start_xys[i])
            self.grid[_LAYER_AGENTS][self.agents[i].y, self.agents[i].x] = 1

        # Place goals
        self.goals = []
        for i in range(len(self.goal_xys)):
            self.goals.append(Goal(
                i, self.goal_xys[i][0], self.goal_xys[i][1]
            ))

            taken_xy.append(self.goal_xys[i])
            self.grid[_LAYER_GOALS][self.goals[i].y, self.goals[i].x] = 1

        # The static layouts are manually-placed walls
        if not self.static_layouts:
            # Place walls
            self.walls = []
            n_walls_placed = 0
            while n_walls_placed < self.n_walls:
                # The below [0] indexing (self.grid_size[0]) assumes the grid is square
                rand_xy = np.random.choice(self.grid_size[0], size=2).tolist()

                while rand_xy in taken_xy:
                    rand_xy = np.random.choice(self.grid_size[0], size=2).tolist()

                self.walls.append(Wall(n_walls_placed, rand_xy[0], rand_xy[1]))
                taken_xy.append(rand_xy)
                self.grid[_LAYER_WALLS][rand_xy[1], rand_xy[0]] = 1

                n_walls_placed += 1

        else:
            # The static layouts are numpy arrays of [0,1]^{*grid_size}, so we can just += to the zero'ed grid
            wall_matrix = self.static_layouts[np.random.choice(len(self.static_layouts))]

            self.grid[_LAYER_WALLS] += wall_matrix

            self.walls = []
            n_walls_placed = 0
            for i in range(wall_matrix.shape[0]):
                for j in range(wall_matrix.shape[1]):
                    if wall_matrix[i, j] == 1:
                        self.walls.append(
                            Wall(n_walls_placed, j, i)
                        )

                        n_walls_placed += 1

        return self._make_obs()

    def step(self, actions):
        # The below list-creation and -shuffle randomizes the agents' move order
        agent_order = list(range(len(self.agents)))
        shuffle(agent_order)

        team_reward = 0

        for i in agent_order:
            current_xy = [self.agents[i].x, self.agents[i].y]

            # Up
            if actions[i] == 0:
                proposed_xy = [current_xy[0], current_xy[1] - 1]
                wall_collision, out_of_grid, agent_collision = self._is_collision(proposed_xy, i)
                if np.any([wall_collision, out_of_grid, agent_collision]):
                    proposed_xy = current_xy

            # Right
            elif actions[i] == 1:
                proposed_xy = [current_xy[0] + 1, current_xy[1]]
                wall_collision, out_of_grid, agent_collision = self._is_collision(proposed_xy, i)
                if np.any([wall_collision, out_of_grid, agent_collision]):
                    proposed_xy = current_xy

            # Down
            elif actions[i] == 2:
                proposed_xy = [current_xy[0], current_xy[1] + 1]
                wall_collision, out_of_grid, agent_collision = self._is_collision(proposed_xy, i)
                if np.any([wall_collision, out_of_grid, agent_collision]):
                    proposed_xy = current_xy

            # Left
            elif actions[i] == 3:
                proposed_xy = [current_xy[0] - 1, current_xy[1]]
                wall_collision, out_of_grid, agent_collision = self._is_collision(proposed_xy, i)
                if np.any([wall_collision, out_of_grid, agent_collision]):
                    proposed_xy = current_xy

            # No-op
            else:
                proposed_xy = [current_xy[0], current_xy[1]]
                wall_collision, out_of_grid, agent_collision = False, False, False

            # Stepping the agent to the new position
            self.agents[i].x = proposed_xy[0]
            self.agents[i].y = proposed_xy[1]
            self.grid[_LAYER_AGENTS][current_xy[1], current_xy[0]] = 0
            self.grid[_LAYER_AGENTS][proposed_xy[1], proposed_xy[0]] = 1

            # Reward is the negative L1 distance
            # individual_reward = np.abs(
            #     np.array([self.agents[i].x, self.agents[i].y]) - np.array([self.agents[i].goal_x, self.agents[i].goal_y])
            # ).sum()

            # Adding a penalty for colliding with a wall
            individual_reward = np.sqrt(np.power(
                np.array([self.agents[i].x, self.agents[i].y]) - np.array(
                    [self.agents[i].goal_x, self.agents[i].goal_y]), 2
            ).sum()) + wall_collision

            team_reward -= individual_reward

        self.current_step += 1

        # Term conditions: (1) both agents on goal or time-limit reached
        if team_reward == 0 or self.current_step == self.time_limit:
            done = [True for _ in range(len(self.agents))]
        else:
            done = [False for _ in range(len(self.agents))]

        
        # return self._make_obs(), team_reward, done, {}
        return self._make_obs(), [team_reward]*2, [done]*2, {}

    def _is_collision(self, proposed_xy, agent_idx):
        wall_collision = False
        out_of_grid = False
        agent_collision = False

        # First check collision with walls
        for wall in self.walls:
            if proposed_xy == [wall.x, wall.y]:
                wall_collision = True
                break

        # Checking if agent is outside the grid
        if np.any([
            proposed_xy[0] < 0,
            proposed_xy[1] < 0,
            proposed_xy[0] > self.grid_size[1] - 1,
            proposed_xy[1] > self.grid_size[0] - 1
        ]):
            out_of_grid = True

        # Checking to see if agent is in another agent's cell
        for i in range(len(self.agents)):
            if i == agent_idx:
                continue

            if proposed_xy == [self.agents[i].x, self.agents[i].y]:
                agent_collision = True

        return wall_collision, out_of_grid, agent_collision

    def _make_obs(self):
        """
        [agent_xy, goal_xy, walls, other_agent_xys]
        Returns:

        """
        team_obs = []

        for i in range(len(self.agents)):
            individual_obs = []

            # agent_xy + goal_xy
            individual_obs.extend([self.agents[i].x / self.grid_size[1], self.agents[i].y / self.grid_size[0]])
            individual_obs.extend([self.agents[i].goal_x / self.grid_size[1], self.agents[i].goal_y / self.grid_size[1]])

            # Looping through all agents (skipping current agent) and appending their xy
            for j in range(len(self.agents)):
                if i == j:
                    continue

                individual_obs.extend([self.agents[j].x / self.grid_size[1], self.agents[j].y / self.grid_size[0]])

            # Wall matrix unrolled. This is currently a one-hot representation
            individual_obs.extend(self.grid[_LAYER_WALLS].ravel().tolist())

            team_obs.append(np.array(individual_obs))

        return team_obs

    def _init_render(self):
        from .rendering import Viewer
        self.viewer = Viewer(self.grid_size)
        self._rendering_initialized = True

    def render(self, mode='human'):
        if not self._rendering_initialized:
            self._init_render()
        return self.viewer.render(self, mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
        self._rendering_initialized = False
