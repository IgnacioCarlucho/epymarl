import numpy as np 
import torch as th
import os 
import yaml
from utilities import get_obs_sizes, _get_input_shape, _get_config, recursive_dict_update, config_copy

from modules.agents import REGISTRY as agent_REGISTRY
from types import SimpleNamespace as SN


class GeneralController:
    def __init__(self, agent_type, seed, act_sizes, state_sizes, agent_o_size, n_agents):

        # Get the default config from default.yaml
        with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
            try:
                # gives error
                # config_dict = yaml.load(f)
                # https://stackoverflow.com/questions/69564817/typeerror-load-missing-1-required-positional-argument-loader-in-google-col
                self.config_dict = yaml.safe_load(f) 
            except yaml.YAMLError as exc:
                assert False, "default.yaml error: {}".format(exc)

        # Get algorithm configuration
        self.alg_config = _get_config(agent_type, "--config", "algs")
        # This configures the environment
        self.scheme = {
        "n_actions": act_sizes,
        "state": {"vshape": state_sizes},
        "obs": {"vshape": agent_o_size, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (act_sizes,),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        }

        self.agent_type = agent_type
        self.act_sizes = act_sizes
        self.state_sizes = state_sizes
        self.agent_o_size = agent_o_size
        self.n_agents = n_agents

        # All gets saved in the environment configuration 
        self.config_dict = recursive_dict_update(self.config_dict, self.alg_config)
        self.config_dict = recursive_dict_update(self.config_dict, self.scheme)


        # get input confi
        self.input_shape = _get_input_shape(self.agent_o_size, self.n_agents, self.config_dict)

        self.simple_config = SN(**self.config_dict)
        self.agent = agent_REGISTRY["rnn"](self.input_shape, self.simple_config)

        path = os.path.join("results", "models", agent_type, seed)
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def act(self, obs, verbose=True):
        
        #  formats the observation 
        inputs = self.get_inputs(obs)
        # forward pass of the agent
        vals, _ = self.agent(th.tensor(inputs).float(), None)
        if verbose:
            print("vals", vals)
        # some methods have logits as outputs
        if self.config_dict["agent_output_type"] == "pi_logits":
            # if self.config_dict["mask_before_softmax"]:
            # DO I NEED TO DO THIS? 
            # Make the logits for unavailable actions very negative to minimise their affect on the softmax
            # reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
            # agent_outs[reshaped_avail_actions == 0] = -1e10
            vals = th.nn.functional.softmax(vals, dim=-1).detach()
        else: 
            if verbose:
                print("No pi_logits")
            pass
        # choose best action 
        actions = th.argmax(vals,axis=-1).numpy()

        if verbose:
            print("actions", actions)
        return actions

    def get_inputs(self, obs):

        """
            Method to process observation to send to the network  
        """
        # Given the observation, we basically just make it (n_agents,obs)

        inputs = np.vstack((obs[0],obs[1]))
        
        if self.config_dict["obs_last_action"]:
            print("Not coded yet")
            exit()

        if self.config_dict["obs_agent_id"]:
            # or if ids are necessary (n_agents,obs+one_hot_id)
            inputs = np.vstack((np.concatenate([obs[0],np.eye(2)[0]],axis=-1), np.concatenate([obs[1],np.eye(2)[1]],axis=-1)))
        return inputs 