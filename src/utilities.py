import os 
import yaml
import collections
import numpy as np

def get_obs_sizes(obs_space, n_agents, env_name):
        """
            Method to get the size of the envs' obs space and length of obs features. Must be defined for every envs.
        """
        out_shape = list(obs_space.shape)
        out_shape[-1] += n_agents
        if "Foraging" in env_name:
            return 2*obs_space.shape[-1] , obs_space.shape[-1] 
        return None

def _get_input_shape(agent_o_size, n_agents, args):
    """
        Method to get the input size of the network 
    """
    input_shape = agent_o_size
    if args["obs_last_action"]:
        input_shape += args["actions_onehot"]["vshape"][0]
    if args["obs_agent_id"]:
        input_shape += n_agents

    return input_shape

def _get_config(config_name, arg_name, subfolder):
    if config_name is not None:
       # Get the defaults from default.yaml
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder ,config_name + ".yaml"), "r") as f:
            try:
                print(f)
                # gives error
                # config_dict = yaml.load(f)
                # https://stackoverflow.com/questions/69564817/typeerror-load-missing-1-required-positional-argument-loader-in-google-col
                config_dict = yaml.safe_load(f) 
            except yaml.YAMLError as exc:
                assert False, "default.yaml error: {}".format(exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def get_inputs_single(obs, config_dict):

    """
        Method to process observation to send to the network  
    """
    # Given the observation, we basically just make it (n_agents,obs)

    inputs = np.vstack((obs[0],obs[1]))
    
    if config_dict["obs_last_action"]:
        print("Not coded yet")
        # exit()

    if config_dict["obs_agent_id"]:
        # or if ids are necessary (n_agents,obs+one_hot_id)
        inputs = np.vstack((np.concatenate([obs[0],np.eye(2)[0]],axis=-1), np.concatenate([obs[1],np.eye(2)[1]],axis=-1)))
    return inputs 