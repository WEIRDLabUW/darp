import torch
import numpy as np

from fast_scaler import FastScaler
from util import load_expert_data
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(module, *args):
    return checkpoint(module, *args, use_reentrant=False)

def set_attributes_from_args(obj, default_config, args):
    # Args optionally contains a config dictionary
    # Hierarchy goes default < config < explicitly provided kwargs

    # First just populate args with the default values
    curr_args = default_config.copy()

    # Extract and remove config dict if present
    config_dict = args.pop("config", {})

    for key, value in config_dict.items():
        if key in curr_args:
            curr_args[key] = value
        else:
            print(f"Key {key} not recognized!")

    for key, value in args.items():
        if key in curr_args:
            curr_args[key] = args[key]
        else:
            print(f"Key {key} not recognized!")

    for key, value in curr_args.items():
        assert value != None, f"{key} must be explicitly set, it has no default!"
        setattr(obj, key, value)

def get_scalers_from_data_path(path, darp=False):
    expert_data = load_expert_data(path)

    obs_scaler = FastScaler()
    obs_scaler.fit(np.concatenate([traj['observations'] for traj in expert_data]))

    act_scaler = FastScaler()
    act_scaler.fit(np.concatenate([traj['actions'] for traj in expert_data]))

    if darp:
        action_mean = act_scaler.mean_np
        delta_mean = np.zeros_like(obs_scaler.mean_np)

        obs_scaler.mean_np = np.hstack((np.zeros_like(obs_scaler.mean_np), action_mean, delta_mean))

        action_std = act_scaler.scale_np
        delta_std = np.ones_like(obs_scaler.scale_np) * np.sqrt(2)
        obs_scaler.scale_np = np.hstack((np.ones_like(obs_scaler.scale_np), action_std, delta_std))
        obs_scaler.mean_torch = torch.as_tensor(obs_scaler.mean_np)
        obs_scaler.scale_torch = torch.as_tensor(obs_scaler.scale_np)

    return obs_scaler, act_scaler

def get_io_size_from_data_path(path, classifier=False, darp=False):
    expert_data = load_expert_data(path)

    if classifier:
        # Assume actions are class indices
        max_action_class = 0
        for traj in expert_data:
            max_action_class = np.max((max_action_class, np.max(traj['actions'])))

        return len(expert_data[0]['observations'][0]), max_action_class + 1
    elif darp:
        return (len(expert_data[0]['observations'][0]) * 2) + len(expert_data[0]['actions'][0]), len(expert_data[0]['actions'][0])
    else:
        return len(expert_data[0]['observations'][0]), len(expert_data[0]['actions'][0])

def get_min_max_len(path, norm_obs=False):
    expert_data = load_expert_data(path)

    actions = np.concatenate([traj['actions'] for traj in expert_data])
    obs = np.concatenate([traj['observations'] for traj in expert_data])

    if norm_obs:
        obs_scaler = FastScaler()
        obs_scaler.fit(obs)
        obs = obs_scaler.transform(obs)

    return torch.tensor(np.min(actions, axis=0)), torch.tensor(np.max(actions, axis=0)), len(actions[0]), torch.tensor(np.min(obs, axis=0)), torch.tensor(np.max(obs, axis=0)), len(obs[0])

