import pickle
from dataclasses import dataclass
import os

import torch
import random

import numpy as np
from fast_scaler import FastScaler
from typing import List
from logging_util import logger

# To be populated if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Dataset:
    name: str
    obs_scaler: FastScaler | None # None if not normalized
    act_scaler: FastScaler
    rot_indices: np.ndarray | torch.Tensor
    weights: np.ndarray | torch.Tensor
    non_rot_indices: np.ndarray | torch.Tensor
    obs_matrix: List
    act_matrix: List
    traj_starts: np.ndarray | torch.Tensor
    flattened_obs_matrix: np.ndarray | torch.Tensor
    flattened_act_matrix: np.ndarray | torch.Tensor
    processed_obs_matrix: np.ndarray | torch.Tensor

def load_and_scale_data(path, rot_indices, weights, ob_type='state', scale=True, bc=False, device='cuda'):
    expert_data = load_expert_data(path)
    
    is_numpy = isinstance(expert_data[0]['observations'][0], np.ndarray)
    if is_numpy:
        observations = np.concatenate([traj['observations'] for traj in expert_data])
        observations = torch.from_numpy(observations)
    else:
        observations = torch.concatenate([traj['observations'] for traj in expert_data])

    rot_indices = torch.tensor(rot_indices, dtype=torch.int32)
    # Separate non-rotational dimensions
    non_rot_indices = torch.tensor([i for i in range(observations.shape[-1]) if i not in rot_indices], dtype=torch.int32)

    if scale:
        obs_scaler = FastScaler()
        obs_scaler.fit(observations)

        if ob_type == 'retrieval':
            obs_scaler.mean_np[rot_indices] = 0.0
            obs_scaler.mean_torch[rot_indices] = 0.0
            obs_scaler.scale_np[rot_indices] = 1.0
            obs_scaler.scale_torch[rot_indices] = 1.0

        for traj in expert_data:
            traj['observations'] = obs_scaler.transform(torch.from_numpy(traj['observations']) if is_numpy else traj['observations'])
                
        new_path = path[:-4] + '_standardized.pkl'
        save_expert_data(expert_data, new_path)
    else:
        new_path = path
        obs_scaler = None

    act_scaler = FastScaler()
    if isinstance(expert_data[0]['actions'][0], np.ndarray):
        act_scaler.fit(np.concatenate([traj['actions'] for traj in expert_data]))
    else:
        act_scaler.fit(torch.concatenate([traj['actions'] for traj in expert_data]))
    
    obs_matrix, act_matrix, traj_starts = create_matrices(expert_data, use_torch=True)
    
    if not bc:
        flattened_obs_matrix = torch.cat([obs for obs in obs_matrix], dim=0).to(device)
        flattened_act_matrix = torch.cat([act for act in act_matrix], dim=0).to(device)
        
        if len(weights) > 0:
            weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
            processed_obs_matrix = flattened_obs_matrix[:, non_rot_indices] * torch.as_tensor(weights[non_rot_indices], dtype=flattened_obs_matrix[0][0].dtype)
        else:
            weights = torch.ones(obs_matrix[0][0].shape[0], dtype=torch.float32, device=device)
            processed_obs_matrix = flattened_obs_matrix


        traj_starts = torch.as_tensor(traj_starts)
    else:
        flattened_obs_matrix = None
        flattened_act_matrix = None
        weights = None
        processed_obs_matrix = None
        traj_starts = None
        
    return Dataset(new_path, obs_scaler, act_scaler, rot_indices, weights, non_rot_indices, obs_matrix, act_matrix, traj_starts, flattened_obs_matrix, flattened_act_matrix, processed_obs_matrix)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info(f"Seeded with {seed}")

def load_expert_data(path):
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)

def save_expert_data(data, path):
    if not os.path.exists(path):
        with open(path, 'wb') as output_file:
            return pickle.dump(data, output_file)

def create_matrices(expert_data, use_torch=False):
    obs_matrix = []
    act_matrix = []
    traj_starts = []

    idx = 0
    for traj in expert_data:
        # We will eventually be flattening all trajectories into a single list,
        # so keep track of trajectory start indices
        traj_starts.append(idx)
        idx += len(traj['observations'])

        # Create matrices for all observations and actions where each row is a trajectory
        # and each column is an single state or action within that trajectory
        if use_torch:
            obs_matrix.append(torch.as_tensor(traj['observations']))
            act_matrix.append(torch.as_tensor(traj['actions']))
        else:
            obs_matrix.append(traj['observations'])
            act_matrix.append(traj['actions'])

    if use_torch:
        traj_starts = torch.as_tensor(traj_starts)
    else:
        traj_starts = np.asarray(traj_starts)
    return obs_matrix, act_matrix, traj_starts

