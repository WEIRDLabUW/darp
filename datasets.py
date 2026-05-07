import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from util import create_matrices
from typing import Tuple

class IndexActionBCDataset(Dataset):
    def __init__(self, dataset_path, act_dataset=None):
        if act_dataset is not None:
            self.act_dataset = act_dataset
            dummy_obs_matrix, _, self.traj_starts = create_matrices(pickle.load(open(dataset_path, 'rb')), use_torch=True)
            last_start = 0
            self.obs_matrix = []
            for i in range(1, len(self.traj_starts)):
                self.obs_matrix.append(np.arange(last_start, self.traj_starts[i]))
                last_start = self.traj_starts[i]
            
            self.obs_matrix.append(np.arange(last_start, len(self.act_dataset)))

            assert len(dummy_obs_matrix) == len(self.obs_matrix)

            for i in range(len(dummy_obs_matrix)):
                assert len(dummy_obs_matrix[i]) == len(self.obs_matrix[i])

            self.action_size = self.act_dataset[0][1].shape[-1]
            self.state_size = 1
        else:
            self.act_dataset = None
            _, self.act_matrix, self.traj_starts = create_matrices(pickle.load(open(dataset_path, 'rb')), use_torch=True)

            self.acts = torch.cat([torch.as_tensor(act) for act in self.act_matrix], dim=0)

            self.action_size = self.acts.shape[-1]
            self.state_size = 1

    def __len__(self):
        if self.act_dataset:
            return len(self.act_dataset)
        else:
            return len(self.acts)

    def __getitem__(self, idx):
        if self.act_dataset:
            return idx, self.act_dataset[idx][1]
        else:
            return idx, self.acts[idx]

    def collate_fn(self, batch):
        idxs = [item[0] for item in batch]
        acts = torch.stack([item[1] for item in batch])

        return idxs, acts

class BCExpertDataset(Dataset):
    def __init__(self, dataset_path, rgb_dataset_path=None):
        print(f"Creating BCExpertDataset with data at {dataset_path}")
        self.obs_matrix, self.act_matrix, self.traj_starts = create_matrices(pickle.load(open(dataset_path, 'rb')), use_torch=True)

        self.obs = torch.cat([torch.as_tensor(obs) for obs in self.obs_matrix], dim=0)
        self.acts = torch.cat([torch.as_tensor(act) for act in self.act_matrix], dim=0)

        self.state_size = self.obs.shape[-1]

        self.action_size = self.acts.shape[-1]

        if rgb_dataset_path is not None:
            rgb_obs_matrix = create_matrices(pickle.load(open(rgb_dataset_path, 'rb')), use_torch=True)[0]
            self.rgb_obs = torch.cat([torch.as_tensor(obs) for obs in rgb_obs_matrix], dim=0)
            assert len(self.obs) == len(self.rgb_obs)
            self.include_rgb = True
            self.state_size += self.rgb_obs.shape[-1]
        else:
            self.include_rgb = False

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if self.include_rgb:
            return (self.obs[idx], self.rgb_obs[idx]), self.acts[idx]
        else:
            return self.obs[idx], self.acts[idx]

    def collate_fn(self, batch):
        obs = [item[0] for item in batch]
        acts = torch.stack([item[1] for item in batch])

        if not self.include_rgb:
            return torch.stack(obs), acts

        prop_parts, rgb_parts = zip(*obs)

        obs_combined = torch.cat((torch.stack(prop_parts), torch.stack(rgb_parts)), dim=-1)

        return obs_combined, acts

class DARPMixedExpertDataset(Dataset):
    def __init__(self, retrieval_dataset_path, delta_state_dataset_path):
        self.retrieval_dataset = BCExpertDataset(retrieval_dataset_path)
        self.delta_state_dataset = BCExpertDataset(delta_state_dataset_path)

        self.state_size = self.retrieval_dataset.state_size + self.delta_state_dataset.state_size

        print(f"{len(self.retrieval_dataset)}, {len(self.delta_state_dataset)}")
        assert len(self.retrieval_dataset) == len(self.delta_state_dataset)

    def __len__(self):
        return len(self.retrieval_dataset)

    def __getitem__(self, idx):
        retrieval_obs, action = self.retrieval_dataset[idx]
        delta_state_obs, debug_action = self.delta_state_dataset[idx]
        assert torch.equal(action, debug_action)
        return (retrieval_obs, delta_state_obs), action

    def collate_fn(self, batch):
        obs = [item[0] for item in batch]
        acts = torch.stack([item[1] for item in batch])

        retrieval_obs, delta_state_obs = zip(*obs)

        obs_combined = torch.cat((torch.stack(retrieval_obs), torch.stack(delta_state_obs)), dim=-1)

        return obs_combined, acts

    def __getattr__(self, name):
        if hasattr(self.retrieval_dataset, name):
            return getattr(self.retrieval_dataset, name)
        if hasattr(self.delta_state_dataset, name):
            return getattr(self.delta_state_dataset, name)

        raise AttributeError(f"Neither '{self.__class__.__name__}' nor either wrapped dataset has attribute '{name}'")

class ChunkingWrapper(Dataset):
    def __init__(self, obs_horizon, act_horizon, wrapped: Dataset, fill_method="repeat"):
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.wrapped = wrapped
        self.fill_method = fill_method

        # Caches
        self.idx_populated = torch.zeros(len(wrapped), dtype=torch.bool)

        self.state_lookup = torch.empty((len(wrapped), self.obs_horizon, self.state_size))
        self.state_idx_lookup = torch.empty((len(wrapped), self.obs_horizon))
        self.action_lookup = torch.empty((len(wrapped), self.act_horizon, self.action_size))

    def __getitem__(self, idx):
        if not self.idx_populated[idx]:
            state_traj = torch.searchsorted(self.traj_starts, idx, right=True) - 1
            traj_start = self.traj_starts[state_traj]

            state_num = idx - traj_start
            traj_len = len(self.obs_matrix[state_traj])

            padding_needed = max(0, self.obs_horizon - state_num - 1)
            obs_indices = list(range(state_num - self.obs_horizon + padding_needed + 1, state_num + 1))

            if self.fill_method == "repeat":
                obs_indices = torch.tensor(([0] * padding_needed + obs_indices)) + traj_start
            else:
                obs_indices = torch.tensor([-1] * padding_needed + obs_indices)
                obs_indices[torch.where(obs_indices != -1)[0]] += traj_start

            assert len(obs_indices) == self.obs_horizon

            obs = torch.empty((self.obs_horizon, self.state_size))

            for i, wrapped_i in enumerate(obs_indices):
                if wrapped_i == -1:
                    obs[i] = torch.full((self.state_size,), torch.nan, dtype=torch.float32)
                else:
                    wrapped_item = self.wrapped[wrapped_i]
                    if hasattr(self.wrapped, "collate_fn"):
                        wrapped_item = self.wrapped.collate_fn([wrapped_item])
                        
                    if isinstance(self.wrapped, IndexActionBCDataset):
                        obs[i] = torch.tensor(wrapped_item[0])
                    else:
                        obs[i] = wrapped_item[0]

            self.state_lookup[idx] = obs
            self.state_idx_lookup[idx] = obs_indices.to(torch.int32)

            padding_needed = max(traj_len, state_num + self.act_horizon) - traj_len
            act_indices = list(range(state_num, state_num + self.act_horizon - padding_needed))
            act_indices = torch.tensor((act_indices + [traj_len - 1] * padding_needed)) + traj_start

            assert len(act_indices) == self.act_horizon

            acts = torch.empty((self.act_horizon, self.action_size))

            for i, wrapped_i in enumerate(act_indices):
                if wrapped_i == -1:
                    acts[i] = torch.full((self.action_size,), torch.nan, dtype=torch.float32)
                else:
                    acts[i] = self.wrapped[wrapped_i][1]

            self.action_lookup[idx] = acts

            self.idx_populated[idx] = True

        obs = self.state_lookup[idx]
        acts = self.action_lookup[idx]

        if self.act_horizon == 1:
            acts = acts.squeeze()

        if self.obs_horizon == 1:
            obs = obs.squeeze()

        return obs, acts

    def __len__(self):
        return len(self.wrapped)

    def __getattr__(self, name):
        if hasattr(self.wrapped, name):
            return getattr(self.wrapped, name)

        raise AttributeError(f"Neither '{self.__class__.__name__}' nor wrapped dataset has attribute '{name}'")

    def collate_fn(self, batch):
        obs = torch.stack([item[0] for item in batch])
        acts = torch.stack([item[1] for item in batch])
        return obs, acts

def create_dataset(env_cfg, model_cfg) -> Tuple[Dataset, Dataset | None]:
    obs_horizon = model_cfg.get("obs_horizon", 1)
    act_horizon = model_cfg.get("act_horizon", 1)

    if env_cfg.get("mixed", False):
        train_dataset = DARPMixedExpertDataset(env_cfg['retrieval']['demo_pkl'], env_cfg['delta_state']['demo_pkl'])

        if env_cfg['retrieval'].get("val_demo_pkl", None):
            val_dataset = DARPMixedExpertDataset(env_cfg['retrieval']['val_demo_pkl'], env_cfg['delta_state']['val_demo_pkl'])
        else:
            val_dataset = None
    else:
        train_dataset = BCExpertDataset(env_cfg['demo_pkl'], rgb_dataset_path=env_cfg.get('rgb_demo_pkl'))

        if env_cfg.get("val_demo_pkl", None):
            val_dataset = BCExpertDataset(env_cfg['val_demo_pkl'], rgb_dataset_path=env_cfg.get('val_rgb_demo_pkl'))
        else:
            val_dataset = None

    if obs_horizon > 1 or act_horizon > 1:
        train_dataset = ChunkingWrapper(obs_horizon, act_horizon, train_dataset)

        if val_dataset is not None:
            val_dataset = ChunkingWrapper(obs_horizon, act_horizon, val_dataset)
    else:
        if 'retrieval_config' in model_cfg and model_cfg['retrieval_config'].get('lookback', 1) > 1:
            train_dataset = ChunkingWrapper(model_cfg['retrieval_config']['lookback'], 1, train_dataset, fill_method='nan')

            if val_dataset is not None:
                val_dataset = ChunkingWrapper(model_cfg['retrieval_config']['lookback'], 1, val_dataset, fill_method='nan')

    return train_dataset, val_dataset

def create_dataloader(train_dataset: Dataset, val_dataset: Dataset | None, rank: int, world_size: int, batch_size: int, shuffle=True, drop_last=False) -> Tuple[DataLoader, DataLoader | None, DistributedSampler | None, DistributedSampler | None]: 
    generator = torch.Generator()
    generator.manual_seed(42)

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            collate_fn=getattr(train_dataset, "collate_fn", None),
            num_workers=0,
            generator=generator,
            sampler=train_sampler,
            pin_memory=False,
            drop_last=drop_last
        )
        if val_dataset is not None:
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                collate_fn=getattr(val_dataset, "collate_fn", None),
                num_workers=1,
                generator=generator,
                sampler=val_sampler,
                drop_last=drop_last
            )
        else:
            val_sampler = None
            val_loader = None
    else:
        train_sampler = None
        val_sampler = None
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            collate_fn=getattr(train_dataset, "collate_fn", None),
            shuffle=shuffle,
            generator=generator,
            persistent_workers=False,
            pin_memory=True,
            drop_last=drop_last,
        )
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                collate_fn=getattr(val_dataset, "collate_fn", None),
                shuffle=shuffle,
                generator=generator,
                drop_last=drop_last,
            )
        else:
            val_loader = None

    return train_loader, val_loader, train_sampler, val_sampler
