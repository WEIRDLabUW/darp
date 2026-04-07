import copy
from typing import Tuple
import torch
import numpy as np
from torch.amp import GradScaler, autocast
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from models.model_factory import ModelFactory
from models.model_utils import set_attributes_from_args
from util import create_matrices, set_seed
from torch.utils.tensorboard import SummaryWriter
from logging_util import logger
import datetime
from models.model_utils import get_scalers_from_data_path
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

import pickle
import os
from types import SimpleNamespace


def save_model(model, optimizer, train_loader, path, sloppy=False, darp=False, is_diffusion=False, ema=None):
    if isinstance(model, DistributedDataParallel):
        model = model.module

    if sloppy:
        if darp:
            if is_diffusion:
                model.wrapped.wrapped.model = model.wrapped.wrapped.model._orig_mod
            else:
                model.wrapped = model.wrapped._orig_mod
            if hasattr(model, "set_transformer"):
                model.set_transformer = model.set_transformer._orig_mod
        else:
            model = model._orig_mod

    checkpoint = {
        'model': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'dataloader_rng_state': train_loader.generator.get_state()
    }
    
    if ema is not None:
        checkpoint['ema'] = ema.state_dict()

    torch.save(checkpoint, path)

    if sloppy and darp:
        model.compile()

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
            #num_workers=4,
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
                # num_workers=0,
                generator=generator,
                #pin_memory=True,
                drop_last=drop_last,
            )
        else:
            val_loader = None

    return train_loader, val_loader, train_sampler, val_sampler

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
            #print(f"{len(self.obs)=}, {len(self.rgb_obs)=}")
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

        # TODO: Check this input is reasonable (enum?)
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

def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def train_model(rank, world_size, env_cfg, policy_cfg, eval_trials=100, eval_epochs=0, start_eval_epoch=0, sloppy=False, eval_result_name=None):
    DEFAULT_CONFIG = {
        'force_retrain': True,
        'epochs': 100,
        'batch_size': 64,
        'shuffle': True,

        # Checkpoint loading
        'model_name': "model",
        'loaded_optimizer_dict': False,
        'loss_fn': 'mse'
    }
    logger.debug(f"GPU {rank + 1}/{world_size}")
    set_seed(env_cfg.get("seed", 42))
    torch.set_float32_matmul_precision('high')

    # If the process group is already initialized, something else started it and needs it, so don't kill at the end
    start_process_group = not dist.is_initialized()
    if world_size > 1:
        # Common fixes for NCCL hangs
        os.environ["NCCL_P2P_DISABLE"] = "1"
        # os.environ["NCCL_IB_DISABLE"] = "1"
        # os.environ["NCCL_DEBUG"] = "INFO"

        torch.cuda.set_device(rank)
        if start_process_group:
            dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30))

    device = f"cuda:{rank}"
    if 'device' not in env_cfg:
        env_cfg['device'] = device

    config = SimpleNamespace()
    set_attributes_from_args(config, DEFAULT_CONFIG, policy_cfg['train_config'] | env_cfg)

    model_cfg = policy_cfg['model_config']
    darp = model_cfg.get("darp", False)
    is_diffusion = policy_cfg['model_config']['type'] == 'diffusion'
    obs_horizon = model_cfg.get("obs_horizon", 1)
    act_horizon = model_cfg.get("act_horizon", 1)

    log_dir = os.path.join("tensorboard_logs", f"{config.model_name}")
    writer = SummaryWriter(log_dir) if rank == 0 else None

    config.model_name = f"model_checkpoints/{config.model_name}.pth"

    if rank == 0:
        os.makedirs(os.path.dirname(config.model_name), exist_ok=True)

    if 'demo_pkl' in env_cfg:
        model_cfg['demo_pkl'] = env_cfg['demo_pkl']
    if 'rgb_demo_pkl' in env_cfg:
        model_cfg['rgb_demo_pkl'] = env_cfg['rgb_demo_pkl']

    # TODO: Probably don't want to expose all of this
    model_cfg['env_cfg'] = env_cfg

    model_cfg['device'] = env_cfg['device']
    model = ModelFactory(model_cfg).create()
    if rank == 0:
        logger.info(model)

    # Check if the model already exists
    if (os.path.exists(config.model_name) and not config.force_retrain):
        if 'dino' in model_cfg['type'] or model_cfg.get("sideload_dino", False):
            # We have to do this to ensure we can save/load our model
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="xFormers is not available*")
                _ = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)

        # Load the model if it exists
        logger.info(f"Skipping training phase, loading model from {config.model_name}")
        checkpoint = torch.load(config.model_name, weights_only=False)

        model.load_state_dict(checkpoint['model'])
        model.to(env_cfg['device'])

        if darp and (obs_horizon > 1 or act_horizon > 1):
            # Set to 1 so that dataset constructor uses lookback
            model_cfg["obs_horizon"] = 1
            model_cfg["act_horizon"] = 1

        train_dataset, _ = create_dataset(env_cfg, model_cfg)
        train_loader, _, _, _ = create_dataloader(train_dataset, None, rank, 1 if darp else world_size, config.batch_size, shuffle=config.shuffle)
        if darp and (obs_horizon > 1 or act_horizon > 1):
            # Set back to what it was
            model_cfg["obs_horizon"] = obs_horizon
            model_cfg["act_horizon"] = act_horizon
        model.eval()
        if darp:
            model.prepare_to_train(train_loader)
        train_index_dataset = IndexActionBCDataset(env_cfg['demo_pkl'], act_dataset=train_dataset)
        if obs_horizon > 1 or act_horizon > 1:
            train_index_dataset = ChunkingWrapper(obs_horizon, act_horizon, train_index_dataset)

            model.create_horizon_mapping(train_index_dataset)

        return model, -1

    if darp and (obs_horizon > 1 or act_horizon > 1):
        # Set to 1 so that dataset constructor uses lookback
        model_cfg["obs_horizon"] = 1
        model_cfg["act_horizon"] = 1

    train_dataset, val_dataset = create_dataset(env_cfg, model_cfg)
    train_loader, val_loader, train_sampler, val_sampler = create_dataloader(train_dataset, val_dataset, rank, 1 if darp else world_size, config.batch_size, shuffle=config.shuffle, drop_last=is_diffusion and not darp)

    if darp and (obs_horizon > 1 or act_horizon > 1):
        # Set back to what it was
        model_cfg["obs_horizon"] = obs_horizon
        model_cfg["act_horizon"] = act_horizon

    model.train()
    model.to(device)

    loss_from_forward = False
    if config.loss_fn == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif config.loss_fn == 'loss_from_forward':
        loss_from_forward = True
        criterion = None
    elif config.loss_fn == 'log_likelihood':
        criterion = None
    else:
        criterion = nn.MSELoss(reduction=('mean'))

    optimizers = []
    scalers = []

    for optimizer_idx, optimizer_cfg_kwargs in policy_cfg['train_config']['optimizers'].items():
        DEFAULT_CONFIG = {
            'lr': 1e-4,
            'weight_decay': 1e-6,
            'eps': 1e-8,
        }

        optimizer_cfg = SimpleNamespace()
        set_attributes_from_args(optimizer_cfg, DEFAULT_CONFIG, optimizer_cfg_kwargs)
        params = []
        for param in model.parameters():
            if param.optimizer == optimizer_idx and param.requires_grad:
                params.append(param)

        optimizer = optim.AdamW(params, lr=float(optimizer_cfg.lr), weight_decay=float(optimizer_cfg.weight_decay), eps=float(optimizer_cfg.eps), amsgrad=False, fused=sloppy, foreach=not sloppy)

        optimizers.append(optimizer)
        scalers.append(GradScaler('cuda', enabled=sloppy))

    assert len(optimizers) > 0
    logger.info(f"{sloppy=}")

    if config.loaded_optimizer_dict is not False:
        optimizer.load_state_dict(config.loaded_optimizer_dict)

    if is_diffusion:
        num_warmup_steps = policy_cfg['train_config'].get('num_warmup_steps', 500)
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=config.epochs * len(train_loader)
        )
        ema = EMAModel(
            parameters=model.model.parameters(),
            power=0.75)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizers[0],
            mode='min',
            factor=0.5,
            patience=5,
        )

    if darp:
        model.prepare_to_train(train_loader)
        if val_loader is not None:
            model.validation = True
            model.prepare_to_train(val_loader)
            model.validation = False
        train_index_dataset = IndexActionBCDataset(env_cfg['demo_pkl'], act_dataset=train_dataset)
        val_index_dataset = IndexActionBCDataset(env_cfg['val_demo_pkl'], act_dataset=val_dataset) if val_loader is not None else None
        if obs_horizon > 1 or act_horizon > 1:
            train_index_dataset = ChunkingWrapper(obs_horizon, act_horizon, train_index_dataset)

            if val_index_dataset is not None:
                val_index_dataset = ChunkingWrapper(obs_horizon, act_horizon, val_index_dataset)
            model.create_horizon_mapping(train_index_dataset)
        train_loader, val_loader, train_sampler, val_sampler = create_dataloader(train_index_dataset, val_index_dataset, rank, world_size, config.batch_size, shuffle=config.shuffle, drop_last=is_diffusion)

    if sloppy:
        torch._dynamo.reset()
        if darp:
            model.compile()
        else:
            model = torch.compile(model, mode="reduce-overhead")

    _, action_scaler = get_scalers_from_data_path(env_cfg['demo_pkl'])
    action_scaler.to_device(device)

    action_mean = action_scaler.mean_torch
    action_scale = action_scaler.scale_torch

    import sys
    print(f"[Rank {rank}] Entering DDP constructor...", flush=True)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank])
    print(f"[Rank {rank}] DDP constructor finished.", flush=True)

    best_val_loss = float('inf')
    early_stopping_patience = 13
    early_stopping_counter = 0

    best_score = 0

    for epoch in range(config.epochs):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        # Training phase
        train_loss = 0.0
        num_train_batches = 0
        for states, actions in train_loader:
            actions = actions.to(device).contiguous()
            if not is_diffusion:
                actions = (actions - action_mean) / action_scale

            if not isinstance(train_loader.dataset, IndexActionBCDataset) and not (isinstance(train_loader.dataset, ChunkingWrapper) and isinstance(train_loader.dataset.wrapped, IndexActionBCDataset)):
                states = states.to(device).contiguous()

                if is_diffusion:
                    states = states.reshape(len(states), -1)
                    actions = actions.reshape(len(actions), -1)
                    states = torch.hstack((states, actions))
            elif is_diffusion or policy_cfg['model_config']['retrieval_config'].get('ic_regularizer', False):
                if isinstance(train_loader.dataset, ChunkingWrapper):
                    states = states.to(device)
                    states = torch.hstack((states.reshape(len(states), -1), actions.reshape(len(states), -1)))
                else:
                    states = torch.hstack((torch.tensor(states, device=device).unsqueeze(1), actions.reshape(len(states), -1)))

            with autocast('cuda', dtype=torch.bfloat16, enabled=sloppy):
                if loss_from_forward:
                    loss = model(states)
                elif config.loss_fn == "log_likelihood":
                    distribution = model(states)
                    loss = (-distribution.log_prob(actions)).mean()
                else:
                    predicted_actions = model(states)
                    if config.loss_fn == "cross_entropy":
                        actions = actions.squeeze()

                    loss = criterion(predicted_actions, actions)

            for optimizer, scaler in zip(optimizers, scalers):
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if is_diffusion:
                    scheduler.step()
                    ema.step(model.parameters())

            train_loss += loss.detach()

            num_train_batches += 1
            del states

        avg_train_loss = train_loss / num_train_batches

        if rank == 0:
            writer.add_scalars('Losses', {
                'train': avg_train_loss,
            }, epoch)

        # Validation phase
        if val_loader is not None:
            if isinstance(model, DistributedDataParallel):
                model.module.validation = True
            else:
                model.validation = True
            torch_rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            if world_size > 1:
                val_sampler.set_epoch(epoch)

            if not is_diffusion:
                model.eval()

            val_loss = 0.0

            if is_diffusion:
                val_noise_loss = 0.0

            num_val_batches = 0
            with torch.no_grad():
                for states, actions in val_loader:
                    actions = actions.to(device).contiguous()
                    if not isinstance(val_loader.dataset, IndexActionBCDataset):
                        states = states.to(device).contiguous()

                        if is_diffusion:
                            states = states.reshape(len(states), -1)
                            actions = actions.reshape(len(actions), -1)
                            states = torch.hstack((states, actions))

                        loss = model(states)
                    else:
                        predicted_actions = model(states)
                        if config.loss_fn == "cross_entropy":
                            actions = actions.squeeze()
                        loss = criterion(predicted_actions, actions)

                    val_loss += loss.detach()
                    num_val_batches += 1
                
                avg_val_loss = val_loss / num_val_batches
                if rank == 0:
                    writer.add_scalars('Losses', {
                        'val': avg_val_loss,
                    }, epoch)

                if is_diffusion:
                    avg_val_noise_loss = val_noise_loss / num_val_batches

                if world_size > 1 and dist.is_initialized():
                    tensor_val_loss = torch.tensor([avg_val_loss], device=device)
                    dist.all_reduce(tensor_val_loss, op=dist.ReduceOp.SUM)
                    tensor_val_loss /= world_size
                    avg_val_loss = tensor_val_loss.item() 
                
                if not is_diffusion:
                    scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stopping_counter = 0

                    logger.info(f"New best val {best_val_loss:.4f}, saving...")
                    best_check = {
                        'model': copy.deepcopy(model.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                        'dataloader_rng_state': train_loader.generator.get_state()
                    }
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stopping_patience:
                    logger.info(f'Recommend early stopping after {epoch+1 - early_stopping_patience} epochs')
                    model.load_state_dict(best_check['model'])

                    save_model(model, optimizer, train_loader, config.model_name, sloppy=sloppy, darp=darp, is_diffusion=is_diffusion, ema=ema if is_diffusion else None)

                    if eval_epochs == 0:
                        best_score = best_val_loss

                    return model, best_score
                
                if rank == 0:
                    if is_diffusion:
                        logger.info(f"Epoch [{epoch + 1}/{config.epochs}], LR {optimizer.param_groups[0]['lr']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Noise Loss: {avg_val_noise_loss:.4f}")
                    else:
                        logger.info(f"Epoch [{epoch + 1}/{config.epochs}], LR {optimizer.param_groups[0]['lr']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            model.train()
            if isinstance(model, DistributedDataParallel):
                model.module.validation = False
            else:
                model.validation = False
            torch.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state)
        elif rank == 0:
            logger.info(f"{rank}: Epoch [{epoch + 1}/{config.epochs}], Train Loss: {avg_train_loss}")

        if eval_epochs != 0 and (epoch > 0 or eval_epochs == 1) and ((epoch + 1) % eval_epochs == 0) and epoch + 1 >= start_eval_epoch:
            if rank == 0:
                logger.info(f"Evaluating epoch {epoch + 1}...")
                save_model(model, optimizer, train_loader, config.model_name, sloppy=sloppy, darp=darp, is_diffusion=is_diffusion, ema=ema if is_diffusion else None)

            # Crucial - have to unwrap module or batched eval will fail
            if world_size > 1:
                torch.distributed.barrier()

                eval_model = ModelFactory(model_cfg).create()
                checkpoint = torch.load(config.model_name, weights_only=False)
                eval_model.load_state_dict(checkpoint['model'])
                eval_model.to(env_cfg['device'])
                eval_model.eval()

                if darp:
                    eval_model.horizon_mapping = model.module.horizon_mapping

                torch_rng_state = torch.get_rng_state()
                cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

                from eval import batched_eval   
                score = batched_eval(env_cfg, eval_model, trials=eval_trials, reset=True, darp=darp, results=(eval_result_name + "_temp") if eval_result_name is not None else None, trials_per_worker=(2 if is_diffusion and eval_trials > 50 else 1))
                if score > best_score:
                    best_score = score
                    if rank == 0:
                        logger.info(f"New best score {score} found at epoch {epoch + 1}")
                        
                        if eval_result_name is not None:
                            with open(f"results/{eval_result_name}_temp.pkl", 'rb') as f:
                                best_data = pickle.load(f)
                            with open(f"results/{eval_result_name}.pkl", 'wb') as f:
                                pickle.dump(best_data, f)

                if rank == 0:
                    writer.add_scalars('Scores', {
                        'score': score,
                    }, epoch)
                        
                torch.distributed.barrier()

                torch.set_rng_state(torch_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state)
            else:
                eval_world_size = int(os.environ.get('SLURM_NTASKS', 1))

                eval_model = ModelFactory(model_cfg).create()
                free_memory, total_memory = torch.cuda.mem_get_info()
                used_memory = total_memory - free_memory
                logger.debug(f"Allocated memory pre-eval: {(used_memory / (1024**2)):.2f} / {(total_memory / (1024**2)):.2f} MB")
                checkpoint = torch.load(config.model_name, weights_only=False)
                eval_model.load_state_dict(checkpoint['model'])
                eval_model.to(env_cfg['device'])
                eval_model.eval()
                if sloppy:
                   torch._dynamo.reset()
                   if darp:
                       eval_model.compile()
                   else:
                       eval_model = torch.compile(eval_model, mode="reduce-overhead")

                import sys
                import subprocess

                free_port = find_free_port()

                cmd = [
                    "python", "-m", "torch.distributed.launch",
                    f"--nproc_per_node={eval_world_size}",
                    f"--master_port={free_port}",
                    "eval_model.py"
                ] + sys.argv[1:] + ["--results_file_name=results"] + (["--batched"]) + (["--darp"] if darp else []) + [f"--trials={eval_trials}"]

                result = subprocess.run(cmd)

                if result.returncode != 0:
                    raise SystemExit(1)

                writer.add_scalars('Scores', {
                    'mean': torch.mean(torch.tensor(pickle.load(open("results/results.pkl", 'rb')))),
                }, epoch)
            model.train()

    if rank == 0:
        save_model(model, optimizer, train_loader, config.model_name, sloppy=sloppy, darp=darp, is_diffusion=is_diffusion, ema=ema if is_diffusion else None)

    if world_size > 1 and start_process_group:
        dist.destroy_process_group()
    if eval_epochs == 0 and val_loader is not None:
        best_score = best_val_loss

    return model, best_score

def launch_train_parallel(env_cfg, policy_cfg, force_nonparallel=False, eval_epochs=0, start_eval_epoch=0, sloppy=False):
    import torch.multiprocessing as mp

    world_size = torch.cuda.device_count()
    if world_size > 1 and not force_nonparallel:
        mp.set_start_method('spawn', force=True)
        master_port = str(find_free_port())
        # os.environ['NCCL_DEBUG'] = 'INFO'
        # os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = master_port
        os.environ['TORCH_DISTRIBUTED_MASTER_PORT'] = master_port
        logger.info(f"Training with {world_size} GPUs")
        mp.spawn(train_model, args=(world_size, env_cfg, policy_cfg, eval_epochs, start_eval_epoch, sloppy), nprocs=world_size)
    else:
        train_model(0, 1, env_cfg, policy_cfg, eval_epochs, start_eval_epoch, sloppy)

if __name__ == "__main__":
    from argparse import ArgumentParser
    import yaml

    parser = ArgumentParser()
    parser.add_argument("env_config_path", help="Path to environment config file")
    parser.add_argument("policy_config_path", help="Path to policy config file")
    parser.add_argument("--force-nonparallel", action='store_true', default=False,)
    parser.add_argument("--eval-epochs", type=int, default=0)
    parser.add_argument("--eval-trials", type=int, default=100)
    parser.add_argument("--start-eval-epoch", type=int, default=0)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--sloppy", action="store_true")
    args, _ = parser.parse_known_args()

    with open(args.env_config_path, 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.policy_config_path, 'r') as f:
        policy_cfg = yaml.load(f, Loader=yaml.FullLoader)

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1 and not args.force_nonparallel:
        logger.info(f"Training with {world_size} GPUs")
        train_model(rank, world_size, env_cfg, policy_cfg, eval_epochs=args.eval_epochs, eval_trials=args.eval_trials, start_eval_epoch=args.start_eval_epoch, sloppy=args.sloppy)
    else:
        train_model(0, 1, env_cfg, policy_cfg, eval_epochs=args.eval_epochs, eval_trials=args.eval_trials, start_eval_epoch=args.start_eval_epoch, sloppy=args.sloppy)
