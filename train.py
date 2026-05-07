import copy
from typing import Tuple
import torch
import numpy as np
from torch.amp import GradScaler, autocast
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import sys
import subprocess
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from models.model_factory import ModelFactory
from models.model_utils import set_attributes_from_args
from datasets import create_dataset, create_dataloader
from util import create_matrices, set_seed, find_free_port
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
                eval_world_size = int(os.environ.get('SLURM_NTASKS', torch.cuda.device_count() or 1))

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

                cmd = [
                    "torchrun",
                    f"--nproc_per_node={eval_world_size}",
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
