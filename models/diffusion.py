import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from models.model_utils import set_attributes_from_args
from typing import Tuple, Sequence, Dict, Union, Optional
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x


class DiffusionPolicy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Configuration
        # ------------------------------------------------------------------
        # Default configuration matches 'real-stanford' best practices for low-dim
        DEFAULT_CONFIG = {
            # Input / Output dimensions (Required)
            'input_len': None,   # Observation dimension
            'output_len': None,  # Action dimension
            'device': None,

            # Diffusion Physics
            'num_train_steps': 100,
            'num_inference_steps': 100,
            'beta_schedule': 'squaredcos_cap_v2', # 'linear' or 'squaredcos_cap_v2'
            'prediction_type': 'epsilon',         # 'epsilon' or 'sample'
            'clip_sample': True,

            # Network Architecture (ConditionalUnet1D defaults)
            'down_dims': [256, 512, 1024],
            'kernel_size': 5,
            'n_groups': 8,
            'diffusion_step_embed_dim': 256,
            'obs_dropout': 0.0,
            'obs_noise_std': 0.0,

            # Training details
            'act_horizon': 1,
            'obs_horizon': 1,
            'optimizer': 0,
        }
        
        set_attributes_from_args(self, DEFAULT_CONFIG, kwargs)
        
        self.action_dim = self.output_len
        self.obs_dim = self.input_len - self.output_len

        if self.obs_dropout > 0:
            self.drop = nn.Dropout(self.obs_dropout)
        else:
            self.drop = nn.Identity()
        
        # ------------------------------------------------------------------
        # 2. The Canonical Model (ConditionalUnet1D)
        # ------------------------------------------------------------------
        # This replaces your custom 'NoisePredictionNet'
        self.model = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim * self.obs_horizon,
            diffusion_step_embed_dim=self.diffusion_step_embed_dim,
            down_dims=self.down_dims,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
        )

        # ------------------------------------------------------------------
        # 3. The Canonical Scheduler
        # ------------------------------------------------------------------
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_steps,
            #beta_start=0.0001,
            #beta_end=0.02,
            beta_schedule=self.beta_schedule,
            clip_sample=self.clip_sample,
            prediction_type=self.prediction_type,
        )

        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        for param in self.model.parameters():
            param.optimizer = self.optimizer

        # Just for debugging
        self.epoch = 0

    def to(self, *args, **kwargs):
        """
        Override to ensure scheduler states (if any) move to device.
        Most Diffusers schedulers are stateless/bufferless, but this is safe.
        """
        super().to(*args, **kwargs)
        if hasattr(self, 'device'):
             # Try to detect device from args if provided, otherwise use self.device
            device = kwargs.get('device', None)
            if device is None and len(args) > 0 and isinstance(args[0], (torch.device, str)):
                device = args[0]
            
            if device is not None:
                self.device = device
        return self

    # ------------------------------------------------------------------
    # 4. Forward Pass (Training & Inference Wrapper)
    # ------------------------------------------------------------------
    def forward(self, input_tensor):
        """
        Args:
            input_tensor: (B, ObsDim + ActionDim) for training
                          (B, ObsDim) for inference
        """

        # --- TRAINING MODE ---
        if self.training:
            self.epoch += 1
            # 1. Slice Inputs (Your style)
            # Assume last 'output_len' columns are the Action
            obs = input_tensor[:, :-self.act_horizon * self.output_len]
            actions = input_tensor[:, -self.act_horizon * self.output_len:]
            
            # Apply observation dropout
            obs = self.drop(obs)
            
            # Apply observation noise augmentation
            if self.obs_noise_std > 0:
                obs = obs + torch.randn_like(obs) * self.obs_noise_std

            # 2. Prepare Data
            # Reshape for 1D Convolution: (B, Dim) -> (B, Dim, Horizon=1)
            # The canonical Unet expects (B, Dim, Horizon)
            trajectory = actions.reshape(len(input_tensor), -1, self.output_len)
            global_cond = obs

            # 3. Sample Noise
            noise = torch.randn(trajectory.shape, device=trajectory.device)

            # 4. Sample Timesteps
            bsz = trajectory.shape[0]
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=trajectory.device
            ).long()

            # 5. Add Noise (Forward Diffusion)
            noisy_trajectory = self.noise_scheduler.add_noise(
                original_samples=trajectory,
                noise=noise,
                timesteps=timesteps
            )

            # 6. Predict Noise (Model Forward)
            pred = self.model(
                sample=noisy_trajectory,
                timestep=timesteps,
                global_cond=global_cond
            )

            # 7. Calculate Loss
            # The target depends on prediction_type (usually 'epsilon')
            target = noise
            if self.noise_scheduler.config.prediction_type == 'sample':
                target = trajectory
            elif self.noise_scheduler.config.prediction_type == 'v_prediction':
                target = self.noise_scheduler.get_velocity(trajectory, noise, timesteps)

            loss = F.mse_loss(pred, target, reduction='mean')
            return loss

        # --- INFERENCE MODE ---
        else:
            # 1. Prepare Observation
            obs = input_tensor
            batch_size = obs.shape[0]

            # 2. Initialize Noisy Action (Latent)
            # Shape: (B, ActionDim, Horizon)
            latents = torch.randn(
                (batch_size, self.act_horizon, self.action_dim), 
                device=self.device
            )

            # 3. Setup Scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_steps)

            # 4. Denoising Loop
            for t in self.noise_scheduler.timesteps:
                # a. Apply Model
                # Model inputs must be on correct device
                t_input = t.to(self.device) 
                
                noise_pred = self.model(
                    sample=latents,
                    timestep=t_input,
                    global_cond=obs
                )

                # b. Scheduler Step
                # inverse_step for DDIM, step for DDPM
                latents = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latents
                ).prev_sample


            # 5. Final Processing
            # Return full predicted trajectory [Batch, Horizon, ActionDim]
            action = latents
            
            # Clip to valid action range if strictly required ([-1, 1])
            if self.clip_sample:
                action = torch.clamp(action, -1, 1)
                
            return action
