from fast_scaler import FastScaler
import torch.nn as nn
import torch

from models.model_wrapper import ModelWrapper

class ScaleWrapper(ModelWrapper):
    def __init__(self, wrapped: nn.Module, input_scaler: FastScaler | None, output_scaler: FastScaler | None):
        super(ScaleWrapper, self).__init__()

        self.wrapped = wrapped
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

    def forward(self, input):
        if self.input_scaler is not None:
            input = self.input_scaler.transform(input)

        output = self.wrapped(input)

        if self.output_scaler is not None and not self.wrapped.training:
            output = self.output_scaler.inverse_transform(output)

        return output

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)

        if self.input_scaler:
            self.input_scaler.to_device(self.device)
        if self.output_scaler:
            self.output_scaler.to_device(self.device)

        return result

    def __getattr__(self, name):
        wrapped = self._modules['wrapped']
        if name == "wrapped":
            return wrapped
        try:
            return getattr(wrapped, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object and its wrapped module have no attribute '{name}'")

class DiffusionScaleWrapper(nn.Module):
    def __init__(self, wrapped: nn.Module, obs_min, obs_max, obs_len, action_min, action_max, action_len, obs_horizon, action_horizon, darp=False):
        super(DiffusionScaleWrapper, self).__init__()

        self.wrapped = wrapped
        self.obs_min = obs_min.to(wrapped.device)
        self.obs_max = obs_max.to(wrapped.device)
        self.obs_len = obs_len
        self.action_min = action_min.to(wrapped.device)
        self.action_max = action_max.to(wrapped.device)
        self.action_len = action_len

        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.darp = darp

        if self.darp:
            diff_max = self.obs_max - self.obs_min
            diff_min = self.obs_min - self.obs_max
            self.obs_len = 2 * self.obs_len + self.action_len
            self.obs_min = torch.concat((self.obs_min, self.action_min, diff_min))
            self.obs_max = torch.concat((self.obs_max, self.action_max, diff_max))

    def forward(self, input):
        input = input.clone()
        if self.wrapped.model.training:
            obs = input[:, :-self.action_len * self.action_horizon]
            obs = obs.reshape(len(input) * self.obs_horizon, -1)
            obs = ((obs - self.obs_min) / (self.obs_max - self.obs_min)) * 2 - 1
            input[:, :-self.action_len * self.action_horizon] = obs.reshape(len(input), -1)

            actions = input[:, -self.action_len * self.action_horizon:]
            actions = actions.reshape(len(input) * self.action_horizon, -1)
            actions = ((actions - self.action_min) / (self.action_max - self.action_min)) * 2 - 1
            input[:, -self.action_len * self.action_horizon:] = actions.reshape(len(input), -1)

            # At training time, this will be noise loss - don't touch it
            output = self.wrapped(input)
        else:
            obs = input
            obs = obs.reshape(len(input) * self.obs_horizon, -1)
            obs = ((obs - self.obs_min) / (self.obs_max - self.obs_min)) * 2 - 1
            input = obs.reshape(len(input), -1)

            output = self.wrapped(input)

            output = ((output + 1) / 2) * (self.action_max - self.action_min) + self.action_min

        return output

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)

        self.wrapped.to(*args, **kwargs)

        new_device = None
        if args:
            if isinstance(args[0], (torch.device, str, int)):
                new_device = torch.device(args[0])
        elif 'device' in kwargs:
            new_device = torch.device(kwargs['device'])

        self.action_min = self.action_min.to(*args, **kwargs)
        self.action_max = self.action_max.to(*args, **kwargs)
        self.obs_min = self.obs_min.to(*args, **kwargs)
        self.obs_max = self.obs_max.to(*args, **kwargs)

        return result

    def __getattr__(self, name):
        wrapped = self._modules['wrapped']
        if name == "wrapped":
            return wrapped
        try:
            return getattr(wrapped, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object and its wrapped module have no attribute '{name}'")
