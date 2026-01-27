import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from models.model_wrapper import ModelWrapper

class GMMWrapper(ModelWrapper):
    def __init__(self, wrapped: nn.Module, num_modes=1, **kwargs):
        super(GMMWrapper, self).__init__()

        self.wrapped = wrapped
        self.num_modes = num_modes
        self.action_len = (self.wrapped.output_len - num_modes) // (num_modes * 2)

        self.generator = torch.Generator()
        self.generator.manual_seed(42)

    def forward(self, input):
        output = self.wrapped(input)
        logits = output[:, -self.num_modes:]
        output = output[:, :-self.num_modes]
        output = output.view((output.shape[0], self.num_modes, self.action_len * 2))

        weights = D.Categorical(logits=logits)
        mean = output[:, :, :self.action_len]
        scale_raw = output[:, :, -self.action_len:]

        scale = F.softplus(scale_raw) + 0.01

        gaussians = D.Independent(D.Normal(loc=mean, scale=scale), 1)
        gmm = D.MixtureSameFamily(
            mixture_distribution=weights,
            component_distribution=gaussians,
        )

        if self.wrapped.training:
            return gmm
        else:
            return gmm.sample()
