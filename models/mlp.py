import torch
import torch.nn as nn
from models.model_utils import set_attributes_from_args
from torch.nn.utils import spectral_norm

class MLP(nn.Module):
    def __init__(self, **kwargs):
        DEFAULT_MLP_CONFIG = {
            # Required, no default
            'input_len': None,
            'output_len': None,
            'device': None,

            'hidden_dims': [],
            'dropout_rate': 0.0,
            'batch_norm': False,
            'spectral_norm': False,
            'optimizer': 0
        }

        super(MLP, self).__init__()
        set_attributes_from_args(self, DEFAULT_MLP_CONFIG, kwargs)

        layers = []
        in_dim = self.input_len
        
        for hidden_dim in self.hidden_dims:
            linear_layer = nn.Linear(in_dim, hidden_dim)

            if self.spectral_norm:
                linear_layer = spectral_norm(linear_layer)

            layers.append(linear_layer)

            if self.batch_norm:
                layers.append(nn.LayerNorm(hidden_dim))
                
            layers.append(nn.ReLU())

            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(self.dropout_rate))

            in_dim = hidden_dim

        final_layer = nn.Linear(self.hidden_dims[-1] if len(self.hidden_dims) > 0 else self.input_len, self.output_len)

        if self.spectral_norm:
            final_layer = spectral_norm(final_layer)

        layers.append(final_layer.to(self.device))

        self.model = nn.Sequential(*layers).to(self.device)

        for param in self.model.parameters():
            param.optimizer = self.optimizer

    # Overload this to set local device attribute
    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)

        new_device = None
        if args:
            if isinstance(args[0], (torch.device, str, int)):
                new_device = torch.device(args[0])
        elif 'device' in kwargs:
            new_device = torch.device(kwargs['device'])

        if new_device is not None:
            self.device = new_device

        return result

    def forward(self, input):
        return self.model(input)
