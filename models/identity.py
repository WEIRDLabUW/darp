import torch.nn as nn
from models.model_utils import set_attributes_from_args

class Identity(nn.Module):
    def __init__(self, **kwargs):
        DEFAULT_IDENTITY_CONFIG = {
            'input_len': None,
        }

        super(Identity, self).__init__()
        set_attributes_from_args(self, DEFAULT_IDENTITY_CONFIG, kwargs)

        self.output_len = self.input_len

    def forward(self, input):
        return input
