import torch.nn as nn
import torch

class ModelWrapper(nn.Module):
    def __init__(self, **kwargs):
        super(ModelWrapper, self).__init__()

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)

        for submodule in self._modules.values():
            submodule.to(*args, **kwargs)

        new_device = None
        if args:
            if isinstance(args[0], (torch.device, str, int)):
                new_device = torch.device(args[0])
        elif 'device' in kwargs:
            new_device = torch.device(kwargs['device'])

        if new_device is not None:
            self.device = new_device

        return result

    def __getattr__(self, name):
        # Due to weird PyTorch behavior, we have to return any wrapped modules like this
        if name in self._modules:
            return self._modules[name]

        # Then, check every wrapped module for an attribute
        for submodule in self._modules.values():
            try:
                return getattr(submodule, name)
            except:
                pass

        # If we get here, no submodules had the attribute
        raise AttributeError(f"'{type(self).__name__}' object nor its wrapped module have attribute '{name}'")


    def train(self, mode=True):
        super().train(mode)
        for submodule in self._modules.values():
            submodule.train(mode)

    def eval(self):
        super().eval()
        for submodule in self._modules.values():
            submodule.eval()

