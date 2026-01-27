import torch.nn as nn
from torch.nn.modules import Sequential
from constants import RESNET_SIZE
from models.model_utils import forward_with_checkpoint, set_attributes_from_args
import torch
import numpy as np
from torchvision import transforms
from models.model_wrapper import ModelWrapper
from r3m import load_r3m

default_transform = transforms.Compose([
                        transforms.Resize((224, 224), antialias=False),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])])

class R3M(ModelWrapper):
    def __init__(self, wrapped: nn.Module, **kwargs):
        DEFAULT_R3M_CONFIG = {
            'input_len': RESNET_SIZE,
            'device': None,
            'rgb_height': 224,
            'rgb_width': 224,
            'grayscale': False,
            'optimizer': 0,
            'model_class': 'resnet18'
        }

        super(R3M, self).__init__()
        set_attributes_from_args(self, DEFAULT_R3M_CONFIG, kwargs)
        assert isinstance(self.rgb_height, int) and isinstance(self.rgb_width, int), "Image dimensions must be integers!"
        assert self.model_class in ["resnet18", "resnet34"]

        self.wrapped = wrapped

        self.r3m = load_r3m(self.model_class).to(self.device)
        self.r3m = Sequential(*(list(self.r3m.module.convnet.children())))
        self.r3m.eval()

        self.output_len = wrapped.output_len

        for param in self.r3m.parameters():
            param.optimizer = self.optimizer

    def forward(self, input):
        return self.wrapped(self.frames_to_r3m(input))

    # Assumes frames dimensions [B, H, W, C] or batch-less [H, W, C]
    def frames_to_r3m(self, frames: np.ndarray | torch.Tensor) -> torch.Tensor:
        # Bad hack - wish I could use self.device, but this isn't stored properly when compiled for whatever reason
        device = next(self.r3m.parameters()).device
        with torch.set_grad_enabled(self.r3m.training):
            batch_size = 2 ** 9
            batches = np.ceil(len(frames) / batch_size).astype(np.uint64)
            r3m_features = torch.empty((len(frames), RESNET_SIZE), device=device)

            for i in range(batches):
                start = i * batch_size
                end = (i + 1) * batch_size
                curr_frames = frames[start:end]

                # If numpy array, turn into tensor and put on device
                if isinstance(curr_frames, np.ndarray):
                    curr_frames = torch.as_tensor(curr_frames, device=device)

                assert isinstance(curr_frames, torch.Tensor)

                # Needs to be reshaped: [H * W * C]
                if curr_frames.dim() == 1:
                    curr_frames = curr_frames.view(1, self.rgb_height, self.rgb_width, 1 if self.grayscale else 3)

                # Needs to be reshaped: [B, H * W * C]
                if curr_frames.dim() == 2:
                    curr_frames = curr_frames.view(curr_frames.shape[0], self.rgb_height, self.rgb_width, 1 if self.grayscale else 3)

                # Add batch dimension if necessary
                if curr_frames.dim() == 3:
                    curr_frames = curr_frames.unsqueeze(0)

                if self.grayscale:
                    # Grayscale -> RGB
                    curr_frames = curr_frames.repeat(1, 1, 1, 3)

                # [B, H, W, C] -> [B, C, H, W]
                # [0, 255] -> [0, 1]
                curr_frames = (curr_frames.permute(0, 3, 1, 2) / 255.0).to(torch.float32)

                curr_frames = default_transform(curr_frames)

                # Extract features
                if self.r3m.training:
                    r3m_features[start:end] = forward_with_checkpoint(self.r3m, curr_frames)
                else:
                    r3m_features[start:end] = self.r3m(curr_frames).squeeze(-1).squeeze(-1)
            return r3m_features
