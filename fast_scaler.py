from numba import jit
import numpy as np
import torch

class FastScaler:
    def __init__(self):
        self.mean_np = None
        self.scale_np = None
        self.mean_torch = None
        self.scale_torch = None

    def fit(self, X):
        if isinstance(X, list):
            if isinstance(X[0], torch.Tensor):
                X = torch.as_tensor(X[0]).unsqueeze(0)
            else:
                X = np.asarray(X)

        if isinstance(X, torch.Tensor):
            self.mean_torch = torch.mean(X, dim=0, dtype=X.dtype).to(X.device)
            self.scale_torch = torch.std(X, dim=0).to(X.dtype).to(X.device)

            if self.scale_torch.dim() == 0:
                self.scale_torch = self.scale_torch.unsqueeze(0)

            # Avoid division by zero
            self.scale_torch = torch.where(self.scale_torch == 0, torch.tensor(1.0, dtype=X.dtype), self.scale_torch)

            self.mean_np = np.asarray(self.mean_torch.detach().cpu())
            self.scale_np = np.asarray(self.scale_torch.detach().cpu())
        else:
            self.mean_np = np.mean(X, axis=0, dtype=X.dtype)
            self.scale_np = np.std(X, axis=0, dtype=X.dtype)

            if np.ndim(self.scale_np) == 0:
                self.scale_np = np.array([self.scale_np])

            # Avoid division by zero
            self.scale_np[self.scale_np == 0] = 1

            self.mean_torch = torch.as_tensor(self.mean_np)
            self.scale_torch = torch.as_tensor(self.scale_np)

        return self

    @staticmethod
    @jit(nopython=True)
    def _transform_numpy(X, mean, scale):
        return (X - mean) / scale

    @staticmethod
    @jit(nopython=True)
    def _inverse_transform_numpy(X, mean, scale):
        return X * scale + mean

    def _transform_torch(self, X, mean, scale):
        return (X - mean) / scale

    def _inverse_transform_torch(self, X, mean, scale):
        return X * scale + mean

    def transform(self, X):
        assert self.mean_torch is not None and self.scale_torch is not None

        if isinstance(X, list):
            if isinstance(X[0], torch.Tensor):
                X = torch.as_tensor(X[0]).unsqueeze(0)
            else:
                X = np.asarray(X)

        if isinstance(X, torch.Tensor):
            return self._transform_torch(X, self.mean_torch, self.scale_torch)
        else:
            return self._transform_numpy(X, self.mean_np, self.scale_np)

    def inverse_transform(self, X):
        assert self.mean_torch is not None and self.scale_torch is not None

        if isinstance(X, list):
            if isinstance(X[0], torch.Tensor):
                X = torch.as_tensor(X)
            else:
                X = np.asarray(X)

        if isinstance(X, torch.Tensor):
            return self._inverse_transform_torch(X, self.mean_torch, self.scale_torch)
        else:
            return self._inverse_transform_numpy(X, self.mean_np, self.scale_np)

    def to_device(self, device):
        self.mean_torch = self.mean_torch.to(device)
        self.scale_torch = self.scale_torch.to(device)

class IdentityScaler(FastScaler):
    def __init__(self, len, dtype, device):
        super(FastScaler, self).__init__()
        self.mean_torch = torch.zeros(len, dtype=dtype, device=device)
        self.scale_torch = torch.ones(len, dtype=dtype, device=device)

        self.mean_np = np.asarray(self.mean_torch.detach().cpu())
        self.scale_np = np.asarray(self.scale_torch.detach().cpu())

def combine_scalers(scaler1: FastScaler, scaler2: FastScaler) -> FastScaler:
    combined_scaler = FastScaler()
    combined_scaler.mean_torch = torch.hstack((scaler1.mean_torch, scaler2.mean_torch))
    combined_scaler.scale_torch = torch.hstack((scaler1.scale_torch, scaler2.scale_torch))
    combined_scaler.mean_np = np.hstack((scaler1.mean_np, scaler2.mean_np))
    combined_scaler.scale_np = np.hstack((scaler1.scale_np, scaler2.scale_np))
    return combined_scaler
