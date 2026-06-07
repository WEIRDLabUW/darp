import torch

class FastScaler:
    def __init__(self):
        self.mean_torch = None
        self.scale_torch = None

    def fit(self, X):
        if isinstance(X, list):
            if len(X) > 0 and isinstance(X[0], torch.Tensor):
                X = torch.stack(X)
            else:
                X = np.asarray(X)

        if not isinstance(X, torch.Tensor):
            import numpy as np
            mean_np = np.mean(X, axis=0, dtype=X.dtype)
            scale_np = np.std(X, axis=0, dtype=X.dtype)

            if np.ndim(scale_np) == 0:
                scale_np = np.array([scale_np])

            # Avoid division by zero
            scale_np[scale_np == 0] = 1

            self.mean_torch = torch.as_tensor(mean_np)
            self.scale_torch = torch.as_tensor(scale_np)

            return self

        self.mean_torch = torch.mean(X, dim=0, dtype=X.dtype).to(X.device)
        self.scale_torch = torch.std(X, dim=0, correction=0).to(X.dtype).to(X.device)

        if self.scale_torch.dim() == 0:
            self.scale_torch = self.scale_torch.unsqueeze(0)

        # Avoid division by zero
        self.scale_torch = torch.where(
            self.scale_torch == 0,
            torch.tensor(1.0, dtype=X.dtype, device=X.device),
            self.scale_torch
        )

        return self

    def transform(self, X):
        assert self.mean_torch is not None and self.scale_torch is not None

        if isinstance(X, list):
            if len(X) > 0 and isinstance(X[0], torch.Tensor):
                X = torch.stack(X)
            else:
                X = torch.tensor(X, dtype=self.mean_torch.dtype, device=self.mean_torch.device)
        elif not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.mean_torch.dtype, device=self.mean_torch.device)
        else:
            X = X.to(self.mean_torch.device)

        return (X - self.mean_torch) / self.scale_torch

    def inverse_transform(self, X):
        assert self.mean_torch is not None and self.scale_torch is not None

        if isinstance(X, list):
            if len(X) > 0 and isinstance(X[0], torch.Tensor):
                X = torch.stack(X)
            else:
                X = torch.tensor(X, dtype=self.mean_torch.dtype, device=self.mean_torch.device)
        elif not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=self.mean_torch.dtype, device=self.mean_torch.device)
        else:
            X = X.to(self.mean_torch.device)

        return X * self.scale_torch + self.mean_torch

    def to_device(self, device):
        if self.mean_torch is not None:
            self.mean_torch = self.mean_torch.to(device)
        if self.scale_torch is not None:
            self.scale_torch = self.scale_torch.to(device)

class IdentityScaler(FastScaler):
    def __init__(self, len, dtype, device):
        super().__init__()
        self.mean_torch = torch.zeros(len, dtype=dtype, device=device)
        self.scale_torch = torch.ones(len, dtype=dtype, device=device)

def combine_scalers(scaler1: FastScaler, scaler2: FastScaler) -> FastScaler:
    combined_scaler = FastScaler()
    combined_scaler.mean_torch = torch.hstack((scaler1.mean_torch, scaler2.mean_torch))
    combined_scaler.scale_torch = torch.hstack((scaler1.scale_torch, scaler2.scale_torch))
    return combined_scaler
