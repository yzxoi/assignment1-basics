import math
import torch
import torch.nn as nn

class Linear(nn.Module):
    """
    A bias-less linear layer: y = x @ W
    Stores weight as W of shape (in_features, out_features).
    Initialized with N(0, σ²) truncated to ±3σ,
    where σ = sqrt(2 / (in_features + out_features)).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {"device": device, "dtype": dtype}
        # store weight as W (not W^T)
        self.W = nn.Parameter(torch.empty(in_features, out_features, **factory_kwargs))

        # truncated normal initialization
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        returns: (..., out_features)
        """
        return x @ self.W