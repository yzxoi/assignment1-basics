import math
import torch
import torch.nn as nn

from .linear import Linear
from .utils import SiLU

class FeedForward(nn.Module):
    """
    SwiGLU position-wise feed-forward network.

    Implements: FFN(x) = W2( SiLU(W1 x) * (W3 x) )
    where d_ff ≈ 8/3 * d_model, rounded up to a multiple of 64.
    """
    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        if d_ff is None:
            # compute intermediate dimension
            raw_dff = 8 * d_model / 3
            # round up to next multiple of 64
            d_ff = int(math.ceil(raw_dff / 64) * 64)
        factory_kwargs = {"device": device, "dtype": dtype}
        # two linear projections to d_ff
        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)
        # output projection back to d_model
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)
        # SiLU activation
        self.activation = SiLU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        # branch1: activated
        x1 = self.activation(self.w1(x))
        # branch2: gating linear
        x2 = self.w3(x)
        # gated elementwise
        return self.w2(x1 * x2)
