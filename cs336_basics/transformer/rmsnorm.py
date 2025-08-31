import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Normalizes each vector over the last dimension:
      x_norm = x / RMS(x) * g
    where RMS(x) = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
    and g is a learnable gain parameter of shape (d_model,).
    """
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        # gain parameter g, shape (d_model,)
        self.g = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (..., d_model)
        returns: tensor of same shape and dtype
        """
        in_dtype = x.dtype
        # upcast to float32 for stability
        x_fp32 = x.to(torch.float32)
        rms = torch.sqrt(
            torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True) + self.eps
        )
        y = x_fp32.div(rms) * self.g
        # cast back to original dtype
        return y.to(in_dtype)