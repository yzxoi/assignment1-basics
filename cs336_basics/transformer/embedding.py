import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    """
    A bias-less token embedding layer: maps token IDs to vectors.
    Embedding matrix shape: (num_embeddings, embedding_dim).
    Initialized with N(0,1) truncated at [-3,3].
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # weight matrix of shape (vocab_size, d_model)
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        # truncated normal initialization: mean=0, std=1, bounds [-3,3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: LongTensor of shape (batch_size, sequence_length)
        returns: FloatTensor of shape (batch_size, sequence_length, embedding_dim)
        """
        return self.weight[token_ids]
    
class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    Applies rotary embedding to the last dimension of x.
    No learnable parameters; precomputes cos/sin buffers.
    """
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()
        half_dim = d_k // 2
        # inverse frequency for each pair
        inv_freq = theta ** (-2 * torch.arange(half_dim, device=device, dtype=torch.float32) / d_k)
        # positions and angles
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        angles = positions * inv_freq.unsqueeze(0)  # (max_seq_len, half_dim)
        # expand to interleaved dims
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # (max_seq_len, d_k)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)
        # non-persistent buffers
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        cos = self.cos[token_positions]  # same shape as x
        sin = self.sin[token_positions]
        # split even and odd dims
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        cos1 = cos[..., ::2]
        sin1 = sin[..., ::2]
        # apply rotary: [x1*cos - x2*sin, x1*sin + x2*cos]
        rx1 = x1 * cos1 - x2 * sin1
        rx2 = x1 * sin1 + x2 * cos1
        # interleave back
        x_rotated = torch.stack((rx1, rx2), dim=-1)  # (..., seq_len, half_dim, 2)
        return x_rotated.view_as(x)