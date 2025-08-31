import torch
import torch.nn as nn
import math

from .linear import Linear
from .embedding import RotaryPositionalEmbedding
from .utils import softmax
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    query: Tensor of shape (..., seq_len_q, d_k)
    key:   Tensor of shape (..., seq_len_k, d_k)
    value: Tensor of shape (..., seq_len_k, d_v)
    mask:  Optional boolean Tensor of shape (seq_len_q, seq_len_k),
           True indicates positions to attend to.

    Returns:
        Tensor of shape (..., seq_len_q, d_v)
    """
    # (..., seq_len_q, d_k) @ (..., d_k, seq_len_k) -> (..., seq_len_q, seq_len_k)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)

    # apply mask if provided
    if mask is not None:
        # mask shape (seq_len_q, seq_len_k) -> broadcast over batch dims
        # True for allowed, False for masked
        scores = scores.masked_fill(~mask, float('-inf'))

    # attention weights
    attn = softmax(scores, dim=-1)
    # (..., seq_len_q, seq_len_k) @ (..., seq_len_k, d_v) -> (..., seq_len_q, d_v)
    return torch.matmul(attn, value)

class CausalMultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention with Rotary Positional Embeddings.
    Prevents attending to future positions via causal mask.
    Applies RoPE to Q and K for each head.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        factory_kwargs = {"device": device, "dtype": dtype}

        # Query, Key, Value, and Output projections
        self.Wq = Linear(d_model, d_model, **factory_kwargs)
        self.Wk = Linear(d_model, d_model, **factory_kwargs)
        self.Wv = Linear(d_model, d_model, **factory_kwargs)
        self.Wo = Linear(d_model, d_model, **factory_kwargs)

        # Rotary positional embeddings for Q and K
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: Tensor of shape (batch_size, seq_len, d_model)
        returns: Tensor of shape (batch_size, seq_len, d_model)
        """
        b, seq_len, _ = x.size()
        device = x.device

        # Linear projections
        q = self.Wq(x)  # (b, seq_len, d_model)
        k = self.Wk(x)
        v = self.Wv(x)

        # Reshape for heads: (b, num_heads, seq_len, d_k)
        q = q.view(b, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(b, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(b, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        # Apply RoPE to Q and K
        if token_positions is None:
            # (L,) → broadcast 到 (B,L)
            positions = torch.arange(seq_len, device=device, dtype=torch.long)
            # Expand to match the batch size of x. Shape: (b, seq_len)
            positions = positions.unsqueeze(0).expand(b, seq_len)
        else:
            # 允许 (L,) 或 (B,L)
            positions = token_positions
            # If provided positions have a batch dim of 1, expand to match input batch size.
            if positions.shape[0] == 1 and b > 1:
                positions = positions.expand(b, -1)

        q = self.rope(q, positions.unsqueeze(1))
        k = self.rope(k, positions.unsqueeze(1))

        # Causal mask (seq_len, seq_len)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

        # Scaled dot-product attention with mask
        attn = scaled_dot_product_attention(q, k, v, mask)

        # Concatenate heads: (b, seq_len, d_model)
        attn = attn.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)

        # Final linear projection
        return self.Wo(attn)
