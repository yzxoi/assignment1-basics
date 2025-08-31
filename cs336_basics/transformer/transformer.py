import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import Linear
from .attention import CausalMultiHeadSelfAttention, softmax
from .feed_forward import FeedForward
from .rmsnorm import RMSNorm
from .embedding import Embedding

class TransformerBlock(nn.Module):
    """
    预归一化 Transformer Block:
      y = x + MHA(RMSNorm(x))
      z = y + FFN(RMSNorm(y))
    注意：MHA 内部已做因果遮罩；对 Q/K 应用 RoPE（取决于传入 token_positions 与 rope 的设置）
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        fw = {"device": device, "dtype": dtype}
        self.rms1 = RMSNorm(d_model, **fw)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )
        self.rms2 = RMSNorm(d_model, **fw)
        self.ffn  = FeedForward(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # sublayer 1: pre-norm + MHA + residual
        y = x + self.attn(self.rms1(x), token_positions=token_positions)
        # sublayer 2: pre-norm + FFN + residual
        z = y + self.ffn(self.rms2(y))
        return z
    
class TransformerLM(nn.Module):
    """
    语言模型：
      x = token_embedding(tokens)  # (B,L,d_model)
      x = Blocks(x, RoPE via attn, causal mask)
      x = RMSNorm(x)
      logits = x @ W_out^T
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        fw = {"device": device, "dtype": dtype}
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embed = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                theta=theta, max_seq_len=context_length,
                device=device, dtype=dtype,
            ) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model, **fw)
        self.lm_head = Linear(d_model, vocab_size, **fw)  # 输出投影到词表

    def forward(self, tokens: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        tokens: (B, L) int64
        token_positions: (B, L) 或 (L,)；若 None，则用 0..L-1
        """
        B, L = tokens.shape
        x = self.embed(tokens)  # (B,L,d_model)

        if token_positions is None:
            pos = torch.arange(L, device=tokens.device).expand(B, L)
        else:
            pos = token_positions if token_positions.dim() == 2 else token_positions.unsqueeze(0).expand(B, L)

        for blk in self.blocks:
            x = blk(x, token_positions=pos)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B,L,vocab)
        return logits