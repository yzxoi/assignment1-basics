import torch
from torch import Tensor

def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute cross entropy.
      logits: Shape (..., V), V = vocab size
      targets: Shape (...), int64 in [0, V)
    Returns:
        Scalar tensor (average over all batch positions)
    """
    assert logits.shape[:-1] == targets.shape, \
        f"targets shape {targets.shape} must match logits shape {logits.shape[:-1]}"

    max_logits = logits.max(dim=-1, keepdim=True).values                  # (..., 1)
    shifted = logits - max_logits                                         # (..., V)

    lse = torch.log(torch.exp(shifted).sum(dim=-1)) + max_logits.squeeze(-1)  # (...,)

    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)      # (...,)

    loss = (lse - target_logits).mean()
    return loss

def perplexity_from_logits(logits: Tensor, targets: Tensor) -> Tensor:
    loss = cross_entropy(logits, targets)
    return torch.exp(loss)