import torch
import math
from typing import Tuple, Union, BinaryIO, IO
from torch import Tensor
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim

FileLike = Union[str, os.PathLike, BinaryIO, IO[bytes]]


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Numerically stable softmax over dimension `dim`.
    """
    # subtract max for numerical stability
    x_max = x.max(dim=dim, keepdim=True).values
    exps = torch.exp(x - x_max)
    sum_exps = exps.sum(dim=dim, keepdim=True)
    return exps / sum_exps

def SiLU(in_features: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation function: SiLU(x) = x * sigmoid(x)
    """
    return in_features * torch.sigmoid(in_features)

def get_lr_scheduler_cosine_with_warmup(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    """
    Cosine learning rate scheduler with warmup.
    """
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    elif it < cosine_cycle_iters:
        return min_learning_rate + (max_learning_rate - min_learning_rate) * (
            0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
        )
    else:
        return min_learning_rate

def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str                   # e.g. 'cpu' | 'cuda:0' | 'mps'
) -> Tuple[Tensor, Tensor]:
    """
      inputs[b]  = x[i : i+context_length]
      targets[b] = x[i+1 : i+context_length+1]
    return:
      inputs, targets -> (batch_size, context_length) LongTensor, on device
    """
    n = x.shape[0]
    assert n >= context_length + 1, "Sequence too short for given context_length."

    starts = np.random.randint(0, n - context_length, size=batch_size)

    inp_np = np.stack([x[i : i + context_length] for i in starts], axis=0)
    tgt_np = np.stack([x[i + 1 : i + 1 + context_length] for i in starts], axis=0)

    inputs  = torch.tensor(inp_np, dtype=torch.long, device=device)
    targets = torch.tensor(tgt_np, dtype=torch.long, device=device)
    return inputs, targets


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: FileLike,
) -> None:
    """
    Save a checkpoint to the given output file.
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: FileLike,
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> int:
    """
    Load a checkpoint from a file.
    """
    # load the checkpoint on cpu first
    checkpoint = torch.load(src, map_location="cpu")

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    return int(checkpoint.get("iteration", 0))