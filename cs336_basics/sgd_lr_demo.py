import torch
import math
import matplotlib.pyplot as plt

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


def run_experiment(lr, steps=10):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    losses = []
    for t in range(steps):
        opt.zero_grad()
        loss = (weights ** 2).mean()
        losses.append(loss.item())
        loss.backward()
        opt.step()
    return losses


if __name__ == "__main__":
    learning_rates = [1.0, 1e1, 1e2, 1e3]
    steps = 10
    results = {lr: run_experiment(lr, steps) for lr in learning_rates}

    plt.figure(figsize=(8, 5))
    for lr, losses in results.items():
        plt.semilogy(range(steps), losses, marker="o", label=f"lr={lr}")
    plt.xlabel("Training step")
    plt.ylabel("Loss (log scale)")
    plt.title("SGD with sqrt decay (different learning rates)")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()