import math

import torch
from torch.optim.optimizer import Optimizer

from mmseg.registry import OPTIMIZERS


@OPTIMIZERS.register_module()
class AdaS(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        gamma=1.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        correct_bias=True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            gamma=gamma,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    def step(self, current_loss=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        elif current_loss is not None:
            loss = current_loss

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdaS does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                if group["correct_bias"]:
                    bias_correction1 = 1 - beta1**state["step"]
                    bias_correction2 = 1 - beta2**state["step"]
                else:
                    bias_correction1 = bias_correction2 = 1

                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"])
                step_size = group["lr"] / bias_correction1
                gamma = group["gamma"]

                if step_size == 0:
                    alpha = 1.0
                else:
                    alpha = (torch.norm(exp_avg)**2 + 2 * gamma /
                             step_size)**0.5 / torch.norm(exp_avg / denom -
                                                          exp_avg)
                    alpha = torch.clip(alpha, min=0.0, max=1.0)

                p.data.addcdiv_(exp_avg, denom, value=-step_size * alpha).add_(
                    exp_avg, alpha=-step_size * (1 - alpha))

        return loss


class LinearGammaScheduler:
    def __init__(self, optimizer, start_gamma=1.0, end_gamma=0.0, total_epochs=100):
        self.optimizer = optimizer
        self.start_gamma = start_gamma
        self.end_gamma = end_gamma
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self._update_gamma(start_gamma)

    def _update_gamma(self, gamma):
        for param_group in self.optimizer.param_groups:
            param_group["gamma"] = gamma

    def step(self):
        self.current_epoch += 1
        if self.total_epochs <= 1:
            new_gamma = self.end_gamma
        else:
            progress = min(self.current_epoch / (self.total_epochs - 1), 1.0)
            new_gamma = self.start_gamma + (
                self.end_gamma - self.start_gamma) * progress

        self._update_gamma(new_gamma)
