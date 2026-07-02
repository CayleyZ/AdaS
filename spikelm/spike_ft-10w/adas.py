import math
import torch
from torch.optim.optimizer import Optimizer
import numpy as np

class AdaS(Optimizer):
    """
    混合 SGD 和 AdamW 下降方向的优化器
    
    Args:
        params: 待优化参数
        lr: 学习率
        betas: Adam 的 beta 参数
        eps: 数值稳定性参数
        weight_decay: 权重衰减系数
        correct_bias: 是否修正偏差
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        gamma=2.0,
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
        """执行单步优化"""
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

                # 状态初始化
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # 计算偏差修正系数
                if group["correct_bias"]:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                else:
                    bias_correction1 = bias_correction2 = 1

                # 在更新前执行权重衰减
                if group["weight_decay"] != 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # 计算 AdamW 方向
                # 更新动量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 计算 AdamW 下降方向
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1

                # 计算最终的更新步长：alpha * adam_方向 + (1-alpha) * sgd_方向
                # SGD 方向就是原始梯度
                
                # 计算alpha
                gamma = group["gamma"]
                alpha = (torch.norm(exp_avg)**2 + 2 * gamma / step_size) ** 0.5 / torch.norm(exp_avg/denom - exp_avg)
                alpha = torch.clip(alpha, min=0.0, max=1.0)
                
                p.data.addcdiv_(
                    exp_avg, 
                    denom, 
                    value=-step_size * alpha
                ).add_(
                    grad, 
                    alpha=-step_size * (1 - alpha)
                )

        return loss