import math
import torch
from torch.optim.optimizer import Optimizer

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
        # if not 0.0 <= gamma <= 1.0:
        #     raise ValueError(f"Invalid gamma value: {gamma}")
            
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
                    exp_avg, 
                    alpha=-step_size * (1 - alpha)
                )

        return loss
    
    
class LinearGammaScheduler:
    """
    Linear scheduler for the gamma parameter in AdaS optimizer.
    Linearly decreases gamma from start_gamma to end_gamma over total_epochs.

    Args:
        optimizer: 目标优化器 (AdaS)
        start_gamma: 初始gamma值
        end_gamma: 最终gamma值
        total_epochs: 总训练轮数
    """
    def __init__(self, optimizer, start_gamma=1.0, end_gamma=0.0, total_epochs=100):
        self.optimizer = optimizer
        self.start_gamma = start_gamma
        self.end_gamma = end_gamma
        self.total_epochs = total_epochs
        self.current_epoch = 0  # 跟踪当前epoch数

        # 初始化gamma为起始值
        self._update_gamma(start_gamma)

    def _update_gamma(self, gamma):
        """更新所有参数组中的gamma值"""
        for param_group in self.optimizer.param_groups:
            param_group['gamma'] = gamma

    def step(self):
        """在每个epoch结束时调用，更新gamma值"""
        self.current_epoch += 1

        if self.total_epochs <= 1:
            # 如果总轮数小于等于1，直接使用最终值
            new_gamma = self.end_gamma
        else:
            # 计算线性插值比例 (限制在[0,1]范围内)
            progress = min(self.current_epoch / (self.total_epochs - 1), 1.0)
            new_gamma = self.start_gamma + (self.end_gamma - self.start_gamma) * progress

        self._update_gamma(new_gamma)