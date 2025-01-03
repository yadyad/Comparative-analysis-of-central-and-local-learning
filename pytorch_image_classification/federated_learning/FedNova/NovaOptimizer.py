from typing import Dict

from flwr.common import NDArrays
from torch.optim.optimizer import Optimizer, required

import torch


class ProxSGD(Optimizer):  # pylint: disable=too-many-instance-attributes
    """Optimizer class for FedNova that supports Proximal, SGD, and Momentum updates.

    SGD optimizer modified with support for :
    1. Maintaining a Global momentum buffer, set using : (self.gmf)
    2. Proximal SGD updates, set using : (self.mu)
    Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                    parameter groups
            ratio (float): relative sample size of client
            gmf (float): global/server/slow momentum factor
            mu (float): parameter for proximal local SGD
            lr (float): learning rate
            momentum (float, optional): momentum factor (default: 0)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            dampening (float, optional): dampening for momentum (default: 0)
            nesterov (bool, optional): enables Nesterov momentum (default: False)
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            params,
            ratio: float,
            gmf=0,
            mu=0,
            lr=required,
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False,
            variance=0,
    ):
        """ constructor class for ProxSGD"""
        self.gmf = gmf
        self.ratio = ratio
        self.momentum = momentum
        self.mu = mu
        self.local_normalizing_vec = 0
        self.local_counter = 0
        self.local_steps = 0
        self.lr = lr

        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "variance": variance,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        """Set the optimizer state."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):  # pylint: disable=too-many-branches
        """Perform a single optimization step."""
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                param_state = self.state[p]

                local_lr = group["lr"]

                # apply momentum updates
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # apply proximal updates
                if self.mu != 0:
                    if param_state["old_init"].device != p.device:
                        param_state["old_init"] = param_state["old_init"].to(p.device)
                    d_p.add_(p.data - param_state["old_init"], alpha=self.mu)

                # update accumalated local updates
                if "cum_grad" not in param_state:
                    param_state["cum_grad"] = torch.clone(d_p).detach()
                    param_state["cum_grad"].mul_(local_lr)

                else:
                    param_state["cum_grad"].add_(d_p, alpha=local_lr)

                p.data.add_(d_p, alpha=-local_lr)

        # compute local normalizing vector a_i
        if self.momentum != 0:
            self.local_counter = self.local_counter * self.momentum + 1
            self.local_normalizing_vec += self.local_counter

        etamu = local_lr * self.mu
        if etamu != 0:
            self.local_normalizing_vec *= 1 - etamu
            self.local_normalizing_vec += 1

        if self.momentum == 0 and etamu == 0:
            self.local_normalizing_vec += 1

        self.local_steps += 1

    def get_gradient_scaling(self) -> Dict[str, float]:
        """Compute the scaling factor for local client gradients.

        Returns: A dictionary containing weight, tau, and local_norm.
        """
        if self.mu != 0:
            local_tau = torch.tensor(self.local_steps * self.ratio)
        else:
            local_tau = torch.tensor(self.local_normalizing_vec * self.ratio)
        local_stats = {
            "weight": self.ratio,
            "tau": local_tau.item(),
            "local_norm": self.local_normalizing_vec,
        }

        return local_stats

    def set_model_params(self, init_params: NDArrays):
        """Set the model parameters to the given values."""
        i = 0
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_tensor = torch.tensor(init_params[i])
                p.data.copy_(param_tensor)
                param_state["old_init"] = param_tensor
                i += 1

    def set_lr(self, lr: float):
        """Set the learning rate to the given value."""
        for param_group in self.param_groups:
            param_group["lr"] = lr
