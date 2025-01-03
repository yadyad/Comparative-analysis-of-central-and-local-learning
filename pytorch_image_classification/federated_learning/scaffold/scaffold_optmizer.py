from torch.optim import Optimizer


class SCAFFOLDOptimizer(Optimizer):
    """
        custom optimizer implementing optimizer class from pytorch
    """
    def __init__(self, params, lr):
        # setting learning rate
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    def step(self, server_cs=None, client_cs=None):
        # overriding step function of optimizer for calculating update step by scaffold
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], server_cs, client_cs):
                p.data.add_(other=(p.grad.data + sc - cc), alpha=-group['lr'])
