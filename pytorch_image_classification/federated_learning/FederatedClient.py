from abc import ABC, abstractmethod

from configuration import Configuration
from utility import utility


class FederatedClient(ABC):
    def __init__(self):
        super().__init__()
        self.cfg = Configuration()
        self.train_slow = False
        self.ut = utility()
        self.logging_path = None

    @abstractmethod
    def train_local(self, net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu,
                    c_local, c_global, device="cpu", comm_round=0, data_set_len=None, val_loader=None, logger=None):
        if data_set_len is None:
            data_set_len = [0]
