from configuration import Configuration


class argument:

    def __init__(self):
        self.cfg = Configuration()
        self.is_same_initial = self.cfg.federated_learning.is_same_initial
        self.comm_round = self.cfg.federated_learning.communication_rounds  #number of communication rounds
        self.n_parties = self.cfg.no_of_local_machines
        self.sample = self.cfg.federated_learning.sample  # ratio of parties that participates in each round
        # self.noise = 0 #maximum value of gaussian noise we add to local party
        self.rho = self.cfg.rho
        self.mu = self.cfg.mu
        self.weight_decay = self.cfg.weight_decay
        self.var_max_epochs = self.cfg.federated_learning.var_max_epochs
        self.var_min_epochs = self.cfg.federated_learning.var_min_epochs
        self.seed = self.cfg.federated_learning.seed
        self.var_local_epochs = self.cfg.federated_learning.var_local_epochs
        self.gmf = self.cfg.federated_learning.gmf
        self.lr = self.cfg.lr
        self.learning_method = None
