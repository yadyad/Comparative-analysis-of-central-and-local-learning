# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch
import numpy as np
import time
from pytorch_image_classification.federated_learning.FederatedClient import FederatedClient
from pytorch_image_classification.federated_learning.FederatedServer import compute_accuracy, print_confusion_matrix
from pytorch_image_classification.federated_learning.scaffold.scaffold_optmizer import SCAFFOLDOptimizer
from utility import utility, plot_train_val_loss


class clientSCAFFOLD(FederatedClient):
    """
        class containing function for simulating scaffold client
    """
    def __init__(self, i, model):
        """
            constructor for scaffold client
        :param i: id of client
        :param model: model of client
        """
        super().__init__()
        self.num_batches = None
        self.model = model
        self.id = i
        self.train_samples = 0
        # initialising custom optimizer for scaffold
        self.optimizer = SCAFFOLDOptimizer(self.model.parameters(), lr=self.cfg.federated_learning.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=self.cfg.federated_learning.learning_rate_decay_gamma
        )
        self.learning_rate = self.cfg.federated_learning.learning_rate
        self.client_c = []
        # initialising the control variate for client
        for param in self.model.parameters():
            self.client_c.append(torch.zeros_like(param))
        self.global_c = None
        self.global_model = None
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0}

    def train_local(self, net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu,
                    c_local, c_global, device="cpu", internal_round=0, data_set_len=None, val_loader=None, logger=None):
        """
            simulates local training on client side of scaffold federated learning
        :param net_id: identifier of client
        :param net: model of this client
        :param global_net: global model shared by server
        :param train_dataloader: train dataloader
        :param test_dataloader: test dataloader
        :param epochs: total number of epochs
        :param lr: learning rate
        :param args_optimizer: not required here
        :param mu: not required here
        :param c_local: client local variate
        :param c_global: global local variate
        :param device: cpu or gpu
        :param internal_round: communication round
        :param data_set_len: length of local dataset
        :param val_loader: validation data loader
        :param logger: logger used for saving logs
        :return: metrics calculated after training the model
        """
        trainloader = train_dataloader
        min_metrics = 100
        val_metrics = None
        self.model.train()
        ut = utility()
        loss_function = ut.get_Criterion()
        start_time = time.time()

        max_local_epochs = epochs
        # not used
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        epoch_loss_collector = []
        val_losses = []
        # iterating through epochs
        for epoch in range(max_local_epochs):
            # iterating through batches
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                else:
                    x = x.to(device)
                y = y.to(device)

                # not used
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                if torch.cuda.is_available() and self.cfg.device == 'cuda':
                    self.model = self.model.cuda()

                # training the model
                output = self.model(x)
                loss = loss_function(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(self.global_c, self.client_c)

            epoch_loss_collector.append(loss.item())

            # calculating the optimum confidence and performance metric of validation
            val_metrics = compute_accuracy(self.model, val_loader, device=device, calc_all=True,
                                           with_threshold_opt=True)
            val_losses.append(val_metrics['loss'])
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            if epoch == 0:
                min_metrics = val_metrics
            if ('best_metrics' in val_metrics.keys() and min_metrics is not None and
                    min_metrics['best_metrics'] > val_metrics['best_metrics']):
                min_metrics = val_metrics

            # printing performance metrics for validation
            output_string = f'Machine {net_id} epoch {epoch} loss: {epoch_loss} val_F_score: {val_metrics["FScore"]}'
            if self.cfg.classification_type == "binary":
                output_string += f'val_FPR: {val_metrics["fpr"]} recall: {val_metrics["recall"]}'
            print(output_string)
            logger.info(output_string)

        # plot train vs validation loss
        plot_train_val_loss(epoch_loss_collector, val_losses, filepath=self.logging_path,
                            comm_round=internal_round, client=net_id)
        print_confusion_matrix(self.model, val_loader, machine=net_id, comm_round=internal_round,filepath=self.logging_path)
        self.num_batches = len(trainloader)
        self.train_samples = len(trainloader.dataset)

        # calculating control variate values that need to be sent to server
        self.update_yc(max_local_epochs)

        #not used
        if self.cfg.federated_learning.learning_rate_decay_gamma:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        return min_metrics, val_metrics

    def set_parameters(self, model, global_c):
        """
            update parameter of model from global model
        :param model: old model to be updated
        :param global_c: global model from where values are copied
        """
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_c = global_c
        self.global_model = model

    def update_yc(self, max_local_epochs=None):
        """
            function to calculate delta c_i from scaffold algorithm
        :param max_local_epochs: total number of epochs
        """
        if max_local_epochs is None:
            max_local_epochs = self.cfg.federated_learning.local_epochs
        for ci, c, x, yi in zip(self.client_c, self.global_c, self.global_model.parameters(), self.model.parameters()):
            ci.data = ci - c + 1 / self.num_batches / max_local_epochs / self.learning_rate * (x - yi)

    def delta_yc(self, max_local_epochs=None):
        """
            function to calcualte delta x_i from scaffold algorithm, this values are used to update global model
            and control variate
        :param max_local_epochs: total number of epochs
        :return: delta_y, delta_c
        """
        if max_local_epochs is None:
            max_local_epochs = self.cfg.federated_learning.local_epochs
        delta_y = []
        delta_c = []
        for c, x, yi in zip(self.global_c, self.global_model.parameters(), self.model.parameters()):
            delta_y.append(yi - x)
            delta_c.append(- c + 1 / self.num_batches / max_local_epochs / self.learning_rate * (x - yi))

        return delta_y, delta_c
