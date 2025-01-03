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

import copy
import logging
import random
import time
import torch

from config_parser.config_parser import dict_from_conf
from pytorch_image_classification.federated_learning.FederatedServer import FederatedServer, compute_accuracy, \
    print_confusion_matrix, calculate_best_threshold_global
from pytorch_image_classification.federated_learning.scaffold.clientSCAFFOLD import clientSCAFFOLD
from utility import utility, plot_loss_round, plot_score_round


class SCAFFOLD(FederatedServer):
    """
        class implemented FederatedServer for simulating SCAFFOLD aggregation strategy
    """
    def __init__(self, identifier=None, iteration=None):
        """
            constructor cor scaffold server
        :param identifier:
        :param iteration:
        """
        super().__init__(identifier, iteration)
        self.selected = None
        self.selected_clients = None
        self.uploaded_weights = None
        self.uploaded_ids = None
        ut = utility()
        self.global_model = ut.get_Model()
        # select slow, the rate of slow clients is set to zero
        self.set_slow_clients()
        self.num_join_clients = 0
        # initialise the clients for training
        self.set_clients(clientSCAFFOLD)
        print("Finished creating server and clients.")

        self.Budget = []
        # set global model learning rate
        self.server_learning_rate = self.cfg.federated_learning.global_lr
        self.global_c = []

        # initialising global control variate
        for param in self.global_model.parameters():
            self.global_c.append(torch.zeros_like(param))

        self.args.learning_method = 'Scaffold'
        self.logger = logging.getLogger(__name__)
        self.start_aggregation(self.args, self.get_netdata_idx_map())

    def start_aggregation(self, args, net_dataidx_map):
        test_metrics = None
        # iterating over communication rounds
        for i in range(self.cfg.federated_learning.communication_rounds):
            s_t = time.time()

            # selecting random clients
            self.selected_clients, self.selected = self.select_clients()
            self.num_join_clients = len(self.selected_clients)

            self.send_models()
            n_epoch = self.cfg.federated_learning.local_epochs
            dict_train_metrics = {}
            # iterating over each clients
            for client_id, client in enumerate(self.selected_clients):
                if client_id not in self.selected:
                    continue
                client.logging_path = self.log.logging_path
                # perform local training
                min_metrics, val_metrics = client.train_local(client_id, client,
                                                              self.global_model,
                                                              self.train_parts[client_id], self.test_parts[client_id],
                                                              n_epoch,
                                                              self.cfg.federated_learning.learning_rate,
                                                              self.cfg.optimizer, None,
                                                              None,
                                                              self.global_c, device=self.cfg.device,
                                                              val_loader=self.val_parts[client_id], internal_round=i,
                                                              logger=self.logger)
                self.metrics.append(min_metrics)
                dict_train_metrics[client_id] = val_metrics
            self.train_metrics_dict_list.append(dict_train_metrics)
            # finding the multiplier by calculating the length of local data divided by total data size
            self.receive_models()
            # updating global model and control variates
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if len(self.metrics) > 0:
                # calculating optimum threshold from thresholds of different clients
                best_threshold = calculate_best_threshold_global(self.metrics)
                # calculating different performance metrics of each client using the said confidence value
                train_F = compute_accuracy(self.global_model, self.global_train_loader, device=self.cfg.device,
                                           best_threshold=best_threshold)
                test_metrics = compute_accuracy(self.global_model, self.global_test_loader, device=self.cfg.device,
                                                calc_all=True, best_threshold=best_threshold)
            else:
                train_F = compute_accuracy(self.global_model, self.global_train_loader, device=self.cfg.device)
                test_metrics = compute_accuracy(self.global_model, self.global_test_loader, device=self.cfg.device,
                                                calc_all=True)
            # saving the global metrixs
            self.global_metrics.append(test_metrics)
            print_confusion_matrix(self.global_model, self.global_test_loader,filepath=self.log.logging_path)
            print('>> Global Model Train F1Score:', train_F)
            print('>> Global Model Test F1Score: ', test_metrics)
            self.logger.info(f'>> Global Model Train F1Score: {train_F}')
            self.logger.info(f'>> Global Model Test Results:{test_metrics}')
        plot_loss_round(self.train_metrics_dict_list, message='SCAFFOLD', filepath=self.log.logging_path)
        plot_score_round([entry['FScore'] for entry in self.global_metrics], message='SCAFFOLD',
                         filepath=self.log.logging_path)
        if self.cfg.logging:
            self.logging(self.global_model, test_metrics)

    def send_models(self):
        """ copying global model parameter to client parameters """
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model, self.global_c)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        """
            calculating updated weights value for each client based on the dataset size of each client
        """
        assert (len(self.selected_clients) > 0)

        active_clients = self.selected_clients

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0

            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            # adding client local data set size to uploaded_weights
            self.uploaded_weights.append(client.train_samples)

        # calculating new uploaded weights for each client based on ratio of uploaded weight and tatal dataset size
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        """
            calculate new values of global model based on variables shared from clients
        """
        global_model = copy.deepcopy(self.global_model)
        global_c = copy.deepcopy(self.global_c)
        for cid in self.uploaded_ids:
            dy, dc = self.clients[cid].delta_yc()
            for server_param, client_param in zip(global_model.parameters(), dy):
                server_param.data += client_param.data.clone() / self.num_join_clients * self.server_learning_rate
            for server_param, client_param in zip(global_c, dc):
                server_param.data += client_param.data.clone() / self.cfg.no_of_local_machines
        self.global_model = global_model
        self.global_c = global_c

    def get_init_nets(self, args):
        """
            initialises global model and local model for server
        :param args: class containing configuration data for federated learning
        :return: global model and list of local model
        """
        nets, local_model_meta_data, layer_type = self.init_nets()
        # initialise global model
        global_models, global_model_meta_data, global_layer_type = self.init_nets("global")
        global_model = global_models[0]
        c_global = None
        c_nets = []
        # initialise control variates
        if self.args.learning_method == 'Scaffold':
            c_nets, _, _ = self.init_nets()
            c_globals, _, _ = self.init_nets("global")
            c_global = c_globals[0]
            c_global_para = c_global.state_dict()
            for net_id, net in c_nets.items():
                net.load_state_dict(c_global_para)
        global_para = global_model.state_dict()
        # copy global model weights
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)
        if self.args.learning_method == 'Scaffold':
            return c_global, c_nets, global_model, nets
        else:
            return global_model, nets

    def logging(self, global_model, train_acc):
        """
            function for logging results
        :param global_model: final global model
        :param metrics: result metrics after testing
        """
        self.log.add_to_log(None, dict_from_conf(self.cfg))
        self.log.add_to_log(None, train_acc)
        # self.log.add_to_log(None, {key: values.tolist() for key, values in global_model.state_dict().items()})
        self.log.save_log()


#S = SCAFFOLD()
