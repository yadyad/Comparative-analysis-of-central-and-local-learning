import logging

import numpy as np
import torch

from pytorch_image_classification.federated_learning.FederatedServer import FederatedServer, print_confusion_matrix, \
    compute_accuracy, calculate_best_threshold_global
from pytorch_image_classification.federated_learning.FederatedAveragingClient import FederatedAveragingClient
import utility as ut
from config_parser.config_parser import dict_from_conf
from pytorch_image_classification.argument import argument


class FederatedAveragingServer(FederatedServer):
    def __init__(self,identifier=None, iteration=None):
        super().__init__(identifier, iteration)
        self.args = argument()
        self.logger = logging.getLogger(__name__)
        self.start_aggregation(self.args, self.get_netdata_idx_map())

    def get_init_nets(self, args):
        nets, local_model_meta_data, layer_type = self.init_nets()
        # a single net is initiated and added to dictionary
        global_models, global_model_meta_data, global_layer_type = self.init_nets('global')
        global_model = global_models[0]
        global_para = global_model.state_dict()
        # args.is_same_initial is true then all the nets are initialised with the same weights and bias
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)
        return global_model, nets

    def logging(self, global_model, metrics):
        self.log.add_to_log(None, dict_from_conf(self.cfg))
        self.log.add_to_log(None, metrics)
        # self.log.add_to_log(None, {key: values.tolist() for key, values in global_model.state_dict().items()})
        self.log.save_log()

    def update_global_weights(self, global_model, global_para, net_dataidx_map, nets, selected):
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]
        # going through each selected clients
        for idx in range(len(selected)):
            net_para = nets[selected[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    global_para[key] = net_para[key] * fed_avg_freqs[idx]
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * fed_avg_freqs[idx]
        global_model.load_state_dict(global_para)
        return global_model

    def initialize_nets(self, args, global_para, nets, round, selected):
        if round == 0:
            if args.is_same_initial:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)
        else:
            for idx in selected:
                nets[idx].load_state_dict(global_para)
        return nets

    def start_aggregation(self, args, net_dataidx_map):
        """
                starting point of federated learning algorithm
                :param args:
                :param net_dataidx_map:
                :return:
                """
        test_metrics = None
        print("Initializing nets")
        # multiple nets are created and added to dictionary consist of no_local_machine number of nets
        global_model, nets = self.get_init_nets(args)
        test_F = None

        # for each communication roung specified
        for comm_round in range(self.cfg.federated_learning.communication_rounds):
            print("in comm round:" + str(comm_round))
            print("=============================================================================================")
            # selects random number of clients for participating in each round.
            # here the args.sample is the ratio of participant in each round compared to total participants
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            # appending global weights and bias to selected participant from the clients
            # in round zero is_same_initial checked otherwise every time same parameters as global model is appended.
            global_para = global_model.state_dict()
            nets = self.initialize_nets(args, global_para, nets, comm_round, selected)

            # local nets are trained here and tested based on global_test_loader which is the concatenated test loader
            self.local_train_net(nets, selected, args, net_dataidx_map, test_dl=self.global_test_loader,
                                 device=self.cfg.device, comm_round=comm_round)

            # update global model
            # federated averaging done here
            global_model = self.update_global_weights(global_model, global_para, net_dataidx_map, nets, selected)
            global_model.to(self.cfg.device)

            if len(self.metrics) > 0 and self.cfg.classification_type == ('binary'
                                                                          ''):
                best_threshold = calculate_best_threshold_global(self.metrics)
                train_F = compute_accuracy(global_model, self.global_train_loader, device=self.cfg.device,
                                           best_threshold=best_threshold)
                test_metrics = compute_accuracy(global_model, self.global_test_loader, device=self.cfg.device,
                                                calc_all=True, best_threshold=best_threshold)
            else:
                train_F = compute_accuracy(global_model, self.global_train_loader, device=self.cfg.device)
                test_metrics = compute_accuracy(global_model, self.global_test_loader, device=self.cfg.device,
                                                calc_all=True)
            self.global_metrics.append(test_metrics)
            print_confusion_matrix(global_model, self.global_test_loader, comm_round=comm_round,
                                   filepath=self.log.logging_path)
            print('>> Global Model Train F1Score: ', train_F)
            print('>> Global Model Test Results: ', test_metrics)
            self.logger.info(f'>> Global Model Train F1Score: {train_F}')
            self.logger.info(f'>> Global Model Test Results:{test_metrics}')
        ut.plot_loss_round(self.train_metrics_dict_list, message='Federated Averaging', filepath=self.log.logging_path)
        ut.plot_score_round([entry['FScore'] for entry in self.global_metrics], message='Federated Averaging',
                            filepath=self.log.logging_path)
        if self.cfg.logging:
            self.logging(global_model, test_metrics)

    def local_train_net(self, nets, selected, args, net_dataidx_map, test_dl=None, device="cpu", comm_round=0):
        avg_acc = 0.0

        test_acc_list = []
        test_size_list = []
        clients = []
        dict_train_metrics = {}
        # Iterating through each nets
        for net_id, net in nets.items():
            if net_id not in selected:
                continue
            # fetching the data loader for the id
            dataidxs = net_dataidx_map[net_id]
            val_df = self.val_parts[net_id]
            client = FederatedAveragingClient()
            client.logging_path = self.log.logging_path
            print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs.dataset)))
            # move the model to cuda device:
            if torch.cuda.is_available() and self.cfg.device == 'cuda':
                net = net.cuda()

            # setting up train loader and testloader
            train_dl_local, test_dl_local = self.train_parts[net_id], self.test_parts[net_id]
            n_epoch = self.cfg.federated_learning.local_epochs
            # Train and test the net on data for each client
            trainacc, test_F, min_metrics, val_metrics = client.train_local(net_id, net, None, train_dl_local,
                                                                            test_dl_local,
                                                                            n_epoch,
                                                                            self.cfg.federated_learning.learning_rate,
                                                                            self.cfg.optimizer, None,None, None,
                                                                            device=device,
                                                                            val_loader=val_df,
                                                                            logger=self.logger,
                                                                            comm_round=comm_round)
            print("net %d final test acc %f" % (net_id, test_F))
            avg_acc += test_F
            test_acc_list.append(test_F)
            test_size_list.append(len(test_dl_local.dataset))
            clients.append(client)
            if min_metrics is not None:
                self.metrics.append(min_metrics)
                dict_train_metrics[net_id] = val_metrics
        self.train_metrics_dict_list.append(dict_train_metrics)
        weighted_average = ut.weighted_average_test_score(test_acc_list, test_size_list)
        avg_acc /= len(selected)
        print("avg test F_score %f" % avg_acc)
        print(f"average weighted F_score {weighted_average}")

        nets_list = list(nets.values())
        return nets_list


#F = FederatedAveragingServer()
