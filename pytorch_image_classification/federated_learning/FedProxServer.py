import logging

import numpy as np
import torch

import utility as ut
from pytorch_image_classification.federated_learning.FederatedServer import FederatedServer, compute_accuracy, \
    print_confusion_matrix, calculate_best_threshold_global
from config_parser.config_parser import dict_from_conf
from pytorch_image_classification.argument import argument
from pytorch_image_classification.federated_learning.FedProxClient import FedProxClient


class FedProxServer(FederatedServer):
    """
        class containing functions related to federated proximal gradient aggregation
    """

    def __init__(self, identifier=None, iteration=None):
        """
            constructor for FedProxServer
        :param identifier: unique identifier for the experiment run
        :param iteration: value of iteration for the experiment run
        """
        super().__init__(identifier, iteration)
        self.args = argument()
        self.logger = logging.getLogger(__name__)
        self.start_aggregation(self.args, self.get_netdata_idx_map())

    def get_init_nets(self, args):
        """
            initialise the different model for different clients.
            initialise the global model
        :param args: class containing federated learning configuration
        :return: global model and client models
        """
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

    def initialize_nets(self, args, global_para, nets, round, selected):
        """
            initialise the different model for different clients based on different configurations
            copies the global model parameters into local models
        :param args: configuration data stored in a class
        :param global_para: global model of federated averaging server
        :param nets: local model of federated averaging clients
        :param round: communication round
        :param selected: clients selected for the current round of training
        :return: initialised client models
        """
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
            starting point of FedProx learning algorithm, the function initialises clients and sends data to client
             and perform aggregation
        :param args: class containing configuration parameters
        :param net_dataidx_map: dictionary containing datapoint and key client_id
        """
        print("Initializing nets")
        global_model, nets = self.get_init_nets(args)
        test_metrics = None
        # iterator for each communication round
        for comm_round in range(args.comm_round):
            print("in comm round:" + str(comm_round))
            print("=============================================================================================")
            # selecting random clients for each round
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]
            # initialising client models with global model weights
            global_para = global_model.state_dict()
            nets = self.initialize_nets(args, global_para, nets, comm_round, selected)
            # sending model for local learning to clients
            self.local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map,
                                         test_dl=self.global_test_loader, device=self.cfg.device)
            global_model.to(self.cfg.device)

            # update global model
            self.update_global_weights(global_model, global_para, net_dataidx_map, nets, selected)

            global_model.to(self.cfg.device)
            if len(self.metrics) > 0 and self.cfg.classification_type == "binary":
                # calculating best confidence value from different threshold values shared by clients
                best_threshold = calculate_best_threshold_global(self.metrics)
                # calculating different metrics for global model based on train and test data loader
                train_FScore = compute_accuracy(global_model, self.global_train_loader, device=self.cfg.device,
                                                best_threshold=best_threshold)
                test_metrics = compute_accuracy(global_model, self.global_test_loader, device=self.cfg.device,
                                                calc_all=True, best_threshold=best_threshold)
            else:
                #calculating the performance metrics for global model of multiclass classification
                train_FScore = compute_accuracy(global_model, self.global_train_loader, device=self.cfg.device)
                test_metrics = compute_accuracy(global_model, self.global_test_loader, device=self.cfg.device,
                                                calc_all=True)

            self.global_metrics.append(test_metrics)
            # plotting confusion matrix
            print_confusion_matrix(global_model, self.global_test_loader, comm_round=comm_round,
                                   filepath=self.log.logging_path)
            print('>> Global Model Train accuracy: ', train_FScore)
            print('>> Global Model Test accuracy: ', test_metrics)
            self.logger.info(f'>> Global Model Train F1Score: {train_FScore}')
            self.logger.info(f'>> Global Model Test Results:{test_metrics}')
        # plotting loss vs rounds
        ut.plot_loss_round(self.train_metrics_dict_list, message='FedProx', filepath=self.log.logging_path)
        # plotting f1-score vs rounds
        ut.plot_score_round([entry['FScore'] for entry in self.global_metrics], message='FedProx',
                            filepath=self.log.logging_path)
        # logging results to log file
        if self.cfg.logging:
            self.logging(global_model, test_metrics)

    def logging(self, global_model, train_acc):
        """
            function for logging results
        :param global_model: final global model
        :param train_acc: result metrics after testing
        """
        self.log.add_to_log(None, dict_from_conf(self.cfg))
        self.log.add_to_log(None, train_acc)
        # self.log.add_to_log(None, {key: values.tolist() for key, values in global_model.state_dict().items()})
        self.log.save_log()

    def update_global_weights(self, global_model, global_para, net_dataidx_map, nets, selected):
        """
            function for updating global weights by using local model shared by clients
        :param global_model: global model of FedProx server
        :param global_para: dictionary containing weights parameter of global model
        :param net_dataidx_map: dictionary containing datapoint and key client_id
        :param nets: local model of FedProx clients
        :param selected: client selected for the current round of training
        """
        #calculating multiplier for weights
        total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]
        for idx in range(len(selected)):
            net_para = nets[selected[idx]].cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    #calculating updated weights and bias
                    global_para[key] = net_para[key] * fed_avg_freqs[idx]
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * fed_avg_freqs[idx]
        global_model.load_state_dict(global_para)

    def local_train_net_fedprox(self, nets, selected, global_model, args, net_dataidx_map, test_dl=None, device="cpu"):
        """
             simulate the training on one client after other
        :param nets: local model for each client
        :param selected:  selected for the current round of training
        :param global_model: global model of FedProx server
        :param args: class containing federated learning configuration
        :param net_dataidx_map: dictionary containing datapoint and key client_id
        :param test_dl: test data loader
        :param device: device where the learning should be performed cpu or gpu
        :return: list of client model
        """
        avg_FScore = 0.0

        test_acc_list = []
        test_size_list = []
        dict_train_metrics = {}
        # iterating through each client
        for net_id, net in nets.items():
            if net_id not in selected:
                continue
            dataidxs = net_dataidx_map[net_id]
            val_df = self.val_parts[net_id]
            # initialising client
            client = FedProxClient()
            client.logging_path = self.log.logging_path
            print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
            # move the model to cuda device:
            if torch.cuda.is_available() and self.cfg.device == "cuda":
                net = net.cuda()

            # fetching train and test dataloader
            train_dl_local, test_dl_local = self.train_parts[net_id], self.test_parts[net_id]
            # fetching epochs
            n_epoch = self.cfg.federated_learning.local_epochs
            # simulating training on local client
            trainF1, testF1, min_metrics, val_metrics = client.train_local(net_id, net, global_model, train_dl_local,
                                                                           test_dl,
                                                                           n_epoch,
                                                                           self.cfg.federated_learning.learning_rate,
                                                                           self.cfg.optimizer,
                                                                           args.mu, None, None, device,
                                                                           None, None, val_loader=val_df,
                                                                           logger=self.logger)
            print("net %d final test F1score %f" % (net_id, testF1))
            avg_FScore += testF1
            test_acc_list.append(testF1)
            test_size_list.append(len(test_dl_local.dataset))
            # saving best confidence value to a list
            if min_metrics is not None:
                self.metrics.append(min_metrics)
                dict_train_metrics[net_id] = val_metrics
        self.train_metrics_dict_list.append(dict_train_metrics)
        avg_FScore /= len(selected)
        # calculating weigted average score across clients
        weighted_average = ut.weighted_average_test_score(test_acc_list, test_size_list)
        avg_FScore /= len(selected)
        print("avg test FScore %f" % avg_FScore)
        print(f"average weighted FScore {weighted_average}")

        nets_list = list(nets.values())
        return nets_list

#F = FedProxServer()
