from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, fbeta_score
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score

from configuration import Configuration
from pytorch_image_classification.TestStandardDS.baseline_dataset_preparation import prepare_CIFAR10_dataloader
from pytorch_image_classification.argument import argument
from pytorch_image_classification.custom_dataset_pytorch import CustomDatasetPytorch
from pytorch_image_classification.data_preparation_pytorch import data_preparation

from results_logging import ResultsLogger, logging_setup
from retrieve_data import Retrieve
import utility as ut
import logging


def prepare_global_test_loader(test_parts):
    """
        here all the individual test loader for each part and concatenated into a single dataframe
        and then a Dataloader is prepared from it.
    :param test_parts: data frame to be concatenated.
    :return: concatenated  dataloader
    """
    test = pd.concat(test_parts)
    test_dataset = CustomDatasetPytorch(test)
    test_loader = DataLoader(test_dataset, batch_size=len(test))
    return test_dataset, test_loader


def print_confusion_matrix(model, dataloader, machine=None, comm_round=None, epoch=None, filepath=None):
    """
        function used to plot confusion matrix
    :param model: model with which the confusion matrix is to be plotted
    :param dataloader: the data which the model is to be evaluated on
    :param machine: if provided will be added to label, the id of client
    :param comm_round: if provided will be added to label, the current communication round
    :param epoch: if provided will be added to label, the current epoch
    :param filepath: the location to which the plot will be stored if given
    """
    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]
    pred_label = None
    tar = None
    cfg = Configuration()
    if torch.cuda.is_available() and cfg.device == 'cuda':
        model = model.cuda()
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                if torch.cuda.is_available() and cfg.device == 'cuda':
                    x = x.cuda()
                    target = target.cuda()
                pred_label = model(x)
                tar = target
    pred_label = pred_label.cpu()
    tar = tar.cpu()
    if cfg.classification_type == "binary":
        predicted = torch.round(torch.sigmoid(pred_label)).squeeze()
        # plot confusion matrix and ROC curve is Binary
        ut.print_confusion_matrix_binary(predicted, tar.squeeze(), filepath=filepath,machine= machine, comm_round=comm_round,epoch=epoch)
        ut.plot_area_under_curve(tar.squeeze(), torch.sigmoid(pred_label).squeeze(), filepath, machine, comm_round, epoch)
    else:
        # plot confusion matrix if multiclass classification
        ut.print_confusion_matrix(torch.argmax(pred_label, 1), tar.data,filepath)


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu", calc_all=False,
                     with_threshold_opt=False, best_threshold=0.5):
    """
        Used to calculate performance metrics of the model on given data loader
    :param best_threshold: if given all the metric calculation will be done with this confidence value
    :param with_threshold_opt: If threshold optimization is to be done on validation set True
    :param model: machine learning model for calculating metric
    :param dataloader: Data for evaluating the model
    :param get_confusion_matrix: to return confusion matrix, if true returns FScore and confusion matrix
    :param device: cpu or cuda
    :param calc_all: if True calculates metrics like FScore , fpr, tps and threshold and returns a dictionary
    :return: either (f1-score, confusion_matrix) or {f1Score, fpr, tpr, threshold}
    """
    global conf_matrix
    cfg = Configuration()
    probabilities = []
    true_labels = []
    utility = ut.utility()
    was_training = False
    metrics = None

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]
    if torch.cuda.is_available() and cfg.device == "cuda":
        model = model.cuda()
    correct, total = 0, 0
    with torch.no_grad():
        # iterate throught the data loader
        for tmp in dataloader:
            # iterate through the batches in data loader
            for batch_idx, (x, target) in enumerate(tmp):
                # evaluating the model on given data loader
                x, target = x.to(device), target.to(device, dtype=torch.int64)
                out = model(x)
                criterion = utility.get_Criterion()
                x = x.cpu()
                out = out.cpu()
                target = target.cpu()
                if cfg.classification_type == "binary":
                    # performs sigmoid activation and then find binary prediction
                    # based on the best_threshold value provided
                    predicted = torch.sigmoid(out)
                    predicted = (predicted > best_threshold).int()
                    if with_threshold_opt:
                        # saving probabilities for calculating threshold
                        probabilities.extend(torch.sigmoid(out).squeeze().cpu().numpy())
                        true_labels.extend(target.squeeze().cpu().numpy())
                    pred_label = predicted.squeeze()
                    total += x.data.size()[0]
                    tar = target.squeeze()

                    # calculating the different metrics for binary classification result from model
                    metrics = ut.calculate_metrics_binary(target.squeeze(), predicted.squeeze(), beta=cfg.metrics.beta)
                    F1Score = metrics['FScore']
                else:
                    _, pred_label = torch.max(out.data, 1)
                    total += x.data.size()[0]
                    tar = torch.argmax(target.data, dim=1)

                    # calculating multiclass f1 score
                    F1Score = multiclass_f1_score(torch.argmax(out, 1), torch.argmax(target.data, 1),
                                                  num_classes=cfg.no_of_classes
                                                  , average="weighted")

                    metrics = {'FScore': F1Score.numpy().tolist()}
                # correct += (pred_label == tar).sum().item()

                # saving the list of probabilities
                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, tar.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, tar.cpu().numpy())
                loss = criterion(out, target.to(torch.float32))
                metrics['loss'] = loss.item()

    # gets the confusion matrix
    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return F1Score, conf_matrix

    # calculating optimal threshold from saved probabilities if with_threshold_opt is true
    if with_threshold_opt and cfg.classification_type == "binary":
        probabilities = np.array(probabilities)
        true_labels = np.array(true_labels)
        best_threshold, best_metric = ut.optimize_threshold(probabilities, true_labels, metric="fpr")
        metrics['best_threshold'] = best_threshold
        metrics['best_metric'] = best_metric
        return metrics

    if calc_all:
        return metrics

    return F1Score


def calculate_best_threshold_global(metrics):
    """
        averaging the thresholds from different client to obtain global threshold
    :param metrics: dictionary containing performance metrics from different clients.
    :return: optimum confidence value of global model
    """
    threshold = 0
    for metric in metrics:
        threshold += metric['best_threshold']
    threshold /= len(metrics)
    print("Aggregated threshold: ", threshold)
    return threshold


def calculate_class_distribution(dataloaders):
    """
        iterates through the dataloaders and create a dictionary with key net_id and value another dictionary
        containing class distribution.
    :param dataloaders:
    :return:
    """
    net_cls_counts = {}
    for net_i, dataloader in enumerate(dataloaders):
        Y_train = []
        for b, (X, y) in enumerate(dataloader):
            # fetch all the y_train from batches
            Y_train.append(y)
        # un-batch and combine to a single list
        y_train = np.concatenate(Y_train, axis=0)
        unq, unq_cnt = np.unique(y_train, return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts


class FederatedServer(ABC):
    """
        abstract class for simulating federated learning server
    """
    def __init__(self, identifier=None, iteration=None):
        """
            constructor for federated learning server, performing data preprocessing steps
            and making data available for child classes.
        :param identifier: unique identifier for this federated learning experiment
        :param iteration: iteration number for this experiment run
        """
        self.send_slow_rate = 0.0
        self.train_slow_rate = 0.0
        # initialising configuration class
        self.cfg = Configuration()
        self.augment = None
        # initialising retrieve class
        self.retrieve = Retrieve()
        # initialising data preprocessing class
        self.preprocess = data_preparation()
        # initialising utility class
        self.utility = ut.utility()
        self.train_slow_clients = []
        self.send_slow_clients = []
        self.clients = []
        self.metrics = []
        self.global_metrics = []
        self.train_metrics_dict_list = []
        self.args = argument()
        # saving global model for scaffold aggregation
        self.model_global_scaffold = self.utility.get_Model()
        # retrieve data frame from save files
        self.train_parts, self.test_parts, self.val_parts = self.retrieve.retrieve_dataframe(iteration)
        classname = self.__class__.__name__
        # initialising results logger class
        self.log = ResultsLogger(self.cfg, classname, identifier)
        # setting up logger to save logs to file
        logging_setup(self.log.logging_path, logging)

        if not self.cfg.use_standard_dataset:
            # preparing global test loader and dataset
            self.test_dataset, self.global_test_loader = prepare_global_test_loader(self.test_parts)
            # preparing global train loader and dataset
            self.train_dataset, self.global_train_loader = prepare_global_test_loader(self.train_parts)
            # preparing local train, test and validation data loader
            self.train_parts, self.test_parts, self.val_parts = self.preprocess.prepare_loaders_local_learning(
                self.train_parts,
                self.test_parts,
                self.val_parts)
        else:
            # preparing data loaders from standard dataset
            self.train_parts, self.test_parts, self.val_parts, self.global_train_loader, self.global_test_loader, self.global_val_loader = prepare_CIFAR10_dataloader()
        if self.cfg.plot.plot_stat_data_loader:
            # plotting the label distribution of data amoung different clients.
            ut.plot_stat(self.train_parts, "Train dataset", self.cfg.classification_type, filepath=self.log.logging_path)
            ut.plot_stat(self.test_parts, "test dataset", self.cfg.classification_type, filepath=self.log.logging_path)
            ut.plot_stat(self.val_parts, "validation dataset", self.cfg.classification_type, filepath=self.log.logging_path)

    def get_netdata_idx_map(self):
        """
            prepare a dictionary with key net_id and value train_dataframe
            each entry in the dictionary is a train data for each client
        :return: dictionary with key net_id and value train_dataframe
        """
        return {i: self.train_parts[i] for i in range(self.cfg.no_of_local_machines)}

    def init_nets(self, types='local', is_control_variate=False):
        """
            prepare a dictionary with key net_id and value the net used for local training in the machines.
        :param is_control_variate: initialiseing for control variate in scaffold if true
        :param types: local ,if: client net is being prepared else global model intialisation
        :return:
            nets: dictionary
            model_meta_data: information about layers in the model
            layer_type: what type of layers are included in the model
        """
        nets = {}
        if types == 'local':
            for net_i in range(self.cfg.no_of_local_machines):
                nets[net_i] = self.utility.get_Model()
        else:
            nets[0] = self.utility.get_Model()
        model_meta_data = []
        layer_type = []
        for (k, v) in nets[0].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)

        return nets, model_meta_data, layer_type

    def select_slow_clients(self, slow_rate=0.0):
        """
            initialise clients where training is slower
        :param slow_rate: the rate of clients from total which is supposed to be slow
        :return: clients with slow_client is True
        """
        slow_clients = [False for i in range(self.cfg.no_of_local_machines)]
        idx = [i for i in range(self.cfg.no_of_local_machines)]
        idx_ = np.random.choice(idx, int(slow_rate * self.cfg.no_of_local_machines))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_clients(self, clientObj):
        """
            function used by scaffold aggregation server to set the model for each client. as we are not enabling
            different client to be slow all the model is initialised with same global model
        :param clientObj:
        """
        for i, train_slow, send_slow in zip(range(self.cfg.no_of_local_machines), self.train_slow_clients,
                                            self.send_slow_clients):
            train_data = self.train_parts[i]
            test_data = self.test_parts[i]
            client = clientObj(i, self.model_global_scaffold)
            self.clients.append(client)

    def set_slow_clients(self):
        """
            used by scaffold aggregation server to set the slow client if needed
        """
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        """
            selecting clients for training during current communication round.
        :return: selected clients and array containing selected client index
        """
        arr = np.arange(self.cfg.no_of_local_machines)
        np.random.shuffle(arr)
        selected = arr[:int(self.cfg.no_of_local_machines * self.cfg.federated_learning.sample)]
        selected_clients = [self.clients[i] for i in selected]
        return selected_clients, selected

    @abstractmethod
    def get_init_nets(self, args):
        pass

    @abstractmethod
    def start_aggregation(self, args, net_dataidx_map):
        pass
