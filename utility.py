import json
from typing import Dict

import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import torch
from sklearn.metrics import roc_auc_score, roc_curve, fbeta_score, recall_score, f1_score, confusion_matrix
from torch import nn

from config_parser import config_parser
from configuration import Configuration
from pytorch_image_classification.model import Net, Net2, prepare_resnet_model, prepare_VGG_model
from torchsummary import summary
from collections import Counter
from datetime import datetime


def configure_yaml():
    """
        fetches configuration details from files
        :return: configuration object can be used to lookup values of different constants etc.
    """
    cfg = Configuration()
    return cfg


def calculate_metrics_binary(truth_label: torch.Tensor, predicted_label: torch.Tensor, beta: int = 0.5) -> Dict:
    """
        function for calculating metrics for binary classification
    :param truth_label: target or y_train torch tensor of shape (N)
    :param predicted_label: prediction torch tensor of shape (N) where N is no of samples
    :param beta: Beta parameter for Fbeta score
    :return: Dictionary containing different metrics
    """
    F1Score = fbeta_score(predicted_label, truth_label, beta=beta)
    fpr, tpr, thresholds = roc_curve(truth_label, predicted_label)
    recall = recall_score(truth_label, predicted_label)
    return {'FScore': F1Score, 'fpr': fpr, 'tpr': tpr, 'threshold': thresholds, 'recall': recall}


def print_confusion_matrix(y_pred, y_val, filepath, machine=None, comm_round=None):
    """
        used to calculate and plot confusion matrix
    :param y_pred: prediction result from model
    :param y_val: actual target value
    :param filepath: filepath to which the plot will be saved
    :param machine: The id of client, if given will be added to the confusion matrix label
    :param comm_round: The communication round during federated learning, if given will be confusion
                        matrix label
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    cfg = configure_yaml()
    confusion = tf.math.confusion_matrix(
        labels=y_val.argmax(axis=1),
        predictions=y_pred,
        num_classes=cfg.no_of_classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    sn.heatmap(confusion, annot=True, cmap='Greens')  # font size
    if machine is None:
        plt.title("Confusion matrix")
    else:
        if comm_round is None:
            plt.title(f"Confusion Matrix: Client = {machine}")
        else:
            plt.title(f"Confusion Matrix: Client = {machine} round = {comm_round}")
    if filepath:
        now = datetime.now()  # Format the date and time as a string
        date_time_str = now.strftime("%Y-%m-%d%H-%M-%S")
        plt.savefig(f'{filepath}/CF{date_time_str}.svg', format='svg')
    plt.show()


def print_confusion_matrix_binary(y_pred, y_val, filepath, machine=None, comm_round=None, epoch=None):
    """
        plots binary confusion matrix
    :param y_pred: prediction result from model
    :param y_val: actual target value
    :param filepath: filepath to which the plot will be saved
    :param machine: The id of client, if given will be added to the confusion matrix label
    :param comm_round: The communication round during federated learning, if given will be confusion
                        matrix label
    :param epoch: the epoch at which the confusion matrix will be plotted, if given will be added to
                    confusion matrix label
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    cfg = configure_yaml()
    confusion = tf.math.confusion_matrix(
        labels=y_val,
        predictions=y_pred,
        num_classes=2)
    sn.heatmap(confusion, annot=True, cmap='Greens', fmt="d",
               xticklabels=["Predicted: 0", "Predicted: 1"],
               yticklabels=["Actual: 0", "Actual: 1"])  # font siz
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    label = f"Confusion Matrix"
    if machine is not None:
        label += f" | Client = {machine}"
    if comm_round is not None:
        label += f" | Round: {comm_round}"
    if epoch is not None:
        label += f" | Epoch: {epoch}"
    plt.title(label)
    if filepath:
        now = datetime.now()  # Format the date and time as a string
        date_time_str = now.strftime("%Y-%m-%d%H-%M-%S")
        plt.savefig(f'{filepath}/CFB{date_time_str}.svg', format='svg')
    plt.show()


def plot_train_val_loss(train_losses, val_losses, client=None, filepath=None, comm_round=None):
    """
        function to plot train vs validation losses across epochs
    :param train_losses: losses accumulated during training
    :param val_losses: validation losses accumulated during validation
    :param client: the id of client used, if given added to label of plot
    :param filepath: the path to which the plot should be saved
    :param comm_round: the communication round where plotting is done
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    epochs = list(range(1, len(train_losses) + 1))
    fig = plt.figure()
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', color='#97C139')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o', color='darkgreen')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    title = f'Loss Over Epochs'
    if client is not None:
        title += f'| Client: {client}'
    if comm_round is not None:
        title += f'| Round: {comm_round}'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if filepath:
        now = datetime.now()  # Format the date and time as a string
        date_time_str = now.strftime("%Y-%m-%d%H-%M-%S")
        data = {'train_losses': torch.tensor(train_losses).tolist(), 'val_losses': torch.tensor(val_losses).tolist()}
        plt.savefig(f'{filepath}/TVL{date_time_str}.svg', format='svg')
        with open(f'{filepath}/TVL{date_time_str}.json', 'w') as f:
            json.dump(data, f)
    plt.show()
    plt.close(fig)


def plot_loss_round(loss_data, message=None, filepath=None):
    """
        function used for plotting loss across rounds of federated learning
    :param loss_data: loss accumulated during each round
    :param message: Label to be printed
    :param filepath: path to which file will be saved
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    # Generate rounds automatically based on index
    rounds = list(range(1, len(loss_data) + 1))  # Round numbers: 1, 2, 3, ...
    clients = loss_data[0].keys()
    client_losses = {client: [] for client in clients}
    colors = cm.Greens(np.linspace(0, 1, len(clients)))
    for round_data in loss_data:
        for client in clients:
            client_losses[client].append(round_data[client]["loss"])
    # Plotting
    fig = plt.figure()
    for idx, (client, losses) in enumerate(client_losses.items()):
        plt.plot(rounds, losses, marker='o', linestyle='-', label=f'Client{client} Validation Loss')
        # plt.plot(rounds, loss_data, marker='o', linestyle='-', color='#97C139', label='Training Loss')
    label = f'Validation Loss vs Communication Rounds'
    if message is not None:
        label += f'| {message}'
    plt.title(label)
    plt.xlabel('Communication Rounds')
    plt.ylabel('Validation Loss')
    plt.grid(True)
    plt.legend()
    if filepath:
        now = datetime.now()  # Format the date and time as a string
        date_time_str = now.strftime("%Y-%m-%d%H-%M-%S")
        plt.savefig(f'{filepath}/LR{date_time_str}.svg', format='svg')
        with open(f'{filepath}/LR{date_time_str}.json', 'w') as f:
            json.dump(client_losses, f)
    plt.show()
    plt.close(fig)


def plot_score_round(f_score, message, filepath):
    """
        function to plot fscore across different rounds for federated learning
    :param f_score: fscore accumulated during each round
    :param message: message to be printed
    :param filepath: path to which file will be saved
    """
    cfg = Configuration()
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    rounds = list(range(1, len(f_score) + 1))  # Round numbers: 1, 2, 3, ...
    # Plotting
    data = {'fscore': f_score, 'round': rounds}
    fig = plt.figure()
    plt.plot(rounds, f_score, marker='o', linestyle='-', color='#97C139', label=f'Global F{cfg.metrics.beta} Score')
    label = f'Global F{cfg.metrics.beta} Score vs Communication Rounds '
    if message is not None:
        label += f'| {message}'
    plt.title(label)
    plt.xlabel('Communication Rounds')
    plt.ylabel(f'Global F{cfg.metrics.beta} Score')
    plt.grid(True)
    plt.legend()
    if filepath:
        now = datetime.now()  # Format the date and time as a string
        date_time_str = now.strftime("%Y-%m-%d%H-%M-%S")
        plt.savefig(f'{filepath}/FR{date_time_str}.svg', format='svg')
        with open(f'{filepath}/FR{date_time_str}.json', 'w') as f:
            json.dump(data, f)
    plt.show()
    plt.close(fig)


def weighted_average_test_score(F1Score_list, test_size_list):
    """
        function to calculate weighted average of f1_score for decentralized training
    :param F1Score_list: f1_score of each client
    :param test_size_list: data size of each client
    :return: weighted average of over each machine
    """
    score = 0
    for i in range(len(F1Score_list)):
        score += F1Score_list[i] * test_size_list[i]
    return score / sum(test_size_list)


def plot_stat(dataloaders, message, classification_type, filepath):
    """
        function to plot data distribution of data loaders
    :param dataloaders: the data loaders used in learning
    :param message: message to be added to label
    :param classification_type: binary or multiclass
    :param filepath: path to which the plot will be saved
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 15
    if type(dataloaders) == type([1]):
        pass
    else:
        dataloaders = [dataloaders]
    length = len(dataloaders)
    y_train_0_counts = []
    y_train_1_counts = []
    client_ids = []
    #iterate through each dataloader
    for i, dataloader in enumerate(dataloaders):
        client_ids.append(i)
        data = []
        for _, target in dataloader:

            if (classification_type == "multiclass"):
                classes = torch.argmax(target, dim=1).tolist()
                data.extend(classes)
            else:
                temp = target.squeeze().tolist()
                data.extend(temp)
        # calculating counts of each class
        data = Counter(data)
        labels = list(data.keys())
        values = list(data.values())
        y_train_0_counts.append(data[0.0])
        y_train_1_counts.append(data[1.0])
    x = np.arange(len(dataloaders))
    fig, ax = plt.subplots(figsize=(12, 8))
    bars0 = ax.bar(x, y_train_0_counts, label='y_train = 0', color='#97C139')
    bars1 = ax.bar(x, y_train_1_counts, bottom=y_train_0_counts, label='y_train = 1', color='green')
    ax.set_xlabel('Clients', fontsize=15)
    ax.set_ylabel('Count', fontsize=15)
    ax.set_title('Data Distribution of Each Client on ' + message, fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(client_ids)
    ax.legend()
    now = datetime.now()  # Format the date and time as a string
    date_time_str = now.strftime("%Y-%m-%d%H-%M-%S")
    plt.savefig(f'{filepath}/DD{i}{message}{date_time_str}.svg', format='svg')
    # Show the plot
    plt.show()
    plt.close(fig)


def plot_area_under_curve(y_test, y_pred, filepath=None, machine=None, comm_round=None, epoch=None):
    """
        function to plot ROC curve
    :param y_test: list of target labels
    :param y_pred: list of predictions from model
    :param filepath: path to which file will be saved
    :param machine: the id of client, if given will be added to label of plot
    :param comm_round: the round at which plotting is done, if given will be added to label of plot
    :param epoch: the epoch at will the diagram is plotted, if given will be added to label of plot
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': auc}
    label = f"AUC = {auc:.2f}"
    if machine is not None:
        label += f" | Client = {machine}"
    if comm_round is not None:
        label += f" | Round: {comm_round}"
    if epoch is not None:
        label += f" | Epoch: {epoch}"
    # Plot ROC curve
    plt.plot(fpr, tpr, label=label, color='#97C139')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    if filepath:
        now = datetime.now()  # Format the date and time as a string
        date_time_str = now.strftime("%Y-%m-%d%H-%M-%S")
        plt.savefig(f'{filepath}/AOC{date_time_str}.svg', format='svg')
        with open(f'{filepath}/AOC_data{date_time_str}.json', 'w') as f:
            json.dump(roc_data, f)
    plt.show()


def calculate_fpr(y_true, y_pred):
    """
        function to calculate FPR
    :param y_true: list of target labels
    :param y_pred: list of prediction label from model
    :return: FPR of the given arrays
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    return fpr


def calculate_threshold_from_predictions(y_pred, y_true, ml_type, max_false_positive_rate):
    """
        function to calculate the best confidence value from target and prediction of validation
    :param y_pred: predicted values of label from model
    :param y_true: actual values of label
    :param ml_type: classification type eith binary_classification or classification
    :param max_false_positive_rate: maximum allowable FPR
    :return: calculated best threshold value
    """
    y_true_neg, y_pred_neg = None, None
    if ml_type == "binary_classification":
        # apply element[0]
        for element in y_pred:
            # if element is not a 0-dim tensor
            y_pred = [element[0] if hasattr(element, "shape") and len(element.shape) > 0 else element for element in
                      y_pred]
            # Get indices of y_true which are 0
            y_true_neg = [i for i, x in enumerate(y_true) if x == 0]
            # Filter y_pred and y_true for y_true_0
            y_pred_neg = [y_pred[i] for i in y_true_neg]
    elif ml_type == "classification":
        # apply softmax to y_pred
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        # get indices of labels which are not the positive class
        y_true_neg = [i for i, x in enumerate(y_true) if x != 0]
        # filter y_pred and y_true for y_true_neg
        y_pred_neg = [y_pred[i] for i in y_true_neg]
        y_pred_neg = [element[0] for element in y_pred_neg]
    if len(y_true_neg) == 0:
        # logging.warning("No negative class found in validation data. Using all classes for threshold calculation.")
        return None
    max_false_positives = len(y_pred_neg) * max_false_positive_rate
    y_pred_neg.sort(reverse=True)
    threshold = y_pred_neg[int(max_false_positives)]
    return threshold


# Optimize threshold
def optimize_threshold(probabilities, true_labels, metric="fpr", tolerance=0.01):
    """
        function used to facilitate calulating the best confidence value
    :param probabilities: predicted probabilities from model
    :param true_labels: actual target labels
    :param metric: for which metrics is confidence adjusted to
    :param tolerance: maximum tolerance allowed
    :return:
    """
    cfg = Configuration()
    best_threshold = calculate_threshold_from_predictions(probabilities, true_labels, "binary_classification",
                                                          cfg.metrics.min_threshold)
    predictions = (probabilities >= best_threshold).astype(int)
    fpr = calculate_fpr(true_labels, predictions)
    return best_threshold, fpr

class utility:
    """
        basic utility functions are added in this class
    """
    def __init__(self):
        #fetching configuration file
        self.cfg = Configuration()

    def get_Model(self, device=None):
        """
        returns model based on the keyword specified in configurations
        :device type of device model is trainied on
        :return:
        """

        model = None
        if self.cfg.classification_type == "multiclass":
            if self.cfg.model == 'basic':
                model = Net(self.cfg.no_of_classes)
            elif self.cfg.model == 'improved':
                model = Net2(self.cfg.no_of_classes)
            elif self.cfg.model == 'resnet':
                model = prepare_resnet_model(self.cfg.no_of_classes)
            elif self.cfg.model == 'VGG':
                model = prepare_VGG_model(self.cfg.no_of_classes)
        else:
            # binary classification the no of neurons in last layer is 1
            if self.cfg.model == 'basic':
                model = Net(1)
            elif self.cfg.model == 'improved':
                model = Net2(1)
            elif self.cfg.model == 'resnet':
                model = prepare_resnet_model(1)
            elif self.cfg.model == 'VGG':
                model = prepare_VGG_model(1)
        print(summary(model, (3, 224, 224)))
        if device:
            model.to(device)
        return model

    def get_Criterion(self):
        """
        returns loss function based on the keyword given
        :return: loss function
        """
        if self.cfg.classification_type == "binary":
            if self.cfg.binary.loss == "binary_crossentropy":
                return nn.BCEWithLogitsLoss()
        else:
            if self.cfg.multi_class.loss == 'categorical_crossentropy':
                return nn.CrossEntropyLoss()

    def get_Optimizer(self, model):
        """
            function to retrieve optimizer
        :param model: model the optimizer will be used on
        :return: optimizer
        """
        if self.cfg.optimizer == 'adam':
            if self.cfg.model == 'basic':
                return torch.optim.Adam(model.parameters(), self.cfg.local_learning.basic_convolution.learning_rate)
            if self.cfg.model == 'improved':
                return torch.optim.Adam(model.parameters(), self.cfg.local_learning.improved_convolution.learning_rate)
            if self.cfg.model == 'resnet':
                return torch.optim.Adam(model.parameters(), self.cfg.local_learning.resnet_50.learning_rate)
            if self.cfg.model == 'VGG':
                return torch.optim.Adam(model.parameters(), self.cfg.local_learning.VGG.learning_rate)
        if self.cfg.optimizer == 'SGD':
            if self.cfg.model == 'basic':
                return torch.optim.SGD(model.parameters(), self.cfg.local_learning.basic_convolution.learning_rate)
            if self.cfg.model == 'improved':
                return torch.optim.SGD(model.parameters(), self.cfg.local_learning.improved_convolution.learning_rate)
            if self.cfg.model == 'resnet':
                return torch.optim.SGD(model.parameters(), lr=self.cfg.local_learning.resnet_50.learning_rate)
            if self.cfg.model == 'VGG':
                return torch.optim.SGD(model.parameters(), lr=self.cfg.local_learning.VGG.learning_rate)

    def get_Optimizer_with_momentum_weight_decay(self, model, mu, weight_decay):
        """
            function to retrieve momentum based optimizers
        :param model: model used in training
        :param mu: momentum
        :param weight_decay: weight_decay
        :return: optimizers
        """
        if self.cfg.optimizer == 'adam':
            if self.cfg.model == 'basic':
                return torch.optim.Adam(model.parameters(), self.cfg.local_learning.basic_convolution.learning_rate,
                                        weight_decay=weight_decay)
            if self.cfg.model == 'improved':
                return torch.optim.Adam(model.parameters(), self.cfg.local_learning.improved_convolution.learning_rate,
                                        weight_decay=weight_decay)
            if self.cfg.model == 'resnet':
                return torch.optim.Adam(model.parameters(), self.cfg.local_learning.resnet_50.learning_rate,
                                        weight_decay=weight_decay)
        if self.cfg.optimizer == 'SGD':
            if self.cfg.model == 'basic':
                return torch.optim.SGD(model.parameters(), self.cfg.local_learning.basic_convolution.learning_rate,
                                       momentum=mu,
                                       weight_decay=weight_decay)
            if self.cfg.model == 'improved':
                return torch.optim.SGD(model.parameters(), self.cfg.local_learning.improved_convolution.learning_rate,
                                       momentum=mu,
                                       weight_decay=weight_decay)
            if self.cfg.model == 'resnet':
                return torch.optim.SGD(model.parameters(), self.cfg.local_learning.resnet_50.learning_rate, momentum=mu,
                                       weight_decay=weight_decay)

    def get_Epoch(self, flag=None):
        """
            to return the epoch based on which type of learning pipeline is used
        :param flag: local, central returns designated epochs for local and central learning pipeline
                        else if basic, improved, resnet, VGG is given epoch based on model architecture
                        described in learning_configuration.yaml is taken
        :return: number of epochs
        """
        if flag == 'local':
            return self.cfg.local.epoch
        elif flag == 'central':
            return self.cfg.central.epoch
        if self.cfg.model == 'basic':
            return self.cfg.local_learning.basic_convolution.epochs
        if self.cfg.model == 'improved':
            return self.cfg.local_learning.improved_convolution.epochs
        if self.cfg.model == 'resnet':
            return self.cfg.local_learning.resnet_50.epochs
        if self.cfg.model == 'VGG':
            return self.cfg.local_learning.VGG.epochs


def initialise_device():
    """
       used for checking which type of device is being used.
    :return: device type
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    return device
