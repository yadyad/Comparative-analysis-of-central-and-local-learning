import pandas as pd
import torch.optim
import torchvision
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler, DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from torchvision import transforms

from config_parser.config_parser import dict_from_conf
from configuration import Configuration
from pytorch_image_classification.TestStandardDS.baseline_dataset_preparation import prepare_CIFAR10_dataloader, \
    CIFAR10OneHot
from pytorch_image_classification.data_preparation_pytorch import data_preparation
import time
import utility as ut
from results_logging import ResultsLogger, logging_setup
from retrieve_data import Retrieve
import logging





class CentralLearning:
    def __init__(self, identifier=None, iteration=None):
        self.train = None
        self.test = None
        self.val = None
        self.augment = None
        self.cfg = Configuration()
        self.retreive = Retrieve()
        self.utility = ut.utility()
        self.preprocess = data_preparation()
        self.log = ResultsLogger(self.cfg, self.__class__.__name__,identifier)
        logging_setup(self.log.logging_path, logging)
        self.train_parts, self.test_parts, self.val_parts = self.retreive.retrieve_dataframe(iteration)
        if not self.cfg.use_standard_dataset:
            self.train, self.val, self.test = self.preprocess.prepare_loaders_centralised_learning(self.train_parts,
                                                                                                   self.test_parts,
                                                                                                   self.val_parts)
        else:
            self.train, self.test, self.val, _, _, _ = prepare_CIFAR10_dataloader(split=False)
        if self.cfg.plot.plot_stat_data_loader:
            ut.plot_stat(self.train, "Train dataset", self.cfg.classification_type, self.log.logging_path)
            ut.plot_stat(self.test, "test dataset", self.cfg.classification_type, self.log.logging_path)
            ut.plot_stat(self.val, "validation dataset", self.cfg.classification_type, self.log.logging_path)


    def train_central_basic(self, train_loader, val_loader, test_DataLoader):
        """
        function for training centralised model based on given configuration
        :param train_loader:
        :param val_loader:
        :param test_DataLoader:
        :return:
        """
        model = self.utility.get_Model()
        criterion = self.utility.get_Criterion()
        optimizer = self.utility.get_Optimizer(model)
        if torch.cuda.is_available() and self.cfg.device == 'cuda':
            model = model.cuda()
        start_time = time.time()

        epochs = self.utility.get_Epoch(flag='central')
        train_losses = []
        test_losses = []
        train_correct = []
        test_correct = []
        F1Scores = []
        min_fpr = 100
        min_threshold = 0

        best_validation_score = -1
        best_model = None
        best_threshold = self.cfg.metrics.threshold
        for i in range(epochs):
            train_corr = 0
            test_corr = 0

            for b, (X_train, y_train) in enumerate(train_loader):
                if torch.cuda.is_available() and self.cfg.device == 'cuda':
                    X_train = X_train.cuda()
                    y_train = y_train.cuda()
                b += 1
                optimizer.zero_grad()
                y_pred = model(X_train)
                loss = criterion(y_pred, y_train)

                predicted = torch.argmax(y_pred.data, 1)
                batch_corr = (predicted == torch.argmax(y_train, 1)).sum()
                train_corr += batch_corr

                loss.backward()
                optimizer.step()

            train_losses.append(loss.item())
            train_correct.append(train_corr)
            #validation
            with torch.no_grad():
                for b, (X_test, y_test) in enumerate(val_loader):
                    if torch.cuda.is_available() and self.cfg.device == 'cuda':
                        X_test = X_test.cuda()
                        y_test = y_test.cuda()
                    y_val = model(X_test)
                    if self.cfg.classification_type == "binary":
                        #calculating prediction based on given threshold on configuration file
                        predicted = (torch.sigmoid(y_val) > best_threshold).int()
                        y_test = y_test.cpu()
                        predicted = predicted.cpu()
                        y_val = y_val.cpu()
                        #calculating new confidence value
                        best_threshold, best_metric = ut.optimize_threshold(torch.sigmoid(y_val).squeeze().numpy(),
                                                                            y_test.squeeze().numpy(),
                                                                            metric="fpr")

                        if min_fpr > best_metric:
                            min_fpr = best_metric
                            min_threshold = best_threshold
                        # predicted = torch.where(y_val.data > 0.5, 1, 0)
                        batch_corr = (predicted == y_test).sum()
                        train_corr += batch_corr


                        metrics = ut.calculate_metrics_binary(y_test.squeeze(), predicted.squeeze(),
                                                              beta=self.cfg.metrics.beta)
                        F1Score = metrics['FScore']
                        ut.plot_area_under_curve(y_test.squeeze(), torch.sigmoid(y_val).squeeze(), epoch=i, filepath=self.log.logging_path)
                        ut.print_confusion_matrix_binary(predicted.squeeze(), y_test.squeeze(), epoch=i, filepath=self.log.logging_path)
                    else:
                        predicted = torch.argmax(y_val.data, 1)
                        predicted = predicted.cpu()
                        F1Score = multiclass_f1_score(predicted, torch.argmax(y_test, 1),
                                                      num_classes=self.cfg.no_of_classes
                                                      , average="weighted")
                        metrics = {'FScore': F1Score}
                    output_string = f"epoch {i} loss {loss.item()}  F1Score {metrics['FScore']}"
                    if self.cfg.classification_type == "binary":
                        output_string += f"val_fpr: {metrics['fpr']} recall: {metrics['recall']}"
                    print(output_string)
                    logging.info(output_string)
                    if best_validation_score < F1Score:
                        best_validation_score = F1Score
                        best_model = model

            loss = criterion(y_val, y_test)
            test_losses.append(loss.item())
            test_correct.append(train_corr)
            F1Scores.append(F1Score)

        ut.plot_train_val_loss(train_losses,test_losses, filepath=self.log.logging_path)
        print("Training done validation started")
        X_test, Y_test = next(iter(test_DataLoader))
        with torch.no_grad():
            if torch.cuda.is_available() and self.cfg.device == 'cuda':
                X_test = X_test.cuda()
                Y_test = Y_test.cuda()
            #the best model from different epochs is used for testing
            y_pred_test = best_model(X_test)
            y_pred_test = y_pred_test.cpu()
            Y_test = Y_test.cpu()
            if self.cfg.classification_type == "binary":
                #calcualting prediction based on threshold calculated during valiation
                predicted = (torch.sigmoid(y_pred_test) > min_threshold).int()
                metrics = ut.calculate_metrics_binary(Y_test.squeeze(), predicted.squeeze(), beta=self.cfg.metrics.beta)
                F1Score = metrics['FScore']
                ut.plot_area_under_curve(Y_test.squeeze(), torch.sigmoid(y_pred_test).squeeze(), filepath=self.log.logging_path)
                ut.print_confusion_matrix_binary(predicted.squeeze(), Y_test.squeeze(), filepath=self.log.logging_path)
            else:
                F1Score = multiclass_f1_score(torch.argmax(y_pred_test, 1), torch.argmax(Y_test, 1),
                                              num_classes=self.cfg.no_of_classes
                                              , average="weighted")
                metrics = {'FScore': F1Score.numpy().tolist()}
                ut.print_confusion_matrix(torch.argmax(y_pred_test, 1), Y_test,filepath=self.log.logging_path)
            print(f'final metrics is {metrics}')
            logging.info(f'final metrics is {metrics}')
        #saving results to logs
        if self.cfg.logging:
            self.log.add_to_log(None, dict_from_conf(self.cfg))
            self.log.add_to_log(None, metrics)
            # self.log.add_to_log(None, {key: values.tolist() for key, values in model.state_dict().items()})
            self.log.save_log()

    def standard_model(self):
        """
            dataset preparation for standard dataset
        """
        data_dir = './data'  # directory of the cifar-10 data you downloaded
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        trainset = CIFAR10OneHot(root='./data', train=True, download=True,
                                 transform=transforms.ToTensor())
        testset = CIFAR10OneHot(root='./data', train=False, download=True,
                                transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=2)
        # The 10 classes in the dataset
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # to get the length of the taindata
        print(len(trainset))
        # get sample of train data and see length
        sample = next(iter(trainset))
        print(len(sample))
        # get the image and it's label
        image, label = sample
        print(type(image))
        print(type(label))
        # view image shape
        # image.shape
        # length of test data
        print(len(testset))


def print_distribution(data_loader):
    """
        calucate label distribution
    :param data_loader:
    :return:
    """
    from collections import Counter
    import torch
    import numpy as np

    # Assuming `dataloader` is your PyTorch DataLoader
    def get_target_distribution(dataloader):
        # Initialize an empty list to store all targets
        all_targets = []

        # Iterate through the DataLoader
        for _, targets in dataloader:
            # Append targets to the list (convert to NumPy or list if necessary)
            # Convert targets to a Python list
            result = torch.argmax(targets, dim=1).numpy().tolist()
            all_targets.extend(result)

        # Count occurrences of each class
        target_distribution = Counter(all_targets)
        return target_distribution

    # Example usage
    # Assuming `dataloader` is your DataLoader
    distribution = get_target_distribution(data_loader)

    # Print or visualize the distribution
    print("Target Distribution:", distribution)



