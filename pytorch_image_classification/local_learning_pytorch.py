from torch.utils.data import ConcatDataset, DataLoader
from configuration import Configuration
from pytorch_image_classification.TestStandardDS.baseline_dataset_preparation import prepare_CIFAR10_dataloader
from pytorch_image_classification.data_preparation_pytorch import data_preparation

from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import fbeta_score
import torch.optim
import time
import utility as ut
from results_logging import ResultsLogger, logging_setup
from utility import utility, plot_stat
from config_parser.config_parser import dict_from_conf
from retrieve_data import Retrieve
import logging
import torch.nn.functional as F


def combine_data_loaders(test_loaders):
    """
    function can be used to combine the data loader which where seperated into parts
    during data preparation step
    :param test_loaders: list of data loaders to be combined
    :return:
    """
    datasets = [loader.dataset for loader in test_loaders]
    combined_dataset = ConcatDataset(datasets)

    # Create a single DataLoader for the combined dataset
    combined_test_loader = DataLoader(combined_dataset, batch_size=len(combined_dataset), shuffle=False)
    return combined_test_loader


class LearningLocal:
    """
        class containing functions for local learning pipeline
        function train_model should be called after initializing this class
        to start the local learning pipeline
    """

    def __init__(self, identifier=None, iteration=None):
        """
            constructor for local learning pipeline, initialises all the other classes that are required for
            proper working of local learning pipeline.
        :param identifier: unique identifier for the execution of training pipeline, is used as savefile name for plots
        :param iteration: used to identify the number of iteration the local learning pipeline is repeated
                            to calculate mean and standard deviation
        """
        #fetches configuration details
        self.cfg = Configuration()
        self.augment = None
        #initialising utitlity class
        self.utility = utility()
        #initialising retrieve class for retrieveing dataframe saved in files
        self.retrieve = Retrieve()
        # initialising data preparation class
        self.preprocess = data_preparation()
        # retreives data from file saved after dataset preparation
        self.train_parts, self.test_parts, self.val_parts = self.retrieve.retrieve_dataframe(iteration)
        # initialising class for saving log files.
        self.log = ResultsLogger(self.cfg, self.__class__.__name__, identifier)
        # setting up logger class for saving log files
        logging_setup(self.log.logging_path, logging)
        # check if standard dataset is used or not
        if not self.cfg.use_standard_dataset:
            #data preparation step to get dataloader after sampling and augmentation
            self.train_parts, self.test_parts, self.val_parts = self.preprocess.prepare_loaders_local_learning(
                self.train_parts,
                self.test_parts,
                self.val_parts)
        else:
            # data preparation step for standard dataset
            self.train_parts, self.test_parts, self.val_parts, _, _, _ = prepare_CIFAR10_dataloader()

        if self.cfg.plot.plot_stat_data_loader:
            #used to plot the data distribution plot for the data loader
            plot_stat(self.train_parts, "Train dataset", self.cfg.classification_type, self.log.logging_path)
            plot_stat(self.test_parts, "test dataset", self.cfg.classification_type, self.log.logging_path)
            plot_stat(self.val_parts, "validation dataset", self.cfg.classification_type, self.log.logging_path)

    def train_model(self, train_loader, val_loader, test_loader):
        """
            function for training given model on each local client with split data
        :param train_loader: training data loader
        :param val_loader: validation data loader
        :param test_loader: testing data loader
        """
        #initialising lists for collecting results
        train_loss_per_machine = []
        val_losses_per_machine = []
        train_correct_per_machine = []
        test_correct_per_machine = []
        F1Score_per_machine = []
        model_per_machine = []
        thresholds = []
        for j in range(len(train_loader)):
            #Iterate through each of the clients and perform training.
            #fetch model for training
            model = self.utility.get_Model()
            #fetch loss function
            criterion = self.utility.get_Criterion()
            #fetch optimizer
            optimizer = self.utility.get_Optimizer(model)
            start_time = time.time()
            # check device
            if torch.cuda.is_available() and self.cfg.device == 'cuda':
                model = model.cuda()
            #fetch epochs
            epochs = self.utility.get_Epoch('local')
            train_losses = []
            val_losses = []
            train_correct = []
            test_correct = []
            F1Scores = []
            # initialize min_fpr,best_validation_score as maximum possible
            # value to find the minimum later
            min_fpr = 100
            min_threshold = 0
            best_validation_score = -1
            best_model = None

            # For each client perform n epochs of training
            for i in range(epochs):
                train_corr = 0
                test_corr = 0
                # doing batch retrieve from the train_loader
                for b, (X_train, y_train) in enumerate(train_loader[j]):
                    if torch.cuda.is_available() and self.cfg.device == 'cuda':
                        X_train = X_train.cuda()
                        y_train = y_train.cuda()
                    b += 1
                    #forward propogation
                    y_pred = model(X_train)
                    #calculating loss
                    loss = criterion(y_pred, y_train)

                    #different final activation for binary and multiclass classification
                    if self.cfg.classification_type == "binary":
                        #calculating sigmoid for binary
                        predicted = torch.round(torch.sigmoid(y_pred))
                        # predicted = torch.where(y_pred.data > 0.5, 1, 0)
                        batch_corr = (predicted == y_train).sum()
                        train_corr += batch_corr

                    else:
                        # using index of maximum value for multiclass as target label
                        predicted = torch.argmax(y_pred.data, 1)
                        batch_corr = (predicted == torch.argmax(y_train, 1)).sum()
                        train_corr += batch_corr
                    # setting gradients to zero
                    optimizer.zero_grad()
                    #backpropogation to calculate new gradients
                    loss.backward()
                    #updating weights and biases of the training model
                    optimizer.step()
                #accumulating loss and number of correct prediction in lists
                train_losses.append(loss.item())
                train_correct.append(train_corr)

                # performing validation
                with torch.no_grad():
                    #initialy for validation the best metrics value saved in configuration file
                    # is used as confidence value.
                    best_threshold = self.cfg.metrics.threshold
                    #iterating through data points in validation loader
                    for b, (X_val, y_val) in enumerate(val_loader[j]):
                        #using appropriate device
                        if torch.cuda.is_available() and self.cfg.device == 'cuda':
                            X_val = X_val.cuda()
                            y_val = y_val.cuda()
                        # calculating predictions on trained model for current epoch
                        y_pred = model(X_val)
                        y_pred = y_pred.cpu()
                        y_val = y_val.cpu()
                        #calculating validation loss
                        val_loss = criterion(y_pred, y_val)
                        if self.cfg.classification_type == "binary":
                            #calculating prediction by applying sigmoid at the end
                            predicted = (torch.sigmoid(y_pred) > best_threshold).int()
                            #calculating optimum threshold for current client
                            best_threshold, best_metric = ut.optimize_threshold(torch.sigmoid(y_pred).squeeze().numpy(),
                                                                                y_val.squeeze().numpy(),
                                                                                metric="fpr")

                            test_corr += (predicted == y_val).sum()
                            # calulating which confidence value from different epochs has best validation score
                            if min_fpr > best_metric:
                                min_fpr = best_metric
                                min_threshold = best_threshold
                            #calculates different metrics like f1-score etc
                            metrics = ut.calculate_metrics_binary(y_val.squeeze(), predicted.squeeze(),
                                                                  beta=self.cfg.metrics.beta)
                            F1Score = metrics['FScore']
                        else:
                            predicted = torch.argmax(y_pred.data, 1)
                            test_corr += (predicted == torch.argmax(y_val, 1)).sum()
                            F1Score = multiclass_f1_score(predicted, torch.argmax(y_val, 1),
                                                          num_classes=self.cfg.no_of_classes
                                                          , average="weighted")
                            metrics = {'FScore': F1Score}
                        # calculates which model from different epochs show the best validation f1-score
                        # this model is used in testing for this client
                        if best_validation_score < F1Score:
                            best_validation_score = F1Score
                            best_model = model
                        #printing outputs for each epoch
                        output_string = f"Machine: {j} epoch: {i} loss: {loss.item()} validation F1Score: {metrics['FScore']}"
                        if self.cfg.classification_type == 'binary':
                            output_string += f" val_fpr: {metrics['fpr']} recall: {metrics['recall']}"
                        print(output_string)
                        logging.info(output_string)
                #saving results of the epoch to a list
                val_losses.append(val_loss.item())
                test_correct.append(train_corr)
                F1Scores.append(F1Score)
            #ploting train vs validation loss over epochs
            ut.plot_train_val_loss(train_losses, val_losses, client=j, filepath=self.log.logging_path)
            #accumulating best model, thresholds for testing to a list
            # also adds the last train, validation losses to a list
            thresholds.append(min_threshold)
            train_loss_per_machine.append(train_losses)
            val_losses_per_machine.append(val_losses)
            train_correct_per_machine.append(train_correct)
            test_correct_per_machine.append(test_correct)
            F1Score_per_machine.append(F1Scores)
            model_per_machine.append(best_model)
        #finding weighted average results for overall validation loss, train loss and f1-score for overall training
        averaging_scores(train_loss_per_machine, val_losses_per_machine, F1Score_per_machine)
        print("Training done testing started")

        F1Score_test_list = []
        test_size_list = []
        #combines test loader for testing against each model from different clients
        combined_test_loader = combine_data_loaders(test_loader)
        # testing output of each model of clients
        for i, model in enumerate(model_per_machine):
            # extract the entire test data
            X_test, Y_test = next(iter(combined_test_loader))
            test_size_list.append(len(Y_test))
            with torch.no_grad():
                if torch.cuda.is_available() and self.cfg.device == 'cuda':
                    X_test = X_test.cuda()
                    Y_test = Y_test.cuda()
                # predict output for model
                y_pred_test = model(X_test)
                y_pred_test = y_pred_test.cpu()
                Y_test = Y_test.cpu()
                if self.cfg.classification_type == "binary":
                    #calculating predictions using best threshold value from validation
                    predicted = (torch.sigmoid(y_pred_test) > thresholds[i]).int()
                    #calculate the f1-score and other metrics
                    metrics = ut.calculate_metrics_binary(Y_test.squeeze(), predicted.squeeze(),
                                                          beta=best_threshold)
                    F1Score = metrics['FScore']
                    #plotting ROC curve
                    ut.plot_area_under_curve(Y_test.squeeze(), torch.sigmoid(y_pred_test).squeeze(),
                                             self.log.logging_path, machine=i)
                    #Printing confusion matrix
                    ut.print_confusion_matrix_binary(predicted.squeeze(), Y_test.squeeze(), self.log.logging_path,
                                                     machine=i)
                else:
                    #calculating f1-score for multiclass classification
                    F1Score = multiclass_f1_score(torch.argmax(y_pred_test, 1), torch.argmax(Y_test, 1),
                                                  num_classes=self.cfg.no_of_classes
                                                  , average="weighted")
                    metrics = {'FScore': F1Score.numpy().tolist()}
                    #printing confusion matrix
                    ut.print_confusion_matrix(torch.argmax(y_pred_test, 1), Y_test, self.log.logging_path)
                logging.info(f'machine: {i} metrics {metrics}')
                F1Score_test_list.append(F1Score.tolist())

        # saving results to log files
        self.log.add_to_log(None, dict_from_conf(self.cfg))
        self.log.add_to_log(None, metrics)
        # self.log.add_to_log(None, {key: values.tolist() for key, values in model.state_dict().items()})
        self.log.save_log()

        logging.info(
            f'The weighted average F1Score is {ut.weighted_average_test_score(F1Score_test_list, test_size_list)}')


def averaging_scores(train_loss, test_loss, F1Scores):
    """
        used to calculate the average of  train loss, test loss and f1-score.
    :param train_loss: list of train-losses from different epochs on different clients
    :param test_loss: list of test-losses from different epochs on different clients
    :param F1Scores: list of f1-score from different epochs on different clients
    """
    #calculating average metrics per machine
    avg_train_loss_per_machine = [sum(row) / len(row) for row in train_loss]
    avg_test_loss_per_machine = [sum(row) / len(row) for row in test_loss]
    avg_F1Score_per_machine = [sum(row) / len(row) for row in F1Scores]

    #calculating average metrics across all machine
    avg_train_loss = sum(avg_train_loss_per_machine) / len(avg_train_loss_per_machine)
    avg_test_loss = sum(avg_test_loss_per_machine) / len(avg_test_loss_per_machine)
    avg_F1Score = sum(avg_F1Score_per_machine) / len(avg_F1Score_per_machine)
    print(f'Average train loss per machine: {avg_train_loss}')
    print(f'Average test loss per machine: {avg_test_loss}')
    print(f'Average F1Score per machine: {avg_F1Score}')
