import torch
from torch import optim, nn

from pytorch_image_classification.federated_learning.FederatedClient import FederatedClient
from pytorch_image_classification.federated_learning.FederatedServer import compute_accuracy, print_confusion_matrix
from utility import plot_train_val_loss


class FedProxClient(FederatedClient):
    """
            used to simulate client characteristic while doing FedProx learning
    """
    def train_local(self, net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu,
                    c_local, c_global, device="cpu", comm_round=0, data_set_len=[0], val_loader=None, logger=None):
        """
            Used for training model for client. this function simulates the client training process
        :param net_id: current client:id
        :param net: current client model
        :param global_net: global model shared from server
        :param train_dataloader: dataloader for training data
        :param test_dataloader: dataloader for testing data
        :param epochs: total number of epochs
        :param lr: learning rate for optimizer
        :param args_optimizer: which optimizer to use
        :param mu: the hyperparameter for proximal term
        :param c_local: not required here
        :param c_global: not required here
        :param device: device for training
        :param comm_round: current communication round
        :param data_set_len: length of data set
        :param val_loader: validation data for current client
        :param logger: logger for current client
        :return: clients model result after training
        """
        min_metrics = None
        optimizer = None
        val_losses = []
        print('Training network %s' % str(net_id))
        # fetching optmizer
        optimizer = self.ut.get_Optimizer(net)
        # fetching loss function
        if self.cfg.classification_type == 'binary':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss().to(device)

        cnt = 0
        global_weight_collector = list(global_net.to(device).parameters())
        epoch_loss_collector = []
        # iterating through each epoch
        for epoch in range(epochs):
            for batch_idx, (x, target) in enumerate(train_dataloader):
                # training local model
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                # target = target.long()

                out = net(x)
                loss = criterion(out, target)

                # for FedProx adding proximal term to loss function
                fed_prox_reg = 0.0
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg

                loss.backward()
                optimizer.step()

                cnt += 1
            epoch_loss_collector.append(loss.item())

            # computing optimum confidence and other metrix with validation data
            val_metrics = compute_accuracy(net, val_loader, device=device, calc_all=True, with_threshold_opt=True)

            # finding the best validation metrics
            if epoch == 0:
                min_metrics = val_metrics
            if ('best_metrics' in val_metrics.keys() and min_metrics is not None and
                    min_metrics['best_metrics'] > val_metrics['best_metrics']):
                min_metrics = val_metrics
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

            # printing the result of validation
            output_string = f'Machine {net_id} epoch {epoch} loss: {epoch_loss} val_F_score: {val_metrics["FScore"]}'
            if self.cfg.classification_type == "binary":
                output_string += f' val_FPR{val_metrics["fpr"]} val_recall: {val_metrics["recall"]}'
            print(output_string)
            logger.info(output_string)
            val_losses.append(val_metrics['loss'])

        #plotting train vs validation loss
        plot_train_val_loss(epoch_loss_collector, val_losses, filepath=self.logging_path,
                            comm_round=comm_round, client=net_id)

        # using the best confidence value obtained from validation to test the local model on test data loader
        # and train data loader
        if 'best_threshold' in min_metrics.keys() and self.cfg.classification_type == "binary":
            train_F = compute_accuracy(net, train_dataloader, device=device,
                                       best_threshold=min_metrics['best_threshold'])
            test_metrics = compute_accuracy(net, test_dataloader, device=device, calc_all=True,
                                            best_threshold=min_metrics['best_threshold'])
        else:
            train_F = compute_accuracy(net, train_dataloader, device=device)
            test_metrics = compute_accuracy(net, test_dataloader, device=device, calc_all=True)
        print_confusion_matrix(net, val_loader, machine=net_id, comm_round=comm_round,filepath=self.logging_path)

        net.to(device)

        return train_F, test_metrics['FScore'], min_metrics, val_metrics
