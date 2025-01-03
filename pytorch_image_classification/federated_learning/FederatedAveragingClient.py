from torch import optim, nn

from pytorch_image_classification.federated_learning.FederatedClient import FederatedClient
from pytorch_image_classification.federated_learning.FederatedServer import compute_accuracy, print_confusion_matrix
from pytorch_image_classification.argument import argument
from utility import utility, plot_train_val_loss


class FederatedAveragingClient(FederatedClient):
    """
        used to simulate clients characteristic while doing federated learning
    """
    def train_local(self, net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu,
                    c_local, c_global, device="cpu", comm_round=0, data_set_len=None, val_loader=None, logger=None):
        """
            Used for training model for client. this function simulates the client training process
        :param global_net: global model shared by server
        :param mu: not required here
        :param c_local: not required here
        :param c_global: not required here
        :param val_loader: validation data loader
        :param logger: used for logging result to file
        :param val_df: validation dataset for each epoch
        :param data_set_len: length of local dataset
        :param comm_round:  current communication round
        :param net_id: Id of the client that is currently training
        :param net: Model of the client that is currently in training
        :param train_dataloader: train DataLoader of the client
        :param test_dataloader: test loader for checking accuracy at the end after training
        :param epochs: No of times training is repeated
        :param lr: Learning rate
        :param args_optimizer: name of the optimizer ex: adam, amsgrad, sgd
        :param device: computational device used
        :return: client model after training
        """

        min_metrics = None
        val_metrics = None
        print('Training network %s' % str(net_id))
        args = argument()
        optimizer = self.ut.get_Optimizer(net)
        criterion = self.ut.get_Criterion()
        min_fpr = 10
        val_losses = []
        cnt = 0
        if type(train_dataloader) == type([1]):
            pass
        else:
            train_dataloader = [train_dataloader]

        epoch_loss_collector = []
        loss = None
        # iterative training model for given epochs
        for epoch in range(epochs):
            for tmp in train_dataloader:
                for batch_idx, (x, target) in enumerate(tmp):
                    x, target = x.to(device), target.to(device)
                    #training model on  each batch of data
                    optimizer.zero_grad()
                    x.requires_grad = True
                    target.requires_grad = False

                    out = net(x)
                    loss = criterion(out, target)

                    loss.backward()
                    optimizer.step()

                    cnt += 1
            epoch_loss_collector.append(loss.item())
            #calculate optimum confidence value and different metrics with validation data loader
            val_metrics = compute_accuracy(net, val_loader, device=device, calc_all=True, with_threshold_opt=True)

            # finding the best validation metrics
            if epoch == 0:
                min_metrics = val_metrics
            if ('best_metrics' in val_metrics.keys() and min_metrics is not None and
                    min_metrics['best_metrics'] > val_metrics['best_metrics']):
                min_metrics = val_metrics

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            #print output of validation
            output_string = f'Machine {net_id} epoch {epoch} val_loss: {val_metrics["loss"]} val_F_score: {val_metrics["FScore"]}'
            if self.cfg.classification_type == "binary":
                output_string += f' val_FPR{val_metrics["fpr"]} val_recall: {val_metrics["recall"]}'
            print(output_string)
            logger.info(output_string)
            val_losses.append(val_metrics['loss'])
        #plotting training vs validation losss
        plot_train_val_loss(epoch_loss_collector, val_losses, filepath=self.logging_path,
                            comm_round=comm_round, client=net_id)
        #using the optimum threshold value computed earlier to test the resulting model from training
        if 'best_threshold' in min_metrics.keys():
            train_acc = compute_accuracy(net, train_dataloader, device=device,
                                         best_threshold=min_metrics['best_threshold'])
            test_metrics = compute_accuracy(net, test_dataloader, device=device, calc_all=True,
                                            best_threshold=min_metrics['best_threshold'])
        else:
            train_acc = compute_accuracy(net, train_dataloader, device=device)
            test_metrics = compute_accuracy(net, test_dataloader, device=device, calc_all=True)
        #plot confusion matrix
        print_confusion_matrix(net, test_dataloader, machine=net_id, comm_round=comm_round)
        print('>> Training accuracy: %f' % train_acc)
        print('>> Test Metrics: ', test_metrics)
        logger.info(f'>> Training accuracy:{train_acc}')
        logger.info(f'>> Test Metrics: {test_metrics}')

        net.to(device)
        print(' ** Training complete **')
        return train_acc, test_metrics['FScore'], min_metrics, val_metrics
