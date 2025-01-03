from typing import Tuple, Dict, List, Any

import numpy as np
import torch
from cv2.typing import Scalar
from flwr.common import NDArrays
from torch import nn

from configuration import Configuration
from pytorch_image_classification.argument import argument
from pytorch_image_classification.federated_learning.FedNova.NovaOptimizer import ProxSGD
from pytorch_image_classification.federated_learning.FederatedClient import FederatedClient
from pytorch_image_classification.federated_learning.FederatedServer import compute_accuracy, print_confusion_matrix
from utility import plot_train_val_loss


class FedNovaClient(FederatedClient):

    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.args = argument()

    def train_local(self, net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu,
                    c_local, c_global, device="cpu", comm_round=0, data_set_len=None, val_loader=None, logger=None):
        """Implement distributed fit function for a given client.
        :param val_loader:
        :param data_set_len:
        :param comm_round:
        """
        #  global_parameter setting
        # self.set_parameters(parameters)

        self.optimizer = ProxSGD(
            params=net.parameters(),
            ratio=data_set_len[net_id] / sum(data_set_len),
            lr=self.args.lr,
            momentum=self.args.mu,
            weight_decay=self.args.weight_decay,
            gmf=self.args.gmf,
            mu=0,
        )

        if self.args.var_local_epochs:
            seed_val = (
                    2023
                    + int(net_id)
                    + int(comm_round)
                    + int(self.args.seed)
            )
            np.random.seed(seed_val)
            num_epochs = np.random.randint(
                self.args.var_min_epochs, self.args.var_max_epochs
            )
        else:
            num_epochs = epochs

        loss_list, min_metrics, val_metrics = train(
            net, self.optimizer, train_dataloader, device, epochs=num_epochs, val_loader=val_loader, net_id=net_id,
            comm_round=comm_round,logger=logger, filepath=self.logging_path
        )

        # Get ratio by which the strategy would scale local gradients from each client
        # We use this scaling factor to aggregate the gradients on the server
        grad_scaling_factor: Dict[str, float] = self.optimizer.get_gradient_scaling()
        results = (self.get_parameters({}), train_dataloader, grad_scaling_factor)
        return loss_list, results, min_metrics, val_metrics

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        params = [
            val["cum_grad"].cpu().numpy()
            for _, val in self.optimizer.state_dict()["state"].items()
        ]
        return params


def train(
        model, optimizer, trainloader, device, epochs, proximal_mu=0.0, val_loader=None,
        net_id=None, comm_round=0,logger=None, filepath=None
):
    """Train the client model for one round of federated learning."""
    cfg = Configuration()
    min_metrics = 100
    if cfg.classification_type == 'multiclass':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    index_with_gradients = []
    if proximal_mu > 0.0:
        global_params = [val.detach().clone() for val in model.parameters()]
    else:
        global_params = None
    model.train()
    loss_list = []
    val_losses = []
    val_metrics = None
    for _epoch in range(epochs):
        for _batch_idx, (data, target) in enumerate(trainloader):
            # data loading
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            # forward pass
            output = model(data)

            if global_params is None:
                loss = criterion(output, target)
            else:
                # Proximal updates for FedProx
                proximal_term = 0.0
                for local_weights, global_weights in zip(
                        model.parameters(), global_params
                ):
                    proximal_term += torch.square(
                        (local_weights - global_weights).norm(2)
                    )
                loss = criterion(output, target) + (proximal_mu / 2) * proximal_term

            # backward pass
            loss.backward()

            # gradient step
            optimizer.step()

        loss_list.append(loss.detach().cpu().item())

        val_metrics = compute_accuracy(model, val_loader, device=device, calc_all=True, with_threshold_opt=True)
        epoch_loss = sum(loss_list) / len(loss_list)
        val_losses.append(val_metrics['loss'])

        if _epoch == 0:
            min_metrics = val_metrics
        if ('best_metrics' in val_metrics.keys() and min_metrics is not None and
                min_metrics['best_metrics'] > val_metrics['best_metrics']):
            min_metrics = val_metrics
        output_string = (f'Machine {net_id} epoch {_epoch} loss: {epoch_loss} val_F_score: {val_metrics["FScore"]}')
        if cfg.classification_type == "binary":
            output_string += f'val_FPR{val_metrics["fpr"]} val_recall: {val_metrics["recall"]}'
        print(output_string)
        logger.info(output_string)
    plot_train_val_loss(loss_list, val_losses, filepath=filepath,
                        comm_round=comm_round, client=net_id)
    print_confusion_matrix(model, val_loader, machine=net_id, comm_round=comm_round,filepath=filepath)
    return loss_list, min_metrics, val_metrics


def comp_accuracy(output, target, topk=(1,)):
    """Compute accuracy over the k top predictions wrt the target."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
