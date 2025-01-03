import copy
import logging
from collections import OrderedDict
from functools import reduce
from typing import List, Tuple, cast, Optional, Dict, Any

import torch
from matplotlib import pyplot as plt

import utility as ut
from flwr.common import Parameters, NDArrays, NDArray, Scalar
from torch import nn
from torcheval.metrics.functional import multiclass_f1_score

import utility
from config_parser.config_parser import dict_from_conf
from configuration import Configuration
from pytorch_image_classification.argument import argument
from pytorch_image_classification.federated_learning.FedNova.fednovaclient import FedNovaClient
from pytorch_image_classification.federated_learning.FederatedServer import FederatedServer, compute_accuracy, \
    print_confusion_matrix, calculate_best_threshold_global
from utility import utility
from io import BytesIO

import numpy as np


class FedNovaServerNew(FederatedServer):
    def __init__(self, identifier=None, iteration=None):
        super().__init__(identifier, iteration)
        self.key_with_gradients = []
        ndarrays = [
            layer_param.cpu().numpy()
            for _, layer_param in self.utility.get_Model().state_dict().items()
        ]
        # initialising parameters
        self.parameters = ndarrays_to_parameters(ndarrays)
        self.initial_parameters = copy.deepcopy(self.parameters)
        self.args = argument()
        self.global_parameters = self.utility.get_Model('cpu').state_dict()
        self.global_momentum_buffer = []
        self.index_with_gradients = []
        self.logger = logging.getLogger(__name__)
        self.start_aggregation(self.args, self.get_netdata_idx_map())

    def get_init_nets(self, args):
        nets, local_model_meta_data, layer_type = self.init_nets()
        global_models, global_model_meta_data, global_layer_type = self.init_nets("global")
        global_model = global_models[0]
        return global_model, nets

    def initialize_nets(self, args, global_model, nets, round, selected):
        global_para = global_model.state_dict()
        if round == 0:
            if args.is_same_initial:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)
        else:
            for idx in selected:
                nets[idx].load_state_dict(global_para)
        return nets

    def logging(self, global_model, train_acc):
        self.log.add_to_log(None, dict_from_conf(self.cfg))
        self.log.add_to_log(None, train_acc)
        # self.log.add_to_log(None, {key: values.tolist() for key, values in global_model.state_dict().items()})
        self.log.save_log()

    def start_aggregation(self, args, net_data_idx):
        test_Metrics = None
        print("fitting server")

        print("Initializing nets")
        global_model, nets = self.get_init_nets(args)

        for round in range(args.comm_round):
            print("in comm round:" + str(round))
            selected = np.arange(args.n_parties)

            nets = self.initialize_nets(args, global_model, nets, round, selected)

            res_fit = self.local_train_net_fednova(nets, selected, global_model, args, self.get_netdata_idx_map(),
                                                   round,
                                                   test_dl=self.global_test_loader, device=self.cfg.device)

            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
            global_model.load_state_dict(self.global_parameters)
            if len(self.metrics) > 0:
                best_threshold = calculate_best_threshold_global(self.metrics)
                train_F = compute_accuracy(global_model, self.global_train_loader, device=self.cfg.device,
                                           best_threshold=best_threshold)
                test_Metrics = compute_accuracy(global_model, self.global_test_loader, device=self.cfg.device,
                                                calc_all=True, best_threshold=best_threshold)
            else:
                train_F = compute_accuracy(global_model, self.global_train_loader, device=self.cfg.device)
                test_Metrics = compute_accuracy(global_model, self.global_test_loader, device=self.cfg.device,
                                                calc_all=True)
            self.global_metrics.append(test_Metrics)
            print_confusion_matrix(global_model, self.global_test_loader,filepath=self.log.logging_path)
            print('>> Global Model Train F1Score:', train_F)
            print('>> Global Model Test Metrics:', test_Metrics)
            self.logger.info(f'>> Global Model Train F1Score: {train_F}')
            self.logger.info(f'>> Global Model Test Results:{test_Metrics}')
        ut.plot_loss_round(self.train_metrics_dict_list, message='FedNova', filepath=self.log.logging_path)
        ut.plot_score_round([entry['FScore'] for entry in self.global_metrics], message='FedNova',
                            filepath=self.log.logging_path)
        if self.cfg.logging:
            self.logging(global_model, test_Metrics)

    def local_train_net_fednova(self, nets, selected, global_model, args, net_dataidx_map, round, test_dl=None
                                , device="cpu"):
        avg_acc = 0.0
        size_dataset_per_client = [len(i.dataset) for i in self.train_parts]
        results = []
        test_acc_list = []
        test_size_list = []
        loss_each_net = []
        dict_train_metrics = {}
        # Iterating through each nets
        for net_id, net in nets.items():
            if net_id not in selected:
                continue
            # fetching the data loader for the id
            dataidxs = net_dataidx_map[net_id]
            val_df = self.val_parts[net_id]
            client = FedNovaClient()
            client.logging_path = self.log.logging_path
            print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs.dataset)))

            # setting up train loader and testloader
            train_dl_local, test_dl_local = self.train_parts[net_id], self.test_parts[net_id]
            n_epoch = self.cfg.federated_learning.local_epochs

            loss_list, res, min_metrics, val_metrics = client.train_local(net_id, net, global_model, train_dl_local, test_dl,
                                                             n_epoch, self.cfg.federated_learning.learning_rate, None,
                                                             None, None, None, device=device, comm_round=round,
                                                             data_set_len=size_dataset_per_client, val_loader=val_df,logger=self.logger)
            results.append(res)
            loss_each_net.append(loss_list)
            test_F1, conf_matrix = compute_accuracy(net, test_dl, get_confusion_matrix=True, device=device)
            test_acc_list.append(test_F1)
            test_size_list.append(len(test_dl_local.dataset))
            avg_acc += test_F1
            self.metrics.append(min_metrics)
            dict_train_metrics[net_id] = val_metrics
        self.train_metrics_dict_list.append(dict_train_metrics)
        weighted_average = ut.weighted_average_test_score(test_acc_list, test_size_list)
        avg_acc /= len(selected)
        print("avg test acc %f" % avg_acc)
        print(f"average weighted FScore {weighted_average}")
        self.find_param_with_gradient(nets)

        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.aggregate_fit(round, results)
        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, results

    def find_param_with_gradient(self, nets):
        state_dict = nets[0].state_dict(keep_vars=True)
        j = 0
        for key, val in state_dict.items():
            if val.grad is not None:
                self.index_with_gradients.append(j)
                self.key_with_gradients.append(key)
            j += 1

    def aggregate_fit(
            self,
            server_round: int,
            results,
    ):
        """Aggregate the results from the clients."""
        if not results:
            return None, {}

        # Compute tau_effective from summation of local client tau: Eqn-6: Section 4.1
        local_tau = [res[2]["tau"] for res in results]
        tau_eff = np.sum(local_tau)

        aggregate_parameters = []

        for params, _client, res, in results:
            # compute the scale by which to weight each client's gradient
            # res.metrics["local_norm"] contains total number of local update steps
            # for each client
            # res.metrics["weight"] contains the ratio of client dataset size
            # Below corresponds to Eqn-6: Section 4.1
            scale = tau_eff / float(res["local_norm"])
            scale *= float(res["weight"])

            aggregate_parameters.append((params, scale))

        # Aggregate all client parameters with a weighted average using the scale
        # calculated above
        agg_cum_gradient = aggregate(aggregate_parameters)

        # In case of Server or Hybrid Momentum, we decay the aggregated gradients
        # with a momentum factor
        self.update_server_params(agg_cum_gradient)

        return self.global_parameters, {}

    def update_server_params(self, cum_grad: NDArrays):
        """Update the global server parameters by aggregating client gradients."""

        for i, layer_cum_grad in enumerate(cum_grad):
            if self.args.gmf != 0:
                # check if it's the first round of aggregation, if so, initialize the
                # global momentum buffer

                if len(self.global_momentum_buffer) < len(cum_grad):
                    buf = layer_cum_grad / self.args.lr
                    self.global_momentum_buffer.append(buf)

                else:
                    # momentum updates using the global accumulated weights buffer
                    # for each layer of network
                    self.global_momentum_buffer[i] *= self.args.gmf
                    self.global_momentum_buffer[i] += layer_cum_grad / self.args.lr
                if self.key_with_gradients[i].split(".")[-1] == 'num_batches_tracked':
                    print(self.key_with_gradients[i])
                    continue

                self.global_parameters[self.key_with_gradients[i]] -= self.global_momentum_buffer[i] * self.args.lr


            else:
                # weight updated eqn: x_new = x_old - gradient
                # the layer_cum_grad already has all the learning rate multiple
                self.global_parameters[self.key_with_gradients[i]] -= layer_cum_grad

    def test(self, model, test_loader, round, device, parameter) -> Tuple[float, Dict[str, float]]:
        criterion = nn.CrossEntropyLoss()

        # load the model parameters
        # params_dict = zip(model.state_dict().keys(), parameter)
        # state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(parameter)

        model = model.to(device)
        model.eval()
        accuracy = []
        total_loss = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                total_loss += criterion(outputs, target).item()
                F1Score = comp_accuracy(outputs, target)
                accuracy.append(F1Score)

        total_loss /= len(test_loader)
        return total_loss, {"accuracy": sum(accuracy) / len(accuracy)}


def comp_accuracy(output, target, topk=(1,)):
    cfg = Configuration()
    tar = torch.argmax(target, dim=1)

    pred = torch.argmax(output, dim=1)
    from utility import print_confusion_matrix
    print_confusion_matrix(pred, target)
    correct = 0
    """Compute accuracy over the k top predictions wrt the target."""
    with torch.no_grad():
        correct += (pred == tar).sum().item()
        F1Score = multiclass_f1_score(pred, tar,
                                      num_classes=cfg.no_of_classes
                                      , average="weighted")
        return F1Score


def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    """Convert parameters object to NumPy ndarrays."""
    return [bytes_to_ndarray(tensor) for tensor in parameters.tensors]


def bytes_to_ndarray(tensor: bytes) -> NDArray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
    return cast(NDArray, ndarray_deserialized)


def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in results)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def ndarrays_to_parameters(ndarrays: NDArrays) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""
    tensors = [ndarray_to_bytes(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


def ndarray_to_bytes(ndarray: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
    np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()


def plot_loss(loss_each_net, round):
    # epochs = len(loss_each_net[0])
    # max_epoch = [max(len(loss_per_client), epochs) for loss_per_client in loss_each_net]
    # epochs = len(loss_each_net[0])
    #
    # # Create a range for epochs
    # epoch_range = list(range(1, max_epoch + 1))

    # Plot each model's loss
    for i, model_loss in enumerate(loss_each_net):
        plt.plot(list(range(1, len(model_loss) + 1)), model_loss, label=f'Model {i + 1}')

    # Add title and labels
    plt.title('Loss of Each Model Over Epochs at round {}'.format(round))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()  # Add a legend to differentiate models
    plt.grid(True)  # Add grid for better readability

    # Show the plot
    plt.show()
    pass


#F = FedNovaServerNew()
