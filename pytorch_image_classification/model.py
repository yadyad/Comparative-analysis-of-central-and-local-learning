import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_f1_score
from torchvision.models import resnet50, vgg16
import torchvision.models as models
from configuration import Configuration


def get_activation(activation, final_layer=False, from_nn=False):
    """
        function to fetch the activation function for each layer
    :param activation: binary, relu, softmax, elu, tanh different activation function
    :param final_layer: true if final layer activation
    :param from_nn: which library to be used torch.nn or torch.nn.functional
    :return:
    """
    cfg = Configuration()
    if from_nn:
        if final_layer:
            if cfg.classification_type == "binary":
                return nn.Sigmoid()
        if activation == 'relu':
            return nn.ReLU()
        if activation == 'softmax':
            return nn.Softmax()
        if activation == 'elu':
            return nn.ELU()

        if activation == 'tanh':
            return nn.Tanh()
    else:
        if final_layer:
            if cfg.classification_type == "binary":
                return F.sigmoid
        if activation == 'relu':
            return F.relu
        if activation == 'softmax':
            return F.softmax
        if activation == 'elu':
            return F.elu

        if activation == 'tanh':
            return F.tanh


class Net(nn.Module):
    """
        class for implementing training model
        basic CNN is implemented here with 3 convolution layer, pooling and fully connected
        layer at the end
    """

    def __init__(self, num_classes: int) -> None:
        """
        implementing neural network structure
        :param num_classes: number of distinct labels
        """
        super(Net, self).__init__()
        self.cfg = Configuration()

        basic_cfg = self.cfg.local_learning.basic_convolution
        #setup the first convolution layer according to the configuration file
        #learning configuration
        self.conv1 = nn.Conv2d(basic_cfg.in_channel,
                               basic_cfg.conv_1.no_of_filter,
                               basic_cfg.conv_1.kernel_size,
                               basic_cfg.conv_1.strides)
        self.optim1 = get_activation(basic_cfg.conv_1.activation)

        self.pool = nn.MaxPool2d(basic_cfg.pool.kernel_size,
                                 basic_cfg.pool.strides)

        self.conv2 = nn.Conv2d(basic_cfg.conv_1.no_of_filter,
                               basic_cfg.conv_2.no_of_filter,
                               basic_cfg.conv_2.kernel_size,
                               basic_cfg.conv_2.strides)
        self.optim2 = get_activation(basic_cfg.conv_2.activation)

        self.conv3 = nn.Conv2d(basic_cfg.conv_2.no_of_filter,
                               basic_cfg.conv_3.no_of_filter,
                               basic_cfg.conv_3.kernel_size,
                               basic_cfg.conv_3.strides)
        self.optim3 = get_activation(basic_cfg.conv_3.activation)

        self.fc1 = nn.Linear(basic_cfg.dense_1.in_features,
                             basic_cfg.dense_1.out_features)
        self.optim_fc1 = get_activation(basic_cfg.dense_1.activation)

        self.fc2 = nn.Linear(basic_cfg.dense_2.in_features,
                             basic_cfg.dense_2.out_features)
        self.optim_fc2 = get_activation(basic_cfg.dense_2.activation)

        self.fc3 = nn.Linear(basic_cfg.dense_3.in_features,
                             num_classes)
        self.optim_fc3 = get_activation(basic_cfg.dense_3.activation, final_layer=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        structure of forward propogation
        :param x: input value
        :return: output value
        """
        x = self.pool(self.optim1(self.conv1(x)))
        x = self.pool(self.optim2(self.conv2(x)))
        x = self.pool(self.optim3(self.conv3(x)))
        x = x.view(-1, self.cfg.local_learning.basic_convolution.dense_1.in_features)
        x = self.optim_fc1(self.fc1(x))
        x = self.optim_fc2(self.fc2(x))
        x = self.optim_fc3(self.fc3(x), dim=0)
        return x


class Net2(nn.Module):
    def __init__(self, number_of_classes: int):
        """
            class for implementing training model
            improved CNN is implemented here with 3 convolution layer, pooling and fully connected
            layer at the end
        """
        super(Net2, self).__init__()
        self.cfg = Configuration()
        cfg = self.cfg.local_learning.improved_convolution
        self.conv1 = nn.Conv2d(cfg.in_channel, cfg.conv_1.no_of_filter,
                               cfg.conv_1.kernel_size, padding=cfg.conv_1.padding)
        self.optim1 = get_activation(cfg.conv_1.activation)
        self.bn1 = nn.BatchNorm2d(cfg.conv_1.no_of_filter)

        self.conv2 = nn.Conv2d(cfg.conv_1.no_of_filter, cfg.conv_2.no_of_filter,
                               cfg.conv_2.kernel_size, padding=cfg.conv_2.padding)
        self.optim2 = get_activation(cfg.conv_2.activation)
        self.bn2 = nn.BatchNorm2d(cfg.conv_2.no_of_filter)

        self.pool1 = nn.MaxPool2d(cfg.pool_1.kernel_size,
                                  cfg.pool_1.strides)

        self.conv3 = nn.Conv2d(cfg.conv_2.no_of_filter, cfg.conv_3.no_of_filter,
                               cfg.conv_3.kernel_size, padding=cfg.conv_3.padding)
        self.optim3 = get_activation(cfg.conv_3.activation)
        self.bn3 = nn.BatchNorm2d(cfg.conv_3.no_of_filter)

        self.conv4 = nn.Conv2d(cfg.conv_3.no_of_filter, cfg.conv_4.no_of_filter,
                               cfg.conv_4.kernel_size, padding=cfg.conv_4.padding)
        self.optim4 = get_activation(cfg.conv_4.activation)
        self.bn4 = nn.BatchNorm2d(cfg.conv_4.no_of_filter)

        self.pool2 = nn.MaxPool2d(cfg.pool_2.kernel_size,
                                  cfg.pool_2.strides)

        self.conv5 = nn.Conv2d(cfg.conv_4.no_of_filter, cfg.conv_5.no_of_filter,
                               cfg.conv_5.kernel_size, padding=cfg.conv_5.padding)
        self.optim5 = get_activation(cfg.conv_5.activation)
        self.bn5 = nn.BatchNorm2d(cfg.conv_5.no_of_filter)

        self.conv6 = nn.Conv2d(cfg.conv_5.no_of_filter, cfg.conv_6.no_of_filter,
                               cfg.conv_6.kernel_size, padding=cfg.conv_6.padding)
        self.optim6 = get_activation(cfg.conv_6.activation)
        self.bn6 = nn.BatchNorm2d(cfg.conv_6.no_of_filter)

        self.pool3 = nn.MaxPool2d(cfg.pool_3.kernel_size)

        self.dropout1 = nn.Dropout(cfg.dropout_1.percent)

        self.fc1 = nn.Linear(cfg.dense_1.no_of_neurons, cfg.dense_1.out_features)
        self.optim_fc1 = get_activation(cfg.dense_1.activation)

        self.dropout2 = nn.Dropout(cfg.dropout_2.percent)

        self.fc2 = nn.Linear(cfg.dense_1.out_features, number_of_classes)
        self.optim_fc2 = get_activation(cfg.dense_2.activation, final_layer=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
                structure of forward propagation
                :param x: input value
                :return: output value
        """
        x = self.bn1(self.optim1(self.conv1(x)))
        x = self.bn2(self.optim2(self.conv2(x)))
        x = self.pool1(x)
        x = self.bn3(self.optim3(self.conv3(x)))
        x = self.bn4(self.optim4(self.conv4(x)))
        x = self.pool2(x)
        x = self.bn5(self.optim5(self.conv5(x)))
        x = self.bn6(self.optim6(self.conv6(x)))
        x = self.pool3(x)
        x = x.view(-1, self.cfg.local_learning.improved_convolution.dense_1.in_features)
        x = self.dropout1(x)
        x = self.optim_fc1(self.fc1(x))
        x = self.dropout2(x)
        x = self.optim_fc2(self.fc2(x), dim=0)
        # x = self.fc2(x)
        return x


def prepare_resnet_model(number_of_classes):
    """
        Prepare pretrained resnet model for training
    :param number_of_classes: number of distinct labels
    :return: resnet model
    """

    cfg = Configuration()
    #fetching pretrained resnet model
    model = resnet50(pretrained=True)
    #changing the final fully connected layer
    num_ftrs = model.fc.in_features

    if cfg.classification_type == 'binary':
        #the last layer neuron if binary is 1
        model.fc = nn.Linear(num_ftrs, number_of_classes)
    else:
        #the last layer neuron if multiclass will be the number of distinct labels
        model.fc = nn.Linear(num_ftrs, number_of_classes)  # CIFAR-10 has 10 classes

    #model to training device.
    if torch.cuda.is_available() and cfg.device == 'cuda':
        model = model.to('cuda')
    else:
        model = model.to('cpu')
    return model


def prepare_VGG_model(number_of_classes):
    """
        function to prepare VGG model
    :param number_of_classes: number of distinct labels
    :return: vgg model
    """
    cfg = Configuration()
    model = vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=number_of_classes)
    return model
