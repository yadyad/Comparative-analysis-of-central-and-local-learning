import torchvision.models as models
import torch
from torch import nn
from typing import Optional


def check_input(model_name, size, batch_size=2):
    model_function = getattr(models, model_name)
    model = model_function(weights=None) if hasattr(model_function, 'weights') else model_function(pretrained=False)
    sample_input = torch.randn(batch_size, 3, size, size)

    try:
        model(sample_input)
        return True
    except:
        return False


def find_input_size(model_name, initial_size=32, max_size=1024, step=1, aggressive_step=16):
    for size in [224, 299] + list(range(initial_size, max_size, step)):
        if check_input(model_name, size):
            return size


def get_activation(activation_name: str):
    """
    Returns an activation layer based on the specified name
    :param activation_name: Strings defining the activation type (available: relu, prelu, softmax, sigmoid, tanh, none)
    :return: Activation layer or None
    """
    if activation_name.lower() == "none" or activation_name == None:
        activation = None
    else:
        activation = getattr(nn, activation_name)()

    return activation


def get_pooling(pooling_name: str, kernel_size: int, stride: int, one_dim_input: bool) -> nn.Module:
    """
    Returns the pooling function specified by pooling_name with the specified kernel_size and stride
    :param pooling_name: Name of the pooling function
    :param kernel_size: Pixel size of the pooling filter
    :param stride: Pixel distance of movement step of the pooling filter
    :param one_dim_input: Specifies if the input is one dimensional
    :return: Pytorch pooling function
    """
    if pooling_name.lower() == "max":
        if one_dim_input:
            return nn.MaxPool1d(kernel_size, stride)
        else:
            return nn.MaxPool2d(kernel_size, stride)
    elif pooling_name.lower() == "avg":
        if one_dim_input:
            return nn.AvgPool1d(kernel_size, stride)
        else:
            return nn.AvgPool2d(kernel_size, stride)
    else:
        raise Exception("Pooling {} not implemented".format(pooling_name))


def create_sequential_layers(in_features: int, layer_neurons: list, activations: list, dropout: list) -> list:
    modules = []
    for i in range(len(layer_neurons)):
        if i == 0:
            layer = nn.Linear(int(in_features), int(layer_neurons[i]))
        else:
            layer = nn.Linear(int(layer_neurons[i - 1]), int(layer_neurons[i]))

        modules.append(layer)

        activation = get_activation(activations[i])
        if activation is not None:
            modules.append(activation)
        if len(dropout) > 0:
            modules.append(nn.Dropout(float(dropout[i])))

    return modules


def create_sequential_model(in_features: int, layer_neurons: list, activations: list, dropout: list, device: str,
                            final_model: bool, output_dim: int) -> nn.Sequential:
    """
    Creates a sequential models with the specified structure

    Args:
        in_features: Number input features for the first layer
        layer_neurons: List containing the numbers of neurons for each layer
        activations: List of strings defining the activation after each layer (available: relu, prelu, softmax, sigmoid, tanh, none)
        dropout: List of dropout rates for each layer
        device: CPU or GPU device
    Returns:
        Sequential model with the specified structure
    """
    model = nn.Sequential(*create_sequential_layers(in_features, layer_neurons, activations, dropout)).to(device)
    if final_model:
        model.add_module("linear", nn.Linear(layer_neurons[-1], output_dim))
        # model.add_module("sigmoid", nn.Sigmoid())
    return model

def create_vision_model(model_init, pre_trained_weights: bool, layer_neurons: list, activations: list, dropout: list, base_trainable: bool,
                        device: str, output_dim: int, final_model: bool) -> models.vgg16:
    """
    Creates a transfer learning model based on the given architecture that can be initialized with the model_function.
    The final classification layer is replaced with a custom layer structure.

    Args:
        model_init: Function that initializes the base model
        pre_trained_weights: Specifies if pretrained weights should be used for transfer learning
        layer_neurons: List containing the numbers of neurons for each layer
        activations: List of strings defining the activation after each layer (available: relu, prelu, softmax, sigmoid, tanh, none)
        dropout: List of dropout rates for each layer
        base_trainable: Specifies if the weights of the base model can be trained
        device: CPU or GPU device
        output_dim: Dimension of the output data
        final_model: Specifies if the current model should be the final model or a branch model
    Returns:
        VGG16 model with custom final layers
    """
    model = model_init(pretrained=pre_trained_weights)
    try:
        model.aux_logits = False
    except:
        pass

    if not base_trainable:
        for param in model.parameters():
            param.requires_grad = False

    params = [param.shape for param in model.parameters()]
    num_ftrs = params[-2][1]

    modules = create_sequential_layers(num_ftrs, layer_neurons, activations, dropout)
    if final_model:
        layer = nn.Linear(layer_neurons[-1], output_dim)

        modules.append(layer)
        # if ml_type == "binary_classification":
        #     modules.append(nn.Sigmoid())

    modules = nn.Sequential(*modules)

    model = replace_final_layer(model, modules)

    return model.to(device)


def replace_final_layer(model, new_module):
    """
    Replace the final layer of a model with a provided module.

    Args:
    - model (nn.Module): A pretrained model.
    - new_module (nn.Module): The module to replace the final layer with.
    """
    # Determine the type and location of the final layer
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Sequential):
            last_layer_name = name

    # Get the final layer
    last_layer = getattr(model, last_layer_name)

    # Check the type of the final layer and replace it
    if isinstance(last_layer, nn.Sequential):
        # Assume the final layer we want to replace is the last layer in the Sequential block
        last_layer[-1] = new_module
    elif isinstance(last_layer, nn.Linear):
        # Replace the Linear layer directly
        setattr(model, last_layer_name, new_module)
    else:
        raise TypeError("The final layer of the model is neither Linear nor Sequential, which is unexpected.")

    return model


def build_model(use_pre_trained_weights: list, layer_neurons: list, activations: list, base_trainable: list,
                model_types: list, device: str, ml_type: str, output_dim: int, conv: list, cnn_activation:list,
                dropout: list, pooling: list, input_shapes: list, print_model_summary: bool) \
        -> [nn.Sequential, models.vgg16, models.inception_v3]:
    """
    Creates a linear or branched ML model according to the specification in the config file
    :param use_pre_trained_weights: Boolean list specifying if pre-trained weights should be used for each transfer learning branch model
    :param layer_neurons: List of lists specifying the number of neurons for each layer and branch model
    :param activations: List if lists of strings specifying the activation functions for each layer and branch model
    :param base_trainable: List of booleans specifying if the base model should be trainable for each branch model
    :param model_types: List of strings specifying the type of each branch model (available: vgg, inception, custom_cnn, fully_connected)
    :param device: cpu or cuda device
    :param ml_type: classification or binary_classification
    :param output_dim: Number of output neurons
    :param conv: List of lists specifying the convolutional layers for the custom CNN, containing channels, kernel_size, stride, padding
    :param cnn_activation: List of lists of strings specifying the activation functions for each convolutional layer and CNN branch model
    :param dropout: List of lists of floats specifying the dropout rates for each layer and CNN branch model
    :param pooling: List of lists of three elements specifying the pooling layers for the custom CNN, containing type, kernel_size, stride
    :param input_shapes: List of lists specifying the input shapes for each branch model
    :param print_model_summary: Boolean specifying if the model summary should be printed
    :return: Linear or branched model consisting of fully connected, CNN, VGG or Inception model_list
    """

    # Creates custom model structures with variable number of layers, number of neurons per layer and activations
    model_list = []
    one_dim_inputs = []
    for i in range(len(model_types)):
        final_model = i == len(model_types) - 1
        # Appends custom layers to VGG base network
        if model_types[i] == "fully_connected":
            if i > 0 and i == len(model_types) - 1:
                in_features = 0
                for j in range(len(layer_neurons) - 1):
                    in_features += layer_neurons[j][-1]
            else:
                in_features = input_shapes[i][1]

            model_list.append(create_sequential_model(
                in_features, layer_neurons[i], activations[i], dropout[i],
                torch.device(device), final_model, output_dim))
        elif model_types[i] == "custom_cnn":
            model_list.append(CNN(
                input_shapes[i], conv[i], cnn_activation[i],
                pooling[i], layer_neurons[i], activations[i],
                dropout[i], output_dim, ml_type, final_model
            ))

        else:
            model_function = getattr(models, model_types[i])
            model_list.append(create_vision_model(model_function,
                use_pre_trained_weights[i], layer_neurons[i],
                activations[i], dropout[i],
                base_trainable[i], torch.device(device), output_dim, final_model))


    # Creates a branched model from multiple branches if desired
    if len(model_list) > 1:
        model = BranchedModel(model_list[:-1], model_list[-1])
    else:
        model = model_list[0]

    # Prints a summary of the network structure
    if print_model_summary:
        print_model(model)

    return model


def print_model(model):
    """
    Prints a summary of the parameters in each layer of the given neural network structure

    Args:
        model: Torch neural network model
    """
    modules = [module for module in model.modules()]
    params = [param.shape for param in model.parameters()]

    # Prints model summary of layers and parameters
    print(modules[0])
    total_params = 0
    for i in range(1, len(modules)):
        j = 2 * i
        try:
            param = (params[j - 2][1] * params[j - 2][0]) + params[j - 1][0]
            total_params += param
            print("Layer", i, "->\t", end="")
            print("Weights:", params[j - 2][0], "x", params[j - 2][1], "\tBias: ", params[j - 1][0], "\tParameters: ", param)
        except:
            pass

    print("\nTotal Params: ", total_params)


class BranchedModel(nn.Module):
    def __init__(self, branch_models: list, combined_model):
        """
        Creates a branched model that consists of multiple branch models that accept individual inputs, concatenates
        their outputs and uses a final model to further process these outputs

        Args:
            branch_models: List of models that accept an input each and create a concatenated output used in the combined_model
            combined_model: Neural network that further processes the concatenated output to the final prediction
        """
        super(BranchedModel, self).__init__()
        for i in range(len(branch_models)):
            self.__setattr__("branch_model" + str(i), branch_models[i])
        self.combined_model = combined_model

    def forward(self, inputs: list):
        """
        Passes the inputs through the branched network

        Args:
            inputs: List that contains the inputs for the branch models

        Returns:
            Prediction of the model for the given input
        """
        branch_outputs = []

        for i in range(len(inputs)):
            branch_outputs.append(self.__getattr__("branch_model" + str(i))(inputs[i]))

        # Flatten the output of each branch model and concatenate them
        branch_outputs = [torch.squeeze(output, dim=1) if len(output.shape) > 2 else output for output in
                          branch_outputs]

        branch_output = torch.cat(branch_outputs, dim=1)

        model_output = self.combined_model(branch_output)
        return model_output


class CNN(nn.Module):

    def __init__(
            self, input_dim: list, conv: list, cnn_activation: list, pooling: list, layer_neurons: list, activations: list,
            dropout: list, output_dim: int, ml_type, final_model: bool = False):
        """
        Initializes the CNN with the specified architecture
        :param input_dim: Dimension of the input data
        :param conv: List if lists of integers, where each sublist specifies one convolutional layer, defined by #Channels, kernel_size, stride, padding
        :param cnn_activation: List of names of activation functions for each convolutional layer
        :param pooling: List of lists of three elements. Each sublist specifies one pooling layer, defined by type, kernel_size, stride
        :param layer_neurons: List of integers specifying the number of neurons for each fully connected layer
        :param activations: List of names of activation functions for each fully connected layer
        :param dropout: List of dropout rates for each fully connected layer
        :param output_dim: Dimension of the output data
        :param final_model: Specifies if the current model should be the final model or a branch model
        """

        super(CNN, self).__init__()
        self.layers = nn.ModuleList()

        one_dim_input = len(input_dim) == 2

        for i in range(len(conv)):
            if i == 0:
                if one_dim_input:
                    self.layers.append(nn.Conv1d(input_dim[0], conv[i][0], conv[i][1], conv[i][2], conv[i][3]))
                else:
                    self.layers.append(nn.Conv2d(input_dim[0], conv[i][0], conv[i][1], conv[i][2], conv[i][3]))
            else:
                if one_dim_input:
                    self.layers.append(nn.Conv1d(conv[i-1][0], conv[i][0], conv[i][1], conv[i][2], conv[i][3]))
                else:
                    self.layers.append(nn.Conv2d(conv[i-1][0], conv[i][0], conv[i][1], conv[i][2], conv[i][3]))
            activation = get_activation(cnn_activation[i])
            if activation is not None:
                self.layers.append(activation)
            self.layers.append(get_pooling(pooling[i][0], pooling[i][1], pooling[i][2], one_dim_input))
        self.layers.append(nn.Flatten())

        image_size = input_dim[1]
        for i in range(len(conv)):
            image_size = int((image_size - conv[i][1] + 2 * conv[i][3]) / conv[i][2] + 1)
            image_size = int((image_size - pooling[i][1]) / pooling[i][2] + 1)


        for i in range(len(layer_neurons)):
            if i == 0:
                if one_dim_input:
                    layer = nn.Linear(image_size * conv[-1][0], layer_neurons[i])
                else:
                    layer = nn.Linear(image_size * image_size * conv[-1][0], layer_neurons[i])
            else:
                layer = nn.Linear(layer_neurons[i - 1], layer_neurons[i])

            self.layers.append(layer)

            activation = get_activation(activations[i])
            if activation != None:
                self.layers.append(activation)
            if len(dropout) > i:
                self.layers.append(nn.Dropout(dropout[i]))

        if final_model:

            if len(layer_neurons) == 0:
                if one_dim_input:
                    layer = nn.Linear(image_size * conv[-1][0], output_dim)
                else:
                    layer = nn.Linear(image_size * image_size * conv[-1][0], output_dim)
            else:
                layer = nn.Linear(layer_neurons[-1], output_dim)

            self.layers.append(layer)

            # if ml_type == "binary_classification":
            #     self.layers.append(nn.Sigmoid())



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network
        :param x: Input data
        :return: Output of the network
        """
        for layer in self.layers:
            x = layer(x)
        return x

import torch.nn.functional as F


class TestCNN(nn.Module):
    def __init__(self):
        super(TestCNN, self).__init__()
        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional Block
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        # Third Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # Fourth Convolutional Block
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        # Fifth Convolutional Block
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * 7 * 7, 16)
        self.fc2 = nn.Linear(16, 1)  # Output 1 neuron for binary classification
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = self.pool(x)

        x = torch.flatten(x, 1)  # Flatten the output

        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = torch.sigmoid(x)
        return x