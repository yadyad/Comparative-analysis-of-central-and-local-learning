import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms

from configuration import Configuration
from torch.utils.data import random_split, DataLoader, Subset

cfg = Configuration()


def prepare_CIFAR10_dataloader(split=True):
    cfg = Configuration()
    if cfg.classification_type == 'multiclass':
        cifar_train = CIFAR10OneHot(root='./data', train=True, download=True,
                                    transform=transforms.ToTensor())
        cifar_test = CIFAR10OneHot(root='./data', train=False, download=True,
                                   transform=transforms.ToTensor())
    else:
        # cifar_train = CIFAR10biinary(root='./data', train=True, download=True,
        #                             transform=transforms.ToTensor())
        # cifar_test = CIFAR10Obinary(root='./data', train=False, download=True,
        #                            transform=transforms.ToTensor())
        trainset = torchvision.datasets.CIFAR10(root='D:/Yadhu/', train=True, download=True,
                                                transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root='D:/Yadhu/', train=False, download=True,
                                               transform=transforms.ToTensor())
        cifar_train = get_binary_dataset(trainset, 3, 5)
        cifar_test = get_binary_dataset(testset, 3, 5)
    cifar_train, cifar_val = train_test_split(cifar_train, test_size=0.2, stratify=cifar_train.targets)

    train_loaders = prepare_loaders_cifar_10(cifar_train, cfg.no_of_local_machines, 128, split)
    val_loaders = prepare_loaders_cifar_10(cifar_val, cfg.no_of_local_machines, len(cifar_val),
                                           split)
    test_loaders = prepare_loaders_cifar_10(cifar_test, cfg.no_of_local_machines, len(cifar_test),
                                            split)
    global_train_loader = prepare_loaders_cifar_10(cifar_train, cfg.no_of_local_machines, 128, False)
    global_test_loader = prepare_loaders_cifar_10(cifar_test, cfg.no_of_local_machines, len(cifar_test),
                                                  False)
    global_val_loader = prepare_loaders_cifar_10(cifar_val, cfg.no_of_local_machines, len(cifar_val),
                                                 False)
    return train_loaders, test_loaders, val_loaders, global_train_loader, global_test_loader, global_val_loader


def prepare_loaders_cifar_10(dataset, n, batch_size, split):
    # Calculate the size of each part

    def splitted_dataset():
        dataset_size = len(dataset)
        part_size = dataset_size // n
        remaining = dataset_size % n

        # Create lengths for each part
        lengths = [part_size] * n
        # Distribute any remaining items to the first few parts
        for i in range(remaining):
            lengths[i] += 1

        # Split the dataset
        datasets_parts = random_split(dataset, lengths)
        data_loaders = [DataLoader(part, batch_size=batch_size) for part in datasets_parts]
        return data_loaders

    def unsplit_dataset():
        data_loaders = DataLoader(dataset, batch_size=batch_size)
        return data_loaders

    if split:
        return splitted_dataset()
    else:
        return unsplit_dataset()


# Function to filter out only cats and dogs
def get_binary_dataset(dataset, class1, class2):
    indices = [i for i, label in enumerate(dataset.targets) if label == class1 or label == class2]
    dataset.targets = [0 if dataset.targets[i] == class1 else 1 for i in indices]  # 0 for class1, 1 for class2
    dataset.targets = torch.tensor(dataset.targets, dtype=torch.float32).unsqueeze(1)
    dataset.data = dataset.data[indices]
    return dataset


import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader


class CIFAR10OneHot(datasets.CIFAR10):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        # One-hot encode the target
        target_one_hot = F.one_hot(torch.tensor(target), 10)
        return image, target_one_hot.float()


class CIFAR10Obinary(datasets.CIFAR10):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        # binary encode the target
        target_binary = 1 if target == 3 else 0
        return image, torch.tensor(target_binary, dtype=torch.float32).unsqueeze(0)