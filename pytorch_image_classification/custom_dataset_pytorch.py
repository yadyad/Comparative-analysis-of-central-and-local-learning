import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config_parser import config_parser
import numpy as np

from configuration import Configuration


class CustomDatasetPytorch(Dataset):
    """
    custom dataset for training pipelines
    """
    def __init__(self, df, transform=None):
        """
            init function saves image and labels from dataframe
        :param df: dataframe to fetch data and labels for training
        :param transform:
        """
        self.cfg = Configuration()
        #fetching image data from dataframe
        self.image_data = df[self.cfg.image_feature_name].values
        if self.cfg.classification_type == "binary":
            #set labels as 0 and 1
            self.labels = df[self.cfg.target_feature_name].apply(lambda x: 1 if x in self.cfg.positive_classes else 0).values
        else:
            #set label as one hot encoded vector
            self.labels = self.one_hot_encode(df[self.cfg.target_feature_number].values)
        self.transform = transform

    def one_hot_encode(self, labels):
        """
            perform one hot encoding for labels
        :param labels: the label on which one hot encoding is done
        :return: result vector after one hot encoding
        """
        return np.eye(self.cfg.Number_of_classes)[labels]

    def __len__(self):
        """
            overrides len function to return size of the dataset
        :return: size of dataset
        """
        return len(self.image_data)

    def __getitem__(self, idx):
        """
            overrides getitem function to enable augmentation and fetching images based on transforms given
        :param idx: index of data point
        :return: image and label for the corresponding index
        """
        #transform for converting image to tensor
        to_tensor_transform = transforms.ToTensor()
        image_tensor = to_tensor_transform(self.image_data[idx])
        if self.transform:
            #perform augmentation
            image_tensor = self.transform(image_tensor)
        if self.cfg.classification_type == "binary":
            #fetching label from dataset based on idx
            label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        else:
            # fetching label from dataset based on idx
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image_tensor, label


