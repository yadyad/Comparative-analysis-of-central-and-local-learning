import pandas as pd
import torch

from config_parser import config_parser
from configuration import Configuration


class Retrieve:
    """
        class for retrieving data that are saved after preprocessing
    """
    def __init__(self):
        #fetching configuration details
        self.cfg = Configuration()

    def retrieve_data(self):
        """
            function to retrieve data_loaders from file
        :return: lists of train, test, validation data_loader
        """
        train_parts = []
        test_parts = []
        val_parts = []
        for i in range(self.cfg.no_of_local_machines):
            train_parts.append(torch.load(f'C:/Users/yadhu/PycharmProjects/trial/image-classification/train/dl{i}.pth'))
            test_parts.append(torch.load(f'C:/Users/yadhu/PycharmProjects/trial/image-classification/test/dl{i}.pth'))
            val_parts.append(torch.load(f'C:/Users/yadhu/PycharmProjects/trial/image-classification/val/dl{i}.pth'))
        return train_parts, test_parts, val_parts

    def retrieve_dataframe(self, iteration=None):
        """
            retrieve data from the prepared data_frame.
        iteration: used to identify which data to pulled, based on which iteration is it
                    for the current training pipeline mean and standard deviation
        :return: lists of train, test, validation data_frame
        """
        train_parts = []
        test_parts = []
        val_parts = []
        for i in range(self.cfg.no_of_local_machines):
            train_parts.append(torch.load(f'{self.cfg.fetch_image}/{iteration}/traindf/df{i}.csv'))
            test_parts.append(torch.load(f'{self.cfg.fetch_image}/{iteration}/testdf/df{i}.csv'))
            val_parts.append(torch.load(f'{self.cfg.fetch_image}/{iteration}/valdf/df{i}.csv'))
        return train_parts, test_parts, val_parts

    def retrieve_dataset(self):
        """
            function to retrieve dataset saved on file
        :return: lists of train, test, validation dataset
        """
        train_parts = []
        test_parts = []
        val_parts = []
        for i in range(self.cfg.no_of_local_machines):
            train_parts.append(torch.load(f'C:/Users/yadhu/PycharmProjects/trial/image-classification/trainds/dl{i}.pt'))
            test_parts.append(torch.load(f'C:/Users/yadhu/PycharmProjects/trial/image-classification/testds/dl{i}.pt'))
            val_parts.append(torch.load(f'C:/Users/yadhu/PycharmProjects/trial/image-classification/valds/dl{i}.pt'))
        return train_parts, test_parts, val_parts

