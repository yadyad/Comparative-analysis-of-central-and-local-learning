from typing import Any

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision.transforms import v2
from configuration import Configuration
from pytorch_image_classification.custom_dataset_pytorch import CustomDatasetPytorch


def random_oversampling(dataframe, target_dataframe):
    """
        used to prepare random oversampler class from imblearn
    :param dataframe: data frame to perform oversampling
    :param target_dataframe: the column of the data frame that is to be predicted
    :return:
    """
    ros = RandomOverSampler(random_state=0)
    print(dataframe.index)
    print(target_dataframe.index)
    return ros.fit_resample(dataframe, target_dataframe)[0]


class data_preparation:
    """
        class that contins function for dataset preparation
    """
    def __init__(self):
        self.augment = None
        self.cfg = Configuration()

    def data_augmentation(self):
        """
        returns the compose of different augmentation techinques.
        :return:
        """
        transform_list = []
        if self.cfg.data_augmentation.resize.on:
            transform_list.append(v2.RandomRotation(self.cfg.data_augmentation.resize.size))
        if self.cfg.data_augmentation.crop_center.on:
            transform_list.append(v2.CenterCrop(self.cfg.data_augmentation.crop_center.size))
        if self.cfg.data_augmentation.rotation.on:
            transform_list.append(v2.RandomRotation(degrees=(self.cfg.data_augmentation.rotation.min_degrees, self.cfg.data_augmentation.rotation.max_degrees)))
        if self.cfg.data_augmentation.normalize.on:
            transform_list.append(v2.Normalize(mean=self.cfg.data_augmentation.normalize.mean, std=self.cfg.data_augmentation.normalize.standard_deviation))
        if self.cfg.data_augmentation.flipping.on:
            transform_list.append(v2.RandomVerticalFlip(p=self.cfg.data_augmentation.flipping.probability))
        if self.cfg.data_augmentation.mirroring.on:
            transform_list.append(v2.RandomHorizontalFlip(p=self.cfg.data_augmentation.mirroring.probability))
        return v2.Compose(transform_list)

    def data_oversampling(self, df, sampling_type) -> list[float | Any] | Any:
        """
            sample unbalanced dataframe accorfing to sampling_type specified. acceptable types of sampling_tpye
                1)random: random oversampling
                    sample the minority class multiple times leading to equal distribution of classes
                    return the new dataframe
                2)weighted: weighted oversampling
                    find out the sample_weights based on class distribution
                    returns the dataframe after adding sample_weights column
        :param df: input dataframe
        :param sampling_type: type of sampling
        :return: return sampled dataframe
        """

        if sampling_type == 'random':
            if self.cfg.classification_type == 'multiclass':
                # return random oversampled data frame
                return random_oversampling(df, df[self.cfg.target_feature_number])
            else:
                # creating a new column in dataframe with binary labels
                df['binary_target'] = df[self.cfg.target_feature_name].apply(lambda x: 1 if x in self.cfg.positive_classes else 0)
                # using this column as target prepare random oversampler
                return random_oversampling(df, df['binary_target'])
        else:
            if self.cfg.classification_type == 'multiclass':
                #calculating weights for each datapoint
                class_counts = df[self.cfg.target_feature_number].value_counts()
                sample_weights = [1 / class_counts[i] for i in df[self.cfg.target_feature_number].values]
            else:
                df['binary_target'] = df[self.cfg.target_feature_name].apply(lambda x: 1 if x in self.cfg.positive_classes else 0)
                #calculating weights for each datapoint
                class_counts = df['binary_target'].value_counts()
                sample_weights = [1 / class_counts[i] for i in df['binary_target'].values]
                #return if weighted oversampling the weights
            return sample_weights

    def prepare_loaders_local_learning(self, train_list, test_list, val_list):
        """
            Used for preparing train, test, and validation loaders for local and federated learning pipeline input.
            also handles augmentation, sampling.
        :param train_list:  list of train dataframe
        :param test_list: list of test dataframe
        :param val_list: list of validation dataframe
        :return: train_loaders, test loaders, val_loaders after oversampling and augmentation if needed
        """
        cfg = self.cfg
        # fetches list of augmentation
        if cfg.data_augmentation.augment:
            self.augment = self.data_augmentation()
        train_loaders = []
        val_loaders = []
        test_loaders = []
        sampler = None
        #iterating on the number of clients
        for i in range(len(train_list)):
            train = train_list[i]
            test = test_list[i]
            val = val_list[i]
            if cfg.random_oversampling:
                #performs random oversampling on train dataframe
                train = self.data_oversampling(train, cfg.sampling_type)
            # prepare custom dataset from each train, test and validation data frame
            # in train dataset also applies the gathered augmentation
            train_dataset = CustomDatasetPytorch(train, self.augment)
            val_dataset = CustomDatasetPytorch(val)
            test_dataset = CustomDatasetPytorch(test)

            if cfg.weighted_sampling:
                #fetching sampling weights
                sample_weights = self.data_oversampling(train, cfg.sampling_type_weighted)
                #prepare sampler based on sampling weights
                sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train), replacement=True)
            #prepare train, test and validation data loader from dataset
            train_loaders.append(DataLoader(train_dataset, batch_size=cfg.batch_size,
                                            shuffle=cfg.shuffle_train, sampler=sampler))
            val_loaders.append(DataLoader(val_dataset, batch_size=len(val_dataset)))
            test_loaders.append(DataLoader(test_dataset, batch_size=len(test)))
        return train_loaders, test_loaders, val_loaders


    def prepare_loaders_centralised_learning(self, train_parts, test_parts, val_parts):
        """

        combine the different dataset for each client into one single dataset and perform
        oversampling augmentation and weigted sampling as specified in configuration file
        this function is used to compute the input for centralised learning pipeline
        :param train_parts:
        :param test_parts:
        :param val_parts:
        :return: return train, test, val loader
        """
        #conbining different train, test and validation dataframe into one
        train = pd.concat(train_parts)
        test = pd.concat(test_parts)
        val = pd.concat(val_parts)
        if self.cfg.data_augmentation:
            #fetching list of augmentation
            self.augment = self.data_augmentation()
        sampler = None
        #performing random oversampling
        if self.cfg.random_oversampling:
            train = self.data_oversampling(train, self.cfg.sampling_type)
        #preparing train, test and validation dataset
        train_dataset = CustomDatasetPytorch(train, self.augment)
        val_dataset = CustomDatasetPytorch(val)
        test_dataset = CustomDatasetPytorch(test)
        if self.cfg.weighted_sampling:
            #calculting sampline weights
            sample_weights = self.data_oversampling(train, self.cfg.sampling_type_weighted)
            #preparing sampler or weighted oversampling
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train), replacement=True)
        #preparing train, test and validation datalaoder
        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size,
                                  shuffle=self.cfg.shuffle_train, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=len(val))
        test_loader = DataLoader(test_dataset, batch_size=len(test))
        return train_loader, val_loader, test_loader
