import shutil
import time

import numpy as np
from typing import Tuple, Any, List
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import cv2
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision.transforms import v2
import os

from configuration import Configuration
from pytorch_image_classification.custom_dataset_pytorch import CustomDatasetPytorch


def check_same_value(df):
    not_same = df[df['errorClassText'] != df['errorClassCombined']]
    print(not_same['imageIndex'])


class DataPreprocessing:
    def extract_data(self):
        """
        Reads and extract images from the location specified in the excel which
        is present in the variable "base_path" of configuration file. Select required features
        from the dataset and create a new column in dataframe for filepath to images.

        :return: dataframe
        """
        data_set_paths = self.cfg.image.base_path
        df_list = []
        for company in data_set_paths:
            start = time.time()
            df = pd.read_excel(company.excel_path)
            # df_grouped = df.groupby(self.cfg.splitting.splitting_feature_name)
            # df = df_grouped.filter(lambda x: len(x) >= self.cfg.splitting.min_samples_per_split)
            df = self.filter_rows(df)
            df = self.remove_rows(df)
            df['errorClassText'] = df['errorClassText'].replace('PseudoFehler', 'good')
            check_same_value(df)
            df = df.filter(self.cfg.image.column_selected)
            df.imageIndex = df.imageIndex.astype(str)
            df[self.cfg.target_feature_name] = df[self.cfg.target_feature_name].astype(str)
            df['file_path'] = company.path + df[self.cfg.image_index_feature] + '.png'
            df[self.cfg.dataset_feature_name] = company.owner
            print("--- %s seconds ---" % (time.time() - start))
            print("length of dataframe", len(df))
            df_list.append(df)
        df_merged = pd.concat(df_list, ignore_index=True, sort=False)
        print(df_merged.columns)
        return df_merged

    def input_image(self, file_path):
        """
        read image from file_path specified into cv2 image.
        :param file_path:
        :return: the scanned image
        """
        img = cv2.imread(str(file_path))
        return img

    def rename_background_images(self, row):
        """
            function to move both red and blue background images and non specific images to class "Background Images".
            Also applies medianBlur to each image.
        :param row:
        :return:
        """
        image = row[self.cfg.image.image_feature_name]
        classname = row[self.cfg.target_feature_name]
        mean, std_dev = cv2.meanStdDev(image)
        blurred_image = cv2.medianBlur(image, 5)
        histb = np.array(cv2.calcHist([blurred_image], [0], None, [256], [0, 256])).reshape(256)
        histg = np.array(cv2.calcHist([blurred_image], [1], None, [256], [0, 256])).reshape(256)
        if np.max(histb) > 1500 and np.argmax(histb) > 250 and np.max(histg) < 300 and std_dev[0] < 30:
            classname = self.cfg.background_image_class_name
        else:
            mean, std_dev = cv2.meanStdDev(image)
            if std_dev[0] < 20 and std_dev[1] < 20 and std_dev[2] < 20:
                classname = self.cfg.background_image_class_name
        return classname

    def create_dictionary(self):
        """
            creates dictionaries with dictionaries specified in the configuration file
        :return:
        """
        import yaml
        with open(self.configuration_file, 'r') as file:
            configuration = yaml.safe_load(file)
        class_name_dict = configuration['class_names_dict']
        class_number_dict = configuration['class_number_dict']
        return [class_number_dict, class_name_dict]

    def create_numeric_label(self, classname):
        """
            fetches  numeric label given classname to map for one hot encoded values.

        :param classname:
        :return:
        """
        return self.dict_list[0].get(classname)

    def getDataFrame(self):
        return self.df

    def __init__(self,iter = None):
        """
         init function that saves the final dataframe after all the preprocessing steps are done to raw input.
         the preprocessing steps include
            extracting data from the specified location
            sorting red and blue background images to new class
            creating classNumber feature
            seperating data based on the configuration details specified
            spliting to train, test and validation for each of the different client specified in the configuration.
        """
        self.augment = None
        self.configuration_file = "C:/Users/yadhu/Desktop/thesis_code/image-classification-master/configuration.yaml"
        self.cfg = Configuration(self.configuration_file)
        self.df = self.extract_data()
        self.df['image'] = self.df['file_path'].apply(self.input_image)
        self.df['errorClassCombined'] = self.df.apply(self.rename_background_images, axis=1)
        self.dict_list = self.create_dictionary()
        self.df['classNumber'] = self.df['errorClassCombined'].apply(self.create_numeric_label).astype(int)
        self.df_parts = self.data_seperation(self.df, self.cfg.separation_type)
        self.train_parts, self.test_parts, self.val_parts = self.train_test_split_dataframe(self.df_parts,
                                                                                            self.cfg.splitting.split_type)
        self.save_df(iter)

    # self.train_parts, self.test_parts, self.val_parts = self.prepare_loaders(self.train_parts, self.test_parts, self.val_parts)

    def save_df(self,iter=None):
        """
        save the train, test, val data parts dataframe into different folder.
        :return:
        """
        if os.path.exists(f'{self.cfg.fetch_image}/{iter}/traindf'):
            shutil.rmtree(f'{self.cfg.fetch_image}/{iter}/traindf')
        if os.path.exists(f'{self.cfg.fetch_image}/{iter}/testdf'):
            shutil.rmtree(f'{self.cfg.fetch_image}/{iter}/testdf')
        if os.path.exists(f'{self.cfg.fetch_image}/{iter}/valdf'):
            shutil.rmtree(f'{self.cfg.fetch_image}/{iter}/valdf')

        if not os.path.exists(f'{self.cfg.fetch_image}/{iter}/traindf'):
            os.makedirs(f'{self.cfg.fetch_image}/{iter}/traindf')
        if not os.path.exists(f'{self.cfg.fetch_image}/{iter}/testdf'):
            os.makedirs(f'{self.cfg.fetch_image}/{iter}/testdf')
        if not os.path.exists(f'{self.cfg.fetch_image}/{iter}/valdf'):
            os.makedirs(f'{self.cfg.fetch_image}/{iter}/valdf')
        for i in range(len(self.train_parts)):
            torch.save(self.train_parts[i], f'{self.cfg.fetch_image}/{iter}/traindf/df{i}.csv')
            torch.save(self.test_parts[i], f'{self.cfg.fetch_image}/{iter}/testdf/df{i}.csv')
            torch.save(self.val_parts[i], f'{self.cfg.fetch_image}/{iter}/valdf/df{i}.csv')

    def save_to_file(self):
        """
            used for saving dataloaders into different files
        :return:
        """
        if not os.path.exists('train'):
            os.makedirs('train')
        if not os.path.exists('test'):
            os.makedirs('test')
        if not os.path.exists('val'):
            os.makedirs('val')
        for i in range(len(self.train_parts)):
            torch.save(self.train_parts[i], f'train/dl{i}.pth')
            torch.save(self.test_parts[i], f'test/dl{i}.pth')
            torch.save(self.train_parts[i], f'val/dl{i}.pth')

    def save_dataset(self, train, test, val):
        """
        used for saving dataset to different files.
        :param train:
        :param test:
        :param val:
        :return:
        """
        if not os.path.exists('trainds'):
            os.makedirs('trainds')
        if not os.path.exists('testds'):
            os.makedirs('testds')
        if not os.path.exists('valds'):
            os.makedirs('valds')
        for i in range(len(self.train_parts)):
            torch.save(self.train_parts[i], f'trainds/dl{i}.pt')
            torch.save(self.test_parts[i], f'testds/dl{i}.pt')
            torch.save(self.train_parts[i], f'valds/dl{i}.pt')

    def data_augmentation(self):
        """
        returns the compose of different augmentation techinques.
        :return:
        """
        transform_list = []
        if self.cfg.data_augmentation.resize.on:
            transform_list.append(v2.RandomRotation(self.cfg.data_augmentation.resize.size))
        if self.cfg.data_augmentation.rotation.on:
            transform_list.append(v2.RandomRotation(degrees=(
                self.cfg.data_augmentation.rotation.min_degrees, self.cfg.data_augmentation.rotation.max_degrees)))
        if self.cfg.data_augmentation.normalize.on:
            transform_list.append(v2.Normalize(mean=self.cfg.data_augmentation.normalize.mean,
                                               std=self.cfg.data_augmentation.normalize.standard_deviation))
        if self.cfg.data_augmentation.flipping.on:
            transform_list.append(v2.RandomVerticalFlip(p=self.cfg.data_augmentation.flipping.probability))
        if self.cfg.data_augmentation.mirroring.on:
            transform_list.append(v2.RandomHorizontalFlip(p=self.cfg.data_augmentation.mirroring.probability))
        return v2.Compose(transform_list)

    def prepare_loaders(self, train_list, test_list, val_list):

        """

            Used for preparing train, test, and validation loaders for macchine learning model input.
            also handles augmentation, sampling.
        :param train_list:
        :param test_list:
        :param val_list:
        :return: train_loaders, test loaders, val_loaders after oversampling and augmentation if needed
        """
        sample_weights = None
        cfg = self.cfg
        if cfg.data_augmentation.augment:
            self.augment = self.data_augmentation()
        train_loaders = []
        val_loaders = []
        test_loaders = []
        sampler = None
        for i in range(len(train_list)):
            train = train_list[i]
            test = test_list[i]
            val = val_list[i]
            if cfg.random_oversampling:
                train = self.data_oversampling(train, cfg.sampling_type)
            # train, val = train_test_split(df, test_size=cfg.local_learning.validation_size)
            train_dataset = CustomDatasetPytorch(train, self.augment)
            val_dataset = CustomDatasetPytorch(val)
            test_dataset = CustomDatasetPytorch(test)
            # self.save_dataset(train_dataset,test_dataset,val_dataset)
            if cfg.weighted_sampling:
                sample_weights = self.data_oversampling(train, cfg.sampling_type_weighted)
                sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train), replacement=True)
            train_loaders.append(DataLoader(train_dataset, batch_size=cfg.batch_size,
                                            shuffle=cfg.shuffle_train, sampler=sampler))
            val_loaders.append(DataLoader(val_dataset, batch_size=cfg.batch_size))
            test_loaders.append(DataLoader(test_dataset, batch_size=len(test)))
        return train_loaders, test_loaders, val_loaders

    def train_test_split_dataframe(self, df_list: List[pd.DataFrame], split_type: str = 'random') -> Tuple[
        List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
        """
                function for returning train dataframe and test dataframe based on the splitting criteria
                available splitting criteria :
                    random
                    dataset
                    equal

        :param df_list: list of dataframe to which spliting is done
        :param split_type: default random: specifies the split type
        :return: list of train dataframe, test dataframe, validation dataframe
        """
        train_list = []
        test_list = []
        val_list = []
        for df in df_list:
            train, test, val = None, None, None
            if split_type == "random":
                train, temp = train_test_split(df, test_size=self.cfg.splitting.train_val_test_split_ratio * 2)
                test, val = train_test_split(temp, test_size=0.5)
            elif split_type == "dataset":
                random_value = df[self.cfg.dataset_feature_name].sample(n=1).iloc[0]
                filtered_values = df[df[self.cfg.dataset_feature_name] != random_value][self.cfg.dataset_feature_name]
                random_value_validation = np.random.choice(filtered_values)
                test = df.loc[df[self.cfg.dataset_feature_name] == random_value]
                val = df.loc[df[self.cfg.dataset_feature_name] == random_value_validation]
                train = df.loc[df[self.cfg.dataset_feature_name] != random_value &
                               df[self.cfg.dataset_feature_name] != random_value_validation]

            else:
                train, temp = train_test_split(df, test_size=self.cfg.splitting.train_val_test_split_ratio,
                                               stratify=df[self.cfg.target_feature_name],
                                               random_state=self.cfg.splitting.random_state)
                test, val = train_test_split(temp, test_size=0.5,
                                             stratify=temp[self.cfg.target_feature_name],
                                             random_state=self.cfg.splitting.random_state)
            train_list.append(train)
            test_list.append(test)
            val_list.append(val)
        return train_list, test_list, val_list

    def random_oversampling(self, dataframe, targetDataframe):
        ros = RandomOverSampler(random_state=0)
        print(dataframe.index)
        print(targetDataframe.index)
        return ros.fit_resample(dataframe, targetDataframe)[0]

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
            return self.random_oversampling(df, df[self.cfg.target_feature_number])
        else:
            class_counts = df[self.cfg.target_feature_number].value_counts()
            sample_weights = [1 / class_counts[i] for i in df[self.cfg.target_feature_number].values]
            return sample_weights

    def data_seperation(self, train_df, separation_type) -> list[Any]:
        """
        seperated data to different clients for simulation
        :param train_df:
        :param separation_type: describes how the data splitting between clients should happen
            accepted keywords
                1)random
                2)feature: seperates based on feature
        :return: train, val for each client as a list
        """
        # removing data which have less than given
        df_grouped = train_df.groupby(self.cfg.splitting.splitting_feature_name)
        df = df_grouped.filter(lambda x: len(x) >= self.cfg.splitting.min_samples_per_split)
        if separation_type == 'random':
            return split_into_parts_at_random(train_df, self.cfg.no_of_local_machines)
        else:
            df_parts = [x[1] for x in df.groupby(self.cfg.splitting.splitting_feature_name)]
            df_parts_combined = self.combine_dataframes(df_parts, self.cfg.no_of_local_machines)
            print('helle')
            # else:
            #     df_grouped = train_df.groupby(self.cfg.splitting.splitting_feature_name)
            #     df = df_grouped.filter(lambda x: len(x) >= self.cfg.splitting.min_samples_per_split)
            #     df_parts = [x[1] for x in df.groupby(self.cfg.splitting.splitting_feature_name)]
            #     self.cfg.no_of_local_machines = len(df_parts)
            #     self.update_no_clients_in_conf(self.cfg.no_of_local_machines)
            return df_parts_combined

    def combine_dataframes(self, dataframes: List[pd.DataFrame], new_size: int) -> List[pd.DataFrame]:
        """
        used to create given number of client data from list of dataframe containing data groupedby seperation feature
        :param dataframes: dataframes grouped based on feature
        :param new_size: New_size to which the list should be reduced by combining
        :return: list of resulting dataframe after combining
        """
        if len(dataframes) < new_size:
            raise ValueError("The number of clients should not be less than the distinct number of features that"
                             " separation is based on")
        new_list = [[] for i in range(new_size)]
        counter = 0
        for df in dataframes:
            new_list[counter].append(df)
            counter += 1
            if counter % new_size == 0:
                counter = 0
        df_parts = []
        for i, val in enumerate(new_list):
            df_parts.append(pd.concat(val))

        return df_parts

    def filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Used to filer out rows based on keys and columns given in filter_row
        :param df:
        :return:
        """
        for key in self.cfg.image.filter_rows:
            df = df[df[key.column_name] == key.key_to_filter]
        return df

    def remove_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Used to remove rows based on keys and columns given  in remove_rows
        :param df:
        :return:
        """
        for key in self.cfg.image.remove_rows:
            df = df[df[key.column_name] != key.key_to_remove]
        return df

    def update_no_clients_in_conf(self, no_of_clients):
        from ruamel.yaml import YAML

        yaml = YAML()
        yaml.preserve_quotes = True
        with open(self.configuration_file, 'r') as file:
            data = yaml.load(file)

        # Modify the data
        data['no_of_local_machines'] = no_of_clients

        # Write the updated data back to the YAML file
        with open(self.configuration_file, 'w') as file:
            yaml.dump(data, file)


def split_into_parts_at_random(train_df, num_parts=20):
    """
        splits train_df into different parts of equal size at picking at random
    :param train_df:
    :param num_parts: no of parts to split.
    :return: List of dataframes.
    """
    np.random.seed(42)  # For reproducibility
    train_indices = np.arange(len(train_df))
    np.random.shuffle(train_indices)

    part_size = len(train_df) // num_parts
    train_parts = []

    for i in range(num_parts):
        start_idx = i * part_size
        end_idx = (i + 1) * part_size

        if i == num_parts - 1:
            end_idx = len(train_df)

        train_part_indices = train_indices[start_idx:end_idx]
        train = train_df.iloc[train_part_indices]
        train_parts.append(train)
    return train_parts


D = DataPreprocessing()
