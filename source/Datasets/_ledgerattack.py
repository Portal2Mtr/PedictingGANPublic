"""Pima Dataset

Managing class for the Pima Diabetes dataset.

"""


# Bitcoin dataset pytorch class
import copy
import sys
from pandas import read_csv
import pandas as pd
from sklearn.preprocessing import minmax_scale
import logging
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
import numpy.random
from torch.utils.data import Subset
import yaml
sys.path.append('./Datasets/')
from source.Datasets._dataset_base import DatasetBase


logger = logging.getLogger(__name__)


class LedgerAttackDataset(DatasetBase):

    def __init__(self, attack_type):
        super(LedgerAttackDataset, self).__init__(majority_label=0.0, minority_label=1.0, synth_label=2)
        file_config = './config/dataconfig/ledgerattack' + attack_type.lower() + 'config.yml'
        with open(file_config, "r") as read_file:
            self.config = yaml.load(read_file, Loader=yaml.FullLoader)

        self.all_data = None
        self.all_labels = None
        self.train_idxs = None
        self.sub_train_idxs = None
        self.test_idxs = None
        self.sub_test_idxs = None
        self.sub_train_data = None
        self.sub_test_data = None
        self.sub_train_inn_idxs = None
        self.sub_test_inn_idxs = None
        self.orig_data = None
        self.min_features = None
        self.max_features = None

        self.num_classes = self.config['dataconfig']['numclasses']
        self.class_array = self.config['dataconfig']['classarray']
        self.category_columns = self.config['dataconfig']['categoryfeatures']
        self.feat_labels = self.config['dataconfig']['featureLabels']
        self.attack_type = self.config['dataconfig']['attackType']
        self.cons_seed = self.config['dataconfig']['constructseed']
        self.attack_split = self.config['dataconfig']['attacksplit']
        self.load_data()
        self.num_features = len(self.all_data[0])
        self.islarge = self.config['dataconfig']['islarge']
        return

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        try:
            data = torch.reshape(Tensor(self.all_data[idx]), (self.num_features,))
        except RuntimeError:
            temp = 0
        label = torch.tensor(self.all_labels[idx], dtype=torch.double, requires_grad=True)
        data.requires_grad = True
        return data, label

    def load_data(self):
        """
        Loads the pima dataset
        Returns:

        """
        file_base = self.config['fileconfig']['dataFile'] + '_' + self.attack_type + \
                   '_split_' + \
                   str(int(self.attack_split)) \
                   + '_seed_' + str(self.cons_seed) + '.csv'

        dataframe = read_csv(file_base)
        # get the values
        values = dataframe.values
        data, labels = values[:, 2:-1], values[:, -1]  # ignore name and index only
        self.orig_data = copy.deepcopy(data)

        # Identify min/max features for rescaling
        min_features = [float('inf') for jdx in range(len(data[0]))]
        max_features = [float('-inf') for jdx in range(len(data[0]))]
        for idx, sample in enumerate(data):
            for jdx, feature in enumerate(sample):
                min_features[jdx] = min(min_features[jdx], feature)
                max_features[jdx] = max(max_features[jdx], feature)

        self.min_features = min_features
        self.max_features = max_features

        # Add noise to each sample for categorical values
        uniform_mag = 0.1
        for idx,sample in enumerate(data):
            for jdx, feature in enumerate(sample):
                if jdx in self.category_columns:
                    data[idx, jdx] = data[idx, jdx] + numpy.random.uniform(0, uniform_mag)

        # Minmax scale data
        data = minmax_scale(data, feature_range=(0, 1), copy=False)
        labels = [int(labels[i]) for i in range(len(labels))]

        # Train Test split
        data = [(i, sample) for i, sample in enumerate(data)]
        data_train, data_test, y_train, y_test = train_test_split(data, labels,
                                                                  test_size=(1 - self.config['dataconfig'][
                                                                      'trainperc']),
                                                                  random_state=0)

        self.orig_data_train_idxs = [i[0] for i in data_train]
        data_train = [i[1] for i in data_train]
        self.orig_data_test_idxs = [i[0] for i in data_test]
        data_test = [i[1] for i in data_test]

        all_data = []
        all_labels = []
        for sample, label in zip(data_train, y_train):
            all_data.append(sample)
            all_labels.append(label)

        train_idxs = [i for i in range(len(all_data))]
        train_inn_idxs = [i for i in range(len(all_labels)) if all_labels[i] == 0]

        for sample, label in zip(data_test, y_test):
            all_data.append(sample)
            all_labels.append(label)

        test_idxs = [i for i in range(train_idxs[-1], len(all_data))]
        test_inn_idxs = [i for i in range(train_idxs[-1], len(all_labels)) if all_labels[i] == 0]

        self.all_data = all_data
        self.all_labels = all_labels
        self.train_idxs = train_idxs
        self.sub_train_idxs = train_idxs
        self.test_idxs = test_idxs
        self.sub_test_idxs = test_idxs
        self.sub_train_data = Subset(self.all_data, self.sub_train_idxs)
        self.sub_test_data = Subset(self.all_data, self.sub_test_idxs)
        self.sub_train_inn_idxs = train_inn_idxs
        self.sub_test_inn_idxs = test_inn_idxs

        return

    def append_train_test_synth(self, train_data, test_data, aug_name=None):
        """
        Append synthetic data to training/testing arrays
        Args:
            train_data (): Synthetic training data
            test_data (): Synthetic testing data

        Returns:

        """

        if aug_name == 'CTGAN':
            # Use original data for testing
            self.train_idxs = self.orig_data_train_idxs
            self.sub_train_idxs = self.orig_data_train_idxs
            self.test_idxs = self.orig_data_test_idxs
            self.sub_test_idxs = self.orig_data_test_idxs
            self.all_data = self.orig_data.tolist()

        for data in train_data:
            self.all_data.append(data)
            self.all_labels.append(self.synthLabel)
            self.train_idxs.append(len(self.all_data) - 1)
            self.sub_train_idxs.append(len(self.all_data) - 1)

        for data in test_data:
            self.all_data.append(data)
            self.all_labels.append(self.synthLabel)
            self.test_idxs.append(len(self.all_data) - 1)
            self.sub_test_idxs.append(len(self.all_data) - 1)

    def rescale(self, feat_vec):
        """
        Rescales the data vector based on the min/max features of the original dataset
        Args:
            feat_vec ():

        Returns:

        """

        rescale_data = []
        for jdx, feature in enumerate(feat_vec):
            value = (self.max_features[jdx] - self.min_features[jdx]) * feature + self.min_features[jdx]
            rescale_data.append(value)

        return rescale_data

    def save_synth(self, grad_mode):
        """
        Saves the synthetic data to another project directory.
        Returns:

        """

        # Save all data to csv
        working_data = []
        for i in range(len(self.all_data)):
            # Only rescale synthetic data and append to original dataset without added noise
            work_vector = []
            if self.all_labels[i] == 2:
                work_data = self.all_data[i]
                work_data = self.rescale(work_data)
                work_vector.extend(work_data)
                work_vector.append(self.all_labels[i])
                working_data.append(work_vector)
            else:
                work_vector.extend(self.orig_data[i])
                work_vector.append(self.all_labels[i])
                working_data.append(work_vector)

        work_labels = self.feat_labels[1:-1]  # Skip 12
        work_labels.append(self.feat_labels[-1])
        working_dataframe = pd.DataFrame(data=working_data, columns=work_labels)

        file_base = self.config['fileconfig']['synthOut'] + '_' + self.config['dataconfig']['attackType'] + \
                   '_split_' + \
                    str(int(self.config['dataconfig']['attacksplit'])) \
                    + '_seed_' + str(self.config['dataconfig']['constructseed']) + '_' + grad_mode + '.csv'
        working_dataframe.to_csv(file_base, index_label='index')


