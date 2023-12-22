"""KDD99 Dataset

Managing class for the KDD99 dataset.

"""

# Bitcoin dataset pytorch class
import sys
import random
import copy

import pandas as pd
from pandas import read_csv, concat
from sklearn.preprocessing import minmax_scale
import logging
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
import numpy.random
from torch.utils.data import Subset
import yaml
import numpy as np
sys.path.append('./Datasets/')
from source.Datasets._dataset_base import DatasetBase


logger = logging.getLogger(__name__)


class KDD99Dataset(DatasetBase):

    def __init__(self, trainPerc):
        super(KDD99Dataset, self).__init__(majority_label=0.0, minority_label=1.0, synth_label=2)
        file_config = './config/dataconfig/adultconfig.yml'
        with open(file_config, "r") as read_file:
            self.config = yaml.load(read_file, Loader=yaml.FullLoader)

        self.orig_data = None
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

        self.numClasses = self.config['dataconfig']['numclasses']
        self.class_array = self.config['dataconfig']['classarray']
        self.category_columns = self.config['dataconfig']['categoryfeatures']
        self.disc_feats = self.config['dataconfig']['todiscfeats']
        self.load_data()
        self.num_features = len(self.all_data[0])
        self.islarge = self.config['dataconfig']['islarge']
        self.downsample = self.config['dataconfig']['downsample']
        return

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        data = torch.reshape(Tensor(self.all_data[idx]), (self.num_features,))
        label = torch.tensor(self.all_labels[idx], dtype=torch.double, requires_grad=True)
        data.requires_grad = True
        return data, label

    def load_data(self):
        """
        Loads the kdd99 dataset
        Returns:

        """

        trainfile = self.config['fileconfig']['dataFile'][0]
        testfile = self.config['fileconfig']['dataFile'][1]

        # Use original train test split
        trainframe = read_csv(trainfile, header=self.config['dataconfig']['header'], sep=', ')
        testframe = read_csv(testfile, header=self.config['dataconfig']['header'], sep=', ')

        trainframe = trainframe.values
        testframe = testframe.values

        # Clean up entries
        remidxs = []
        for idx, row in enumerate(trainframe):
            for jdx in range(len(trainframe[idx, :])):
                if type(trainframe[idx,jdx]) == str:
                    trainframe[idx, jdx] = trainframe[idx, jdx].strip()

        for idx, row in enumerate(testframe):
            for jdx, val in enumerate(row):
                if type(val) == str:
                    testframe[idx, jdx] = testframe[idx, jdx].strip()

        for idx, row in enumerate(trainframe):
            for jdx, val in enumerate(row):
                if '?' == val:
                    remidxs.append(idx)
                    break

        keepidxs = [i for i in range(len(trainframe)) if i not in remidxs]
        trainframe = trainframe[keepidxs, :]

        remidxs = []

        for idx, row in enumerate(testframe):
            for jdx, val in enumerate(row):
                if '?' == val:
                    remidxs.append(idx)
                    break

        keepidxs = [i for i in range(len(testframe)) if i not in remidxs]
        testframe = testframe[keepidxs, :]

        # Remove label dots on frames
        for idx, label in enumerate(trainframe[:, -1]):

            if label == '>50K.':
                trainframe[idx] = '>50K'
            if label == '<=50K.':
                trainframe[idx] = '<=50K'

        for idx, label in enumerate(testframe[:, -1]):

            if label == '>50K.':
                testframe[idx,-1] = '>50K'
            if label == '<=50K.':
                testframe[idx,-1] = '<=50K'

        # Now binary classification dataset

        # Convert categorical to numeric

        for idx in self.disc_feats:
            _, invcol = np.unique(trainframe[:, idx], return_inverse=True)
            trainframe[:, idx] = invcol.astype(int)

        for idx in self.disc_feats:
            _, invcol = np.unique(testframe[:, idx], return_inverse=True)
            testframe[:, idx] = invcol.astype(int)

        # Shuffle data
        trainframe = pd.DataFrame(trainframe)
        trainframe = trainframe.sample(frac=1).reset_index(drop=True)
        testframe = pd.DataFrame(testframe)
        testframe = testframe.sample(frac=1).reset_index(drop=True)

        dataframe = concat([trainframe, testframe], ignore_index=True)

        # Get the values
        values = dataframe.values
        data, labels = values[:, :-1], values[:, -1]
        self.orig_data = copy.deepcopy(data)

        # Add noise to each sample for categorical values
        uniform_mag = 0.1
        for idx, sample in enumerate(data):
            for jdx, feature in enumerate(sample):
                if jdx in self.category_columns:
                    try:
                        data[idx, jdx] = data[idx, jdx] + numpy.random.uniform(0, uniform_mag)
                    except TypeError:
                        test = 0

        # Minmax scale data
        data = minmax_scale(data, feature_range=(0, 1), copy=False)
        labels = [int(labels[i]) for i in range(len(labels))]

        # Use original train/test split
        data = [(i, sample) for i, sample in enumerate(data)]

        data_train = data[0:trainframe.shape[0]][:]
        data_test = data[testframe.shape[0]:][:]
        y_train = labels[0:trainframe.shape[0]]
        y_test = labels[testframe.shape[0]:]

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
