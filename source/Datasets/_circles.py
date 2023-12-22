"""Circles Dataset

Managing class for the Overlapping Circles Dataset.

"""


# Bitcoin dataset pytorch class

from pandas import read_csv
from sklearn.preprocessing import minmax_scale
import logging
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
from sklearn.datasets import make_circles

from source.Datasets._dataset_base import DatasetBase
from torch.utils.data import Subset
import yaml
import math

logger = logging.getLogger(__name__)


class CirclesDataset(DatasetBase):

    def __init__(self):
        super(CirclesDataset, self).__init__(majority_label=[0], minority_label=1, synth_label=2)
        file_config = './config/dataconfig/circlesconfig.yml'
        with open(file_config, "r") as read_file:
            self.config = yaml.load(read_file, Loader=yaml.FullLoader)

        self.seed = int(self.config['dataconfig']['seed'])
        self.size = int(self.config['dataconfig']['size'])
        self.reduceperc = self.config['dataconfig']['reduceperc']
        self.numClasses = self.config['dataconfig']['numclasses']
        self.class_array = self.config['dataconfig']['classarray']
        self.category_columns = self.config['dataconfig']['categoryfeatures']
        self.load_data()
        self.num_features = len(self.all_data[0])
        self.islarge = self.config['dataconfig']['islarge']
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
        Create the blobs dataset
        Returns:

        """

        data, labels = make_circles(n_samples=self.size,
                                 shuffle=False,
                                 random_state=self.seed,
                                    factor=0.5,
                                    noise=0.01)

        nummin = math.floor((self.size / 2) * self.reduceperc)

        # Reduce dataset artificially
        tempdata = []
        templabels = []
        mincnt = 0
        for sample, label in zip(data, labels):

            if label == 0:
                tempdata.append(sample.tolist())
                templabels.append(int(label))
            else:
                tempdata.append(sample.tolist())
                templabels.append(int(label))
                mincnt += 1
                if mincnt >= nummin:
                    break

        data = tempdata
        labels = templabels


        # Minmax scale data
        data = minmax_scale(data, feature_range=(0, 1), copy=False)

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

        #Add in synthetic data
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
