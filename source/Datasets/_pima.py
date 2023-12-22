"""Pima Dataset

Managing class for the Pima Diabetes dataset.

"""


# Bitcoin dataset pytorch class
import sys
import copy
from pandas import read_csv
from sklearn.preprocessing import minmax_scale
import logging
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
import numpy.random
import numpy as np
from torch.utils.data import Subset
from IDN.networks.mlp import MLP
from IDN.utils import get_softmax_out
from IDN.ops import train
import yaml
sys.path.append('./Datasets/')
from source.Datasets._dataset_base import DatasetBase


logger = logging.getLogger(__name__)


class PimaDataset(DatasetBase):

    def __init__(self,trainSplit=None,idnPerc=None):
        super(PimaDataset, self).__init__()
        file_config = './config/dataconfig/pimaconfig.yml'
        with open(file_config, "r") as read_file:
            self.config = yaml.load(read_file, Loader=yaml.FullLoader)

        if trainSplit is not None:
            self.config['dataconfig']['trainperc'] = trainSplit

        self.train_split = self.config['dataconfig']['trainperc']

        self.orig_data = None
        self.all_data = None
        self.all_labels = None
        self.train_idxs = None
        self.test_idxs = None
        self.train_data = None
        self.test_data = None

        self.idn_perc = idnPerc
        self.has_idn = self.config['dataconfig']['has_idn']
        self.numClasses = self.config['dataconfig']['numclasses']
        self.synth_label = "SYNTH"
        self.class_array = self.config['dataconfig']['classarray']
        self.category_columns = self.config['dataconfig']['categoryfeatures']
        self.num_gen_array = []
        self.load_data()
        self.num_features = len(self.all_data[0])
        self.islarge = self.config['dataconfig']['islarge']
        self.overwrite_synth_labels = False

        return

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        if self.overwrite_synth_labels:
            data = torch.reshape(Tensor(self.all_data[idx]), (self.num_features,))
            if type(self.all_labels[idx]) == str:
                data_label = int(self.all_labels[idx].split('_')[0])
                label = torch.tensor(data_label, dtype=torch.double, requires_grad=True)
            else:
                data = torch.reshape(Tensor(self.all_data[idx]), (self.num_features,))
                label = torch.tensor(self.all_labels[idx], dtype=torch.double, requires_grad=True)
        else:
            data = torch.reshape(Tensor(self.all_data[idx]), (self.num_features,))
            label = self.all_labels[idx]

        data.requires_grad = True
        return data, label

    def calc_synth_gen(self):
        """
        Calculate the number of samples to generate for GAN approaches
        Returns:

        """

        label_set = self.class_array
        num_labels = [0 for _ in range(len(label_set))]
        for j, label in enumerate(label_set):
            for i in range(len(self.all_labels)):
                if self.all_labels[i] == label:
                    num_labels[j] += 1

        largestidx = np.argmax(num_labels)
        largestcnt = num_labels[largestidx]
        num_gen = [largestcnt - j for j in num_labels]
        self.num_gen_array = num_gen

    def load_data(self):
        """
        Loads the pima dataset
        Returns:

        """

        dataframe = read_csv(self.config['fileconfig']['dataFile'])
        # get the values
        values = dataframe.values
        data, labels = values[:, :-1], values[:, -1]
        self.orig_data = copy.deepcopy(data)

        # Add noise to each sample for categorical values
        uniform_mag = 0.01
        for idx, sample in enumerate(data):
            for jdx, feature in enumerate(sample):
                if jdx in self.category_columns:
                    data[idx, jdx] = data[idx, jdx] + numpy.random.uniform(0, uniform_mag)

        # Minmax scale data
        data = minmax_scale(data, feature_range=(0, 1), copy=False)
        labels = [int(labels[i]) for i in range(len(labels))]

        # Train Test split
        data = [(i, sample) for i, sample in enumerate(data)]
        data_train, data_test, y_train, y_test = train_test_split(data, labels,
                                                            test_size=(1-self.config['dataconfig']['trainperc']),
                                                            random_state=0)

        self.train_idxs = [i[0] for i in data_train]
        data_train = [i[1] for i in data_train]
        self.test_idxs = [i[0] for i in data_test]
        data_test = [i[1] for i in data_test]

        all_data = []
        all_labels = []
        for sample, label in zip(data_train, y_train):
            all_data.append(sample)
            all_labels.append(label)

        train_idxs = [i for i in range(len(all_data))]

        # Add IDN noise

        if self.has_idn and self.idn_perc >= 0.01:
            all_labels = self.add_idn(all_data,all_labels)

        for sample, label in zip(data_test, y_test):
            all_data.append(sample)
            all_labels.append(label)

        test_idxs = [i for i in range(train_idxs[-1], len(all_data))]

        self.all_data = all_data
        self.all_labels = all_labels
        self.train_idxs = train_idxs
        self.test_idxs = test_idxs
        self.train_data = Subset(self.all_data, self.train_idxs)
        self.test_data = Subset(self.all_data, self.test_idxs)

        self.calc_synth_gen()
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
            self.all_data = self.orig_data.tolist()

        for data in train_data:
            self.all_data.append(data[0])
            self.all_labels.append("{}_{}".format(data[1].item(), self.synth_label))
            self.train_idxs.append(len(self.all_data) - 1)

        for data in test_data:
            self.all_data.append(data[0])
            self.all_labels.append("{}_{}".format(data[1].item(), self.synth_label))
            self.test_idxs.append(len(self.all_data) - 1)

    def add_idn(self,data,labels):

        input_feat = len(data[0])
        num_classes = self.numClasses
        model = MLP(input_feat=input_feat,num_classes=num_classes)

        # Training
        softmax_out_avg = np.zeros([len(data), num_classes])
        train_dataset = [(i,float(j)) for i,j in zip(data,labels)]
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(data), shuffle=False)
        softmax_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(data), shuffle=False)
        for epoch in range(1, 300):
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
            args = {'epochs':300}
            train(args, model, 'cpu', train_loader, optimizer, epoch)
            softmax_out_avg += get_softmax_out(model, softmax_loader, 'cpu')

        softmax_out_avg /= 300

        logger.info('Generating noisy labels according to softmax_out_avg for training set...')
        label = np.array([i for i in labels])
        label_noisy_cand, label_noisy_prob = [], []
        for i in range(len(labels)):
            pred = softmax_out_avg[i,:].copy()
            pred[int(label[i])] = -1
            label_noisy_cand.append(np.argmax(pred))
            label_noisy_prob.append(np.max(pred))

        label_noisy = label.copy()
        idn_form = self.idn_perc
        index = np.argsort(label_noisy_prob)[-self.idn_perc:]
        label_noisy[index] = np.array(label_noisy_cand)[index]

        return label_noisy.tolist()