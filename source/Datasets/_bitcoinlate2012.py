"""Bitcoin Dataset Class

Managing class for the Bitcoin dataset

"""
import random
import sys
sys.path.append('./Datasets/')

import csv
import numpy as np
import logging
from sklearn.utils import shuffle
import math
import numpy.random
# noinspection PyUnresolvedReferences
from source.Datasets._dataset_base import DatasetBase
from torch.utils.data import Subset
import yaml
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class BitcoinDataset(DatasetBase):

    def __init__(self, trainSplit=None,downsample=False):
        super(BitcoinDataset, self).__init__()
        file_config = './config/dataconfig/bitcoinlate2012config.yml'

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

        self.numClasses = self.config['dataconfig']['numclasses']
        self.synth_label = "SYNTH"
        self.class_array = self.config['dataconfig']['classarray']
        self.category_columns = self.config['dataconfig']['categoryfeatures']
        self.downsample = downsample # Downsample innocent class
        self.downsamplesize = self.config['dataconfig']['inndownsample']
        self.num_gen_array = []
        self.load_data()
        self.num_features = len(self.all_data[0])
        self.islarge = self.config['dataconfig']['islarge']
        self.has_idn = self.config['dataconfig']['has_idn']
        self.overwrite_synth_labels = False

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
        Loads the Late 2012 3-month dataset period for testing
        Returns:

        """

        # Get late 2012 transactions and attacks
        logger.info("No pickle file found! Generating datasets!")
        trxn_data = []
        # Load 2831819 Transactions...
        with open(self.config['fileconfig']['datafile'], newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_reader.__next__()
            for idx, row in enumerate(csv_reader):
                trxn_data.append(row)

        malc_data = []
        # Load Attack transactions
        with open(self.config['fileconfig']['malcFile'], newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_reader.__next__()
            for row in csv_reader:
                malc_data.append(row)

        # Get trojan data from malicious data for generation
        train_labels_using = ["Trojan"]
        trojan_data = []
        other_attacks = []
        for entry in malc_data:
            parse_data = [i for i in entry]
            inlabels = True in [label in entry[-1] for label in train_labels_using]

            if inlabels:
                trojan_data.append(parse_data)
            else:
                other_attacks.append(parse_data)

        # Get innocent data
        malc_ind = [i[-2] for i in trxn_data]  # Offset for block number
        inn_idxs = [entry == "0" for idx, entry in enumerate(malc_ind)]
        inn_data = [trxn_data[idx] for idx, entry in enumerate(inn_idxs) if entry]

        if self.downsample:
            inn_data = random.sample(inn_data, self.downsamplesize)

        # Parse working data
        inn_rel_data = [i[1:8] for i in inn_data]  # Remove hash and block number
        inn_rel_labels = [0] * len(inn_data)
        trojan_rel_data = [i[1:8] for i in trojan_data]  # Remove hash
        trojan_rel_labels = [1] * len(trojan_data)
        other_attack_rel_data = [i[1:8] for i in other_attacks]  # Remove hash
        other_attack_rel_labels = [1] * len(other_attacks)

        # Array of unscaled malicious datapoints
        malc_array = []
        for entry in malc_data:
            malc_entry = [float(i) for i in entry[1:8]]
            malc_array.append(malc_entry)

        self.malc_array = malc_array

        data_lists = [inn_rel_data, trojan_rel_data, other_attack_rel_data]

        for idx, data_list in enumerate(data_lists):
            for jdx, sample in enumerate(data_list):
                for kdx, feature in enumerate(sample):
                    data_lists[idx][jdx][kdx] = float(data_lists[idx][jdx][kdx])

        # uniform_mag = 0.01
        # for idx, data_list in enumerate(data_lists):
        #     for jdx, sample in enumerate(data_list):
        #         for kdx, feature in enumerate(sample):
        #             if kdx in self.category_columns:
        #                 data_lists[idx][jdx][kdx] = data_lists[idx][jdx][kdx] + numpy.random.uniform(0, uniform_mag)

        # Minmaxscale all data manually
        # feat_max = [0] * len(inn_rel_data[0])
        # feat_min = [1] * len(inn_rel_data[0])
        # for data_list in data_lists:
        #     for sample in data_list:
        #         for idx, feature in enumerate(sample):
        #             feat_max[idx] = max(feat_max[idx], feature)
        #             feat_min[idx] = min(feat_min[idx], feature)

        # Scale  to [0,1] manually
        # for idx, data_list in enumerate(data_lists):
        #     for jdx, sample in enumerate(data_list):
        #         for kdx, feature in enumerate(sample):
        #             data_lists[idx][jdx][kdx] = \
        #                 (1/(feat_max[kdx]-feat_min[kdx]))*(data_lists[idx][jdx][kdx] - feat_min[kdx])

        inn_rel_data = data_lists[0]
        trojan_rel_data = data_lists[1]
        other_attack_rel_data = data_lists[2]

        # Train/test split innocent data based on number of attack samples used in training set
        allattacknum = len(trojan_data) + len(other_attacks)
        trainperc = len(trojan_data) / float(allattacknum)

        totalsamples = allattacknum + len(inn_rel_data)

        trainsize = math.floor(trainperc * totalsamples) - len(trojan_data)

        trainset = []
        for sample in inn_rel_data[0:trainsize]:
            trainset.append(sample)

        for sample in trojan_rel_data:
            trainset.append(sample)

        testset = []
        for sample in inn_rel_data[trainsize:]:
            testset.append(sample)

        for sample in other_attack_rel_data:
            testset.append(sample)

        trainlabels = inn_rel_labels[0:trainsize]
        trainlabels.extend(trojan_rel_labels)
        trainidxs = [i for i in range(len(trainlabels))]
        trainidxs = shuffle(trainidxs)
        trainset = [trainset[i] for i in trainidxs]
        trainlabels = [trainlabels[i] for i in trainidxs]

        testlabels = inn_rel_labels[trainsize:]
        testlabels.extend(other_attack_rel_labels)
        testidxs = [i for i in range(len(testset))]
        testidxs = shuffle(testidxs)
        testset = [testset[i] for i in testidxs]
        testlabels = [testlabels[i] for i in testidxs]

        # Format dataset for simulations
        all_data = []
        all_labels = []
        for sample, label in zip(trainset, trainlabels):
            all_data.append(np.array(sample))
            all_labels.append(label)

        train_idx = [i for i in range(len(all_data))]
        train_inn_idx = [i for i in range(len(all_labels)) if all_labels[i] == 0]

        for sample, label in zip(testset, testlabels):
            all_data.append(np.array(sample))
            all_labels.append(label)

        test_idx = [i for i in range(train_idx[-1], len(all_data))]
        test_inn_idx = [i for i in range(train_idx[-1], len(all_labels)) if all_labels[i] == 0]

        self.all_data = all_data
        self.all_labels = all_labels
        self.train_idxs = train_idx
        self.test_idxs = test_idx
        self.train_data = Subset(self.all_data, self.train_idxs)
        self.test_data = Subset(self.all_data, self.test_idxs)

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
