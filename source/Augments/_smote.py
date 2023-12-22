"""SMOTE Augmentation Class

Wrapper class for smote from sklearn.

"""

import sys

import numpy as np

sys.path.append('./Augments/')
import logging
from imblearn.over_sampling._smote import SMOTE
import torch
from sklearn.model_selection import train_test_split
# noinspection PyUnresolvedReferences
from _trainer_base import TrainerBase

logger = logging.getLogger(__name__)


class SmoteTrainer(TrainerBase):
    """
    Wrapper class for SMOTE from sklearn.
    """

    def __init__(self, config):
        super().__init__()
        torch.set_default_tensor_type('torch.DoubleTensor')
        # Start basic training loop

        self.config = config
        self.dataset = []

        return

    def init_data(self,trainPerc=None):
        """
        Wrapper function for initializing the class dataset
        :return:
        :rtype:
        """
        data_name = self.config['augconfig']['dataset']
        self.dataset = self.get_dataset(data_name,trainPerc=trainPerc)
        self.num_features = self.dataset.num_features

    def init_networks(self):
        # No networks to manage
        return

    def conduct_training(self, trial=None,classes=None):
        """
        No training for smote
        Args:
            trial (): Optuna trial object

        Returns:

        """

        return None, None, trial, None, None

    def synth_samples(self, aug_config, total_dataset):
        """
        Synthesize samples using smote.
        :param aug_config: Augmenation config 
        :param total_dataset: Dataset used for augmentation
        :return: 
        """

        datalabels = total_dataset.all_labels
        numlabels = np.unique(datalabels,return_counts=True)[1]
        minlabels = min(numlabels)
        aug_config['augconfig']['neighbors'] = min(aug_config['augconfig']['neighbors'], minlabels - 1)

        # Load generator
        self.smote_gen = SMOTEClass(aug_config)
        train_synth_data = []
        train_synth_labels = []
        test_synth_data = []
        test_synth_labels = []
        all_data = []

        for i in total_dataset.train_idxs:
            all_data.append(total_dataset[i])

        for i in total_dataset.test_idxs:
            all_data.append(total_dataset[i])

        all_samples = []
        all_labels = []
        for idx, data in enumerate(all_data):
            temp_data = data[0]
            temp_label = data[1]
            all_samples.append(temp_data.tolist())
            all_labels.append(int(temp_label))

        synth_start_idx = len(all_data)

        # Use smote to balance labels for data input
        all_synth_samples, all_synth_labels = self.smote_gen(all_samples, all_labels)

        synth_sample_torch = [torch.Tensor(all_synth_samples[i])
                              for i in range(synth_start_idx, len(all_synth_samples))]

        synth_label_torch = [torch.tensor(all_synth_labels[i],
                                          dtype=torch.int) for i in range(synth_start_idx, len(all_synth_samples))]

        # Train/test split data for test execution
        x_train, x_test, y_train, y_test = train_test_split(synth_sample_torch,
                                                            synth_label_torch,
                                                            test_size=(1 - total_dataset.config['dataconfig']['trainperc']),
                                                            random_state=0)
        # Add data to appropriate arrays
        for data,label in zip(x_train,y_train):
            train_synth_data.append((data,label))

        for data,label in zip(x_test,y_test):
            test_synth_data.append((data,label))

        return train_synth_data, test_synth_data


class SMOTEClass:
    """
    Wrapper class for sklearn smote function
    """

    def __init__(self, config):

        self.aug_name = config['augconfig']['augname']
        self.aug_config = config['augconfig']

        self.SMOTE = SMOTE(random_state=config['augconfig']['seed'],
                           k_neighbors=config['augconfig']['neighbors'])

    def __call__(self, data, labels):
        
        x_rebalanced, y_rebalanced = self.SMOTE.fit_resample(data, labels)
        
        return x_rebalanced, y_rebalanced
        

