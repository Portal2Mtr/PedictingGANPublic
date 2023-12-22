"""Vanilla GAN Class

Contains the torch implementation of the VANILLA GAN for data augmentation with batch processing.

"""
import sys
sys.path.append('./source/Augments/')
sys.path.append('./Datasets/')
import torch
import logging
import torch.nn
# noinspection PyUnresolvedReferences
from _trainer_base import TrainerBase

DEVICE = torch.device("cpu")
logger = logging.getLogger(__name__)


class ControlTrainer(TrainerBase):
    """
    Class for managing training for the BASIC GAN.
    """

    def __init__(self, config):
        super().__init__(seed=config['augconfig']['seed'])
        torch.set_default_tensor_type('torch.DoubleTensor')
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

    def init_networks(self, trial=None, numFeatures=None):
        """
        Initializes the MLP networks for training
        :param numFeatures: Number of features for dataset
        :return:
        """

        return

    def conduct_training(self, trial=None,classes=None):
        """
        Conducts trainign for individual testing or parameter study.
        :param trial: Optuna trial
        :return:
        """

        gen_loss = None
        disc_loss = None
        if trial is None:
            return gen_loss, disc_loss
        else:
            return gen_loss, disc_loss, trial, None, None

    def create_networks(self, num_features, config, trial=None):
        """
        Creates the networks for the BASIC GAN
        :param num_features:
        :type num_features:
        :param config:
        :type config:
        :return:
        :rtype:
        """

        return

    def train(self, trial=None):
        """
        Wrapper function to manage entire training cycle of BASIC GAN.
        :param trial: Optuna trail
        :return:
        :rtype:
        """

        if trial is not None:
            return None, None, trial
        else:
            return None, None


