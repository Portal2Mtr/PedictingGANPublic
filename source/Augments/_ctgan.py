"""Conditional Tabular GAN  (CTGAN)

Classes for managing training for the CTGAN approach.
Originally modified from https://github.com/sdv-dev/CTGAN.

"""


import sys
sys.path.append('./Augments/')
sys.path.append('./Datasets/')
import torch
import logging
# noinspection PyUnresolvedReferences
from _trainer_base import TrainerBase
from ctgan import CTGANSynthesizer

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class CTGanTrainer(TrainerBase):

    def __init__(self, config):
        self.ct_gan = None
        self.num_features = None
        super().__init__(config['augconfig']['seed'])
        # Start basic training loop
        self.config = config
        self.epochs = self.config['augconfig']['epochs']
        self.dataset = []
        return

    def init_networks(self, num_features=None, class_array=None):

        return

    def conduct_training(self, trial=None):
        """
        Conducts training for the WGAN approach
        Args:
            trial (): Optuna trial object

        Returns:

        """
        gen_loss, disc_loss, trial = self.train(trial)
        if trial is None:
            return gen_loss, disc_loss
        else:
            return gen_loss, disc_loss, trial, self.ct_gan, None

    def init_data(self,trainPerc=None):
        """
        Wrapper function for initializing the class dataset
        :return:
        :rtype:
        """
        data_name = self.config['augconfig']['dataset']
        self.dataset = self.get_dataset(data_name, trainPerc=trainPerc)
        self.num_features = self.dataset.num_features

    def train(self, trial):
        """
        Training loop for CTGAN approach
        Args:
            trial (): Optuna trial object

        Returns:

        """

        # Grab original dataset without added noise
        data = self.dataset.orig_data
        num_features = len(data[0])
        dataframe = pd.DataFrame(data, columns=[str(i) for i in range(num_features)])
        train_frame = dataframe.iloc[self.dataset.train_idxs]
        batchsize = int(len(self.dataset.train_idxs) / self.config['ganconfig']['batchsizediv'])
        batchsize = batchsize - (batchsize % 10)

        gen_dim = tuple(self.config['ganconfig']['gen']['layer_nodes'])

        disc_dim = tuple(self.config['ganconfig']['disc']['layer_nodes'])

        self.ct_gan = CTGANSynthesizer(
            epochs=self.config['augconfig']['epochs'],
            generator_lr=self.config['ganconfig']['gen']['genlr'],
            generator_decay=self.config['ganconfig']['gen']['gendecay'],
            generator_dim=gen_dim,
            discriminator_lr=self.config['ganconfig']['disc']['disclr'],
            discriminator_decay=self.config['ganconfig']['disc']['discdecay'],
            batch_size=batchsize,
            discriminator_steps=self.config['ganconfig']['disc']['discsteps'],
            discriminator_dim=disc_dim,
            verbose=True
        )

        # Names of the columns that are discrete
        discrete_columns = [str(i) for i in self.dataset.category_columns]
        self.ct_gan.fit(train_frame, discrete_columns)

        if trial is None:
            return None, None, trial
        else:
            return None, None, trial

    def load_training(self, get_loss=True, trial=None, gen=None, disc=None):
        """
        Load trained networks and data
        Args:
            get_loss (): Get training loss
            trial (): Optuna trial object

        Returns:

        """

        if trial is not None:
            self.ct_gan = gen

        return

    def synth_samples(self, aug_config, total_dataset):
        """
        Synthesize training samples with WGAN approach
        Args:
            aug_config (): Augmentation config
            total_dataset (): Dataset used for augmentation

        Returns:

        """

        synth_samples = []
        class_labels = []
        num_features = total_dataset.num_features
        cat_cols = [str(i) for i in self.dataset.category_columns]
        with torch.no_grad():
            for i, (class_lab, num_gen, cat_col) in enumerate(zip(total_dataset.class_array, total_dataset.num_gen_array, cat_cols)):
                synth_sample = self.ct_gan.sample(num_gen,cat_col,str(i))
                synth_samples.append(synth_sample[0])
                class_labels.append(torch.tensor(class_lab))


        synth_train_X, synth_test_X, synth_train_Y, synth_test_Y = train_test_split(synth_samples, class_labels)

        train_synth_data = [(x,y) for x,y in zip(synth_train_X, synth_train_Y)]
        test_synth_data = [(x,y) for x,y in zip(synth_test_X, synth_test_Y)]

        return train_synth_data, test_synth_data

