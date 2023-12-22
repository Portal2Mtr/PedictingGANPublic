"""Optuna Training classes

Wrapper classes for the Optuna training loop

"""

import sys

import optuna.integration
import torch
import random
import numpy as np
import torch.distributed as dist

DEVICE = torch.device("cpu")

sys.path.append('./source/Augments/')
sys.path.append('./source/Sup_Learning/')

# noinspection PyUnresolvedReferences
from _vanilla_gan import VanillaGANTrainer
# noinspection PyUnresolvedReferences
from _wgan import WassGanTrainer
# noinspection PyUnresolvedReferences
from _vecgan import VecGanTrainer
# noinspection PyUnresolvedReferences
from _rcgan import RCGanTrainer
# noinspection PyUnresolvedReferences
from _treeclass import TreeTrainer
# noinspection PyUnresolvedReferences
from _smote import SmoteTrainer
# noinspection PyUnresolvedReferences
# from _ctgan import CTGanTrainer
# noinspection PyUnresolvedReferences
from _control import ControlTrainer
import logging
from statistics import mean

logger = logging.getLogger(__name__)


class ParamTrainAug(object):

    def __init__(self, aug_config, sup_config, largeDataset,trainperc):
        train_augs = {'VANILLAGAN': VanillaGANTrainer,
                      'WGAN': WassGanTrainer,
                      'VECGAN': VecGanTrainer,
                      'SMOTE': SmoteTrainer,
                      # 'CTGAN': CTGanTrainer,
                      'CONTROL': ControlTrainer,
                      'RCGAN':RCGanTrainer}
        self.augconfig = aug_config
        self.optuna_config = aug_config['optunaconfig']
        self.params = self.optuna_config['params']
        self.aug_name = self.augconfig['augconfig']['augname']

        # Load training class with data
        self.aug_trainer = train_augs[self.aug_name](self.augconfig)
        self.aug_trainer.init_data(trainPerc=trainperc)
        self.sup_config = sup_config
        self.largeDataset = largeDataset
        self.trainperc = trainperc
        self.use_mlp = sup_config['supconfig']['use_mlp']

    def init_optuna(self, trial):
        """
        Initializes the parameters for training in optuna
        Args:
            trial (): Optuna trial object

        Returns:

        """

        if self.aug_name != 'CONTROL':

            # Assign parameters for training in optuna
            assign_list = self.optuna_config['paramlist']
            param_list = []
            if len(self.optuna_config['params']):
                param_list = list(self.optuna_config['params'].items())
            layers = [[0, 0] for i in range(10)]  # (genlayers, disclayers)
            for assign, param in zip(assign_list, param_list):
                if isinstance(param[1][1], int):

                    if ('genlayer' in param[0]) or ('disclayer' in param[0]):
                        net_idx = 0 if 'genlayer' in param[0] else 1

                        idx = int(param[0][-1])
                        if idx == 1:
                            layer_nodes = trial.suggest_int(param[0], low=param[1][0], high=param[1][1], step=param[1][2])
                        else:
                            layer_nodes = trial.suggest_int(param[0], low=param[1][0], high=layers[idx-2][net_idx], step=param[1][2])
                        exec("%s = %d" % (assign, layer_nodes))
                        layers[idx - 1][net_idx] = layer_nodes
                        continue

                    exec("%s = %d" % (assign, trial.suggest_int(param[0], low=param[1][0], high=param[1][1], step=param[1][2])))
                else:
                    exec("%s = %f" % (assign, trial.suggest_float(param[0], low=param[1][0], high=param[1][1], step=param[1][2])))

        param_list = []
        if not self.largeDataset:
            assign_list = self.sup_config['optunaconfig']['paramlist']
            if len(self.sup_config['optunaconfig']['params']):
                param_list = list(self.sup_config['optunaconfig']['params'].items())
        elif self.use_mlp:
            assign_list = self.sup_config['optunaconfig']['mlpparamlist']
            if len(self.sup_config['optunaconfig']['mlpparams']):
                param_list = list(self.sup_config['optunaconfig']['mlpparams'].items())
        else:
            assign_list = self.sup_config['optunaconfig']['bigparamlist']
            if len(self.sup_config['optunaconfig']['bigparams']):
                param_list = list(self.sup_config['optunaconfig']['bigparams'].items())

        for assign, param in zip(assign_list, param_list):
            if isinstance(param[1][1], int):
                exec("%s = %d" % (assign, trial.suggest_int(param[0], low=param[1][0], high=param[1][1], step=param[1][2])))
            else:
                exec("%s = %f" % (assign, trial.suggest_float(param[0], low=param[1][0], high=param[1][1], step=param[1][2])))

        return trial

    def __call__(self, single_trial):

        # trial = optuna.integration.TorchDistributedTrial(single_trial)
        trial = single_trial
        trial = self.init_optuna(trial)

        # Reset learner seed in trial
        SEED = 0
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)  # Used in sklearn

        # Start training and update parameters
        self.aug_trainer.config = self.augconfig
        self.aug_trainer.init_data(self.trainperc)
        self.aug_trainer.init_networks()
        self.sup_class = TreeTrainer(self.sup_config, self.augconfig)
        self.sup_class.init_data(self.trainperc)
        self.sup_class.init_params()
        classes = self.sup_class.init_networks()
        gen_loss, disc_loss, trial, gen, disc = self.aug_trainer.conduct_training(trial,classes=classes)
        # Models are saved for testing

        # Get sup learner accuracy for study, loads from training

        self.sup_class.init_augment(trial=trial, gen=gen, disc=disc)
        acc_scores, f1_scores, trial = self.sup_class.conduct_training(trial)

        if self.use_mlp:
            return acc_scores

        if self.augconfig['augconfig']['has_idn']:
            return mean([acc_scores[0],acc_scores[1],acc_scores[2]])

        return mean([acc_scores[0],acc_scores[1],acc_scores[2]])
