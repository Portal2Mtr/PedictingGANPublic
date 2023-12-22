# Optuna training loops for Gan and supervised learner
import optuna
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
import logging
import sys
import plotly
import torch.distributed as dist
import pickle
import os
import psycopg2
from source.Optuna._optuna_classes import ParamTrainAug

logger = logging.getLogger(__name__)



def conduct_optuna_augments(aug_config, sup_config, worker_seed,largedataset=False,trainperc=None):
    gan_config = aug_config['augconfig']
    optuna_config = aug_config['optunaconfig']
    aug_name = aug_config['augconfig']['augname']
    data_name = aug_config['augconfig']['dataset']

    trainer = ParamTrainAug(aug_config, sup_config, largedataset,trainperc)

    # Setup Optuna
    # Use optuna for parameter study
    logger.info("Setting up parameter study...")
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Sup. Loss

    with open('./output/HOSTS/host.txt') as f:
        lines = f.readlines()
    working_host = lines[0].split(' ')[0]

    if not gan_config['has_idn']:

        directions = ["maximize"]
    else:
        directions = ["maximize"]

    if aug_name != 'VECGAN':
        study_name = 'Study' + aug_name + data_name
    else:
        grad_mode = aug_config['augconfig']['gradmode']
        study_name = 'Study' + aug_name + data_name + grad_mode

    if sup_config['supconfig']['use_mlp']:
        study_name = study_name + "_MLP_MODE_{}".format(sup_config['supconfig']['mlp']['mode'])

        if aug_name == 'CONTROL':

            if sup_config['supconfig']['vecganoverride']['override']:
                study_name = study_name + '_OVERRIDE_VECGAN_' + sup_config['supconfig']['vecganoverride']['mode']

        directions = ['maximize']
        study = optuna.create_study(
            study_name=study_name,
            directions=directions,  # Maximize classification accuracy, loss is different for each augmentation
            load_if_exists=True,
            storage="postgresql://postgres@"+working_host,
            sampler=optuna.samplers.CmaEsSampler(seed=worker_seed,warn_independent_sampling=False),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=1500,interval_steps=500))
        # Conduct Optuna
        study.optimize(trainer, n_trials=optuna_config['num_tests'])
    elif not gan_config['has_idn']:

        # Vanilla case
        study = optuna.create_study(
            study_name=study_name,
            directions=directions,  # Maximize classification accuracy, loss is different for each augmentation
            load_if_exists=True,
            storage="postgresql://postgres@"+working_host,
            sampler=optuna.samplers.CmaEsSampler(seed=worker_seed,
                                                 warn_independent_sampling=False),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=1500,interval_steps=500))

        # Conduct Optuna
        study.optimize(trainer, n_trials=optuna_config['num_tests'])

    else:

        train_get = int(trainperc * 100)
        study_name = study_name + "_TRAIN_{}_IDN_{}".format(train_get,gan_config['idn_perc'])
        study = optuna.create_study(
            study_name=study_name,
            directions=directions,  # Maximize classification accuracy, loss is different for each augmentation
            load_if_exists=True,
            storage="postgresql://postgres@"+working_host,
            sampler=optuna.samplers.CmaEsSampler(seed=worker_seed,warn_independent_sampling=False,restart_strategy='ipop'),
            pruner=optuna.pruners.PercentilePruner(25.0,n_startup_trials=5,n_warmup_steps=1500,interval_steps=500))
        # Conduct Optuna
        study.optimize(trainer, n_trials=optuna_config['num_tests'])





    logger.info("!!!!!!!!!!!!!!!!! Parameter Study Complete!!!!!!!!!!!!!!!!!!")
