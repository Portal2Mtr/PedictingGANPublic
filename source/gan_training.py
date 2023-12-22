"""Malicious Blockchain Transaction Grenerative Adversarial Network

Contains the generative adversarial network (GAN) for synthesizing malicious
blockchain transactions. The GAN uses transactions from a large 2012 attack to
generate synthetic data for producing a balanced dataset for a supervised
learner in 'supGan.py'. The GAN uses Pytorch as a backend.

"""

__author__ = "Charles Rawlins"
__copyright__ = "Copyright 2020"
__credits__ = ["Charles Rawlins"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Charles Rawlins"
__email__ = "crfmb@mst.edu"
__status__ = "Prototype"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys
sys.path.append('./source/Augments/')
sys.path.append('./source/Misc/')
sys.path.append('./source/Optuna/')
sys.path.append('./')
import argparse
import yaml
import torch
import pickle
import logging
import random
import tracemalloc
import os
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from statistics import mean
from colorlog import ColoredFormatter
from multiprocessing import Pool
# noinspection PyUnresolvedReferences
from _vanilla_gan import VanillaGANTrainer
# noinspection PyUnresolvedReferences
from _wgan import WassGanTrainer
# noinspection PyUnresolvedReferences
# from _plotting import plot_loss_aug
# noinspection PyUnresolvedReferences
from _vecgan import VecGanTrainer
# noinspection PyUnresolvedReferences
from _rcgan import RCGanTrainer
# # noinspection PyUnresolvedReferences
# from _ctgan import CTGanTrainer
# noinspection PyUnresolvedReferences
from _smote import SmoteTrainer
# noinspection PyUnresolvedReferences
from _optuna_loop import conduct_optuna_augments
# noinspection PyUnresolvedReferences
from _treeclass import TreeTrainer
# noinspection PyUnresolvedReferences
from _control import ControlTrainer
# # noinspection PyUnresolvedReferences
# from _rcganu import RCGANUTrainer

# Setup logger
logger = logging.getLogger(__name__)


def configure_logging(verbosity, enable_colors):
    """
    Configures the code logger for reporting various information.
    :param verbosity: Sets logger verbosity level
    :type verbosity: str
    :param enable_colors: Enables logger colors
    :type enable_colors: bool
    :return:
    :rtype:
    """
    root_logger = logging.getLogger()
    console = logging.StreamHandler()

    if enable_colors:
        # create a colorized formatter
        formatter = ColoredFormatter(
            "%(log_color)s[%(filename)s] %(asctime)s %(levelname)-8s%(reset)s %(white)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "cyan,bg_red",
            },
            secondary_log_colors={},
            style="%"
        )
    else:
        # create a plain old formatter
        formatter = logging.Formatter(
            "[%(filename)s] %(asctime)s %(levelname)-8s %(message)s"
        )

    # Add the formatter to the console handler, and the console handler to the root logger
    console.setFormatter(formatter)
    for hdlr in root_logger.handlers:
        root_logger.removeHandler(hdlr)
    root_logger.addHandler(console)

    # Set logging level for root logger
    root_logger.setLevel(verbosity)


def validate_data(seed=0, aug_name='VANILLAGAN', aug_config=[], sup_config=[], dataconfig=[],memoryStudy=0):
    """
    Runs a parallel-processed study to validate a particular GAN configuration
    Args:
        seed ():
        aug_name ():
        aug_config ():
        sup_config ():

    Returns:

    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)  # Used in sklearn
    aug_config['augconfig']['seed'] = seed
    sup_config['supconfig']['seed'] = seed

    trainer = None
    if aug_name == 'VANILLAGAN':
        trainer = VanillaGANTrainer(aug_config)
    elif aug_name == 'WGAN':
        trainer = WassGanTrainer(aug_config)
    # elif aug_name == 'CTGAN':
    #     trainer = CTGanTrainer(aug_config)
    elif aug_name == 'VECGAN':
        trainer = VecGanTrainer(aug_config)
    elif aug_name == 'RCGAN':
        trainer = RCGanTrainer(aug_config)
    elif aug_name == 'SMOTE':
        trainer = SmoteTrainer(aug_config)
    elif aug_name == 'CONTROL':
        trainer = ControlTrainer(aug_config)

    if not memoryStudy:
        trainer.init_data(dataconfig['dataconfig']['trainperc'])
    else:
        trainer.init_data()
    trainer.init_networks()
    gen_loss, disc_loss, trial, gen, disc = trainer.conduct_training(trial=1)
    logger.info("Training for seed {} complete!".format(seed))

    sup_class = TreeTrainer(sup_config, aug_config)
    if not memoryStudy:
        sup_class.init_data(dataconfig['dataconfig']['trainperc'])
    else:
        sup_class.init_data()
    sup_class.init_params()
    classes = sup_class.init_networks()
    sup_class.init_augment(trial=1,gen=gen,disc=disc)
    acc_scores, f1_scores, _ = sup_class.conduct_training(trial)
    logger.info("Testing for seed {} complete!".format(seed))

    return acc_scores, f1_scores


def assignParams(augconfig, supconfig, dataconfig, new_values):

    aug_name = augconfig['augconfig']['augname']

    if aug_name != 'CONTROL':

        # Assign parameters for training in optuna
        assign_list = config['optunaconfig']['paramlist']
        key_list = []
        if len(config['optunaconfig']['params']):
            key_list = list(config['optunaconfig']['params'].keys())
        for assign, key in zip(assign_list, key_list):

            new_assign = assign[5:]

            for new_key in new_values.keys():
                if new_key == key:
                    if isinstance(new_values[new_key],int):
                        exec("%s = %d" % (new_assign, new_values[new_key]))
                    else:
                        exec("%s = %f" % (new_assign, new_values[new_key]))
                    break


    if not dataconfig['dataconfig']['islarge']:
        sup_assign_list = sup_config['optunaconfig']['paramlist']
        sup_key_list = []
        if len(sup_config['optunaconfig']['params']):
            sup_key_list = list(sup_config['optunaconfig']['params'].keys())
    else:
        sup_assign_list = sup_config['optunaconfig']['bigparamlist']
        sup_key_list = []
        if len(sup_config['optunaconfig']['bigparams']):
            sup_key_list = list(sup_config['optunaconfig']['bigparams'].keys())

    for sup_assign, sup_key in zip(sup_assign_list, sup_key_list):

        new_sup_assign = sup_assign[5:]

        for new_sup_key in new_values.keys():

            if new_sup_key == sup_key:

                if isinstance(new_values[new_sup_key], int):
                    exec("%s = %d" % (new_sup_assign, new_values[new_sup_key]))
                else:
                    exec("%s = %f" % (new_sup_assign, new_values[new_sup_key]))

                break

    return config,sup_config


def assignNeurons(config,newnerons):

    if 'ganconfig' in config.keys():
        config['ganconfig']['gen']['layer_nodes'] = newnerons
        config['ganconfig']['disc']['layer_nodes'] = newnerons

    return config


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Config file input')
    parser.add_argument('--fileConfig', type=str,
                        default='./config/augconfig/vanillaganconfig.yml',
                        help='Config files for simulations (SMOTE, VANILLAGAN, WGAN, CTGAN, VECGAN)',
                        required=True)

    parser.add_argument('--workerSeed', type=int,default=0,help='Optuna worker seed')
    parser.add_argument('--numcpus', type=int, default=None, help='Optuna worker seed')
    parser.add_argument('--gradMode', type=str, default='SGD', help='VecGAN grad mode')
    parser.add_argument('--datasetconfig',type=str,default=None, help='Learning dataset')
    parser.add_argument('--runStudy', type=int, default=0, help='Run a parameter study')
    parser.add_argument('--validateStudy', type=int, default=0, help='Validate a parameter study')
    parser.add_argument('--traintestsplit', type=int, default=None, help='Train/test split percentage (ex: 80)')
    parser.add_argument('--memoryStudy', type=int, default=None, help='Run a memory study ')
    parser.add_argument('--isFixed', type=int,default=0,help='Whether doing fixed parameters for validation')
    parser.add_argument('--newNeurons', type=str,default=None, help='Fixed parameter study neurons')
    parser.add_argument('--numEpochs', type=int,default=None, help='Number of augmentation epochs')
    parser.add_argument('--debugMode', type=int,default=None, help='Number of augmentation epochs')
    parser.add_argument('--idnPerc', type=int,default=None, help='Percent IDN Noise')
    args = parser.parse_args()
    fileConfig = args.fileConfig
    num_cpus = args.numcpus

    configure_logging("INFO", False)
    # Setup parser and parse arguments for training
    with open(fileConfig, "r") as read_file:
        config = yaml.load(read_file, Loader=yaml.FullLoader)

    idn_perc = args.idnPerc
    if idn_perc is not None:
        if type(idn_perc) == str:
            if idn_perc.strip() == 'None':
                idn_perc = None
    config['augconfig']['idn_perc'] = idn_perc

    # Override args
    dataset = args.datasetconfig
    if dataset is not None:
        with open(dataset, "r") as read_file:
            data_config = yaml.load(read_file, Loader=yaml.FullLoader)

        config['augconfig']['dataset'] = data_config['dataconfig']['dataname']

        train_test = args.traintestsplit
        if train_test is not None:
            temp = round(float(train_test) / 100, ndigits=2)
            data_config['dataconfig']['trainperc'] = temp

    aug_name = config['augconfig']['augname']
    data_name = config['augconfig']['dataset']

    sup_prefix = fileConfig.split('/')[-1]
    sup_prefix = sup_prefix.split('config')[0]

    if aug_name != 'VECGAN':
        sup_config_file = './config/supconfig/sup' + sup_prefix +'config.yml'
    else:
        sup_config_file = './config/supconfig/sup' + sup_prefix + str.lower(config['augconfig']['gradmode']) + 'config.yml'

    with open(sup_config_file, "r") as read_file:
        sup_config = yaml.load(read_file, Loader=yaml.FullLoader)

    sup_config['supconfig']['dataset'] = config['augconfig']['dataset']
    numEpochs = args.numEpochs
    if numEpochs is not None:
        config['augconfig']['epochs'] = numEpochs
        logger.info("Changed to new epochs: {}".format(numEpochs))

    do_mem_study = 1 if args.memoryStudy is not None else 0
    debug = 1 if args.debugMode is not None else 0
    mem_workers = None

    if args.runStudy:
        controlConfig = {'do_training': False,
                         'do_testing': False,
                         'plot_tsne_sample': False,
                         'plot_loss': False,
                         'conduct_study': True,
                         'plot_study': False,
                         'validate_study': False}
    elif args.validateStudy:
        controlConfig = {'do_training': False,
                         'do_testing': False,
                         'plot_tsne_sample': False,
                         'plot_loss': False,
                         'conduct_study': False,
                         'plot_study': False,
                         'validate_study': True}
    elif do_mem_study:
        controlConfig = {'do_training': False,
                         'do_testing': False,
                         'plot_tsne_sample': False,
                         'plot_loss': False,
                         'conduct_study': False,
                         'plot_study': False,
                         'validate_study': True}
        config['augconfig']['dataset'] = 'PIMA'
        mem_workers = 1
    elif debug:
        controlConfig = {'do_training': True,
                         'do_testing': False,
                         'plot_tsne_sample': False,
                         'plot_loss': False,
                         'conduct_study': False,
                         'plot_study': False,
                         'validate_study': False}
    else:
        # Use config from file
        controlConfig = config['controlConfig']

    SEED = 0
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)  # Used in sklearn

    trainer = None
    # Various options to speedup execution and reduce computation
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    pd.options.mode.chained_assignment = None
    if num_cpus is None:
        torch.set_num_threads(6)
    else:
        torch.set_num_threads(num_cpus)
    # Pytorch parallel processing vars

    if aug_name == 'VANILLAGAN':
        trainer = VanillaGANTrainer(config)
    elif aug_name == 'WGAN':
        trainer = WassGanTrainer(config)
    # elif aug_name == 'CTGAN':
    #     trainer = CTGanTrainer(config)
    elif aug_name == 'VECGAN':
        grad_mode = args.gradMode
        if grad_mode is not None:
            config['augconfig']['gradmode'] = grad_mode
        trainer = VecGanTrainer(config)
    elif aug_name == 'SMOTE':
        trainer = SmoteTrainer(config)
    elif aug_name == 'CONTROL':
        trainer = ControlTrainer(config)
    elif aug_name == 'RCGAN':
        trainer = RCGanTrainer(config)

    # Conduct main training loop with params loaded from yml files
    if controlConfig['do_training']:
        logger.info('Starting GAN training!')
        trainer.init_data()
        trainer.init_networks()
        gen_loss, disc_loss, trial, gen, disc = trainer.conduct_training(trial=1)
        aug_config = config
        sup_class = TreeTrainer(sup_config, aug_config)
        sup_class.init_data()
        sup_class.init_params()
        sup_class.init_networks()
        sup_class.init_augment(trial=1,gen=gen, disc=disc)
        save_synth = do_mem_study is None or do_mem_study == 0
        acc_scores, f1_scores = \
            sup_class.conduct_training(save_synth=save_synth)

        # logger.info("Accuracy Scores: {}, f1_scores: {}".format(acc_scores,f1_scores))

    # if controlConfig['do_testing']:
    #
    #     aug_config = config
    #     sup_class = TreeTrainer(sup_config, aug_config)
    #     sup_class.init_data()
    #     sup_class.init_params()
    #     sup_class.init_networks()
    #     sup_class.init_augment()  # Loads models from memory
    #     save_synth = do_mem_study is None or do_mem_study == 0
    #     class1_acc, class2_acc, class3_acc, class1_f1, class2_f1, class3_f1 = \
    #         sup_class.conduct_training(save_synth=save_synth)

    if controlConfig['conduct_study']:
        # Pytorch DDP implementation of parameter study
        worker_seed = args.workerSeed

        conduct_optuna_augments(config, sup_config, worker_seed, data_config['dataconfig']['islarge'], data_config['dataconfig']['trainperc'])

    if controlConfig['validate_study']:

        is_fixed_study = args.isFixed
        new_neurons = args.newNeurons
        has_idn = config['augconfig']['has_idn']
        if has_idn:

            idn_perc = config['augconfig']['idn_perc']

            train_perc = train_test
            if aug_name != 'VECGAN':
                pickle_loc = './output/' + aug_name + '/validation/' + data_name + \
                             'studyparams' +'_TRAIN_'+str(train_test)+'_IDN_'+str(idn_perc)+'.p'
            else:
                pickle_loc = './output/' + aug_name + '/validation/' + data_name + \
                             'studyparams' +config['augconfig']['gradmode']+'_TRAIN_'+str(train_test)+'_IDN_'+str(idn_perc)+'.p'

            best_params = pickle.load(open(pickle_loc, 'rb'))  # Update based on best average parameters
            config, sup_config = assignParams(config, sup_config, data_config, best_params)



        elif not is_fixed_study:

            if aug_name != 'VECGAN':
                pickle_loc = './output/' + aug_name + '/validation/' + data_name + 'studyparams' + '.p'
            else:
                pickle_loc = './output/' + aug_name + '/validation/' + data_name + 'studyparams' + grad_mode + '.p'

            best_params = pickle.load(open(pickle_loc, 'rb'))  # Update based on best average parameters
            config, sup_config = assignParams(config, sup_config, data_config, best_params)

        elif (new_neurons is not None) and ("]" in new_neurons):
            new_neurons = new_neurons.strip('][').split(',')
            new_neurons = [int(i) for i in new_neurons]
            config = assignNeurons(config, new_neurons)
            logger.info("Changed to new neurons: {}".format(new_neurons))

        logger.info("Running workers, going quiet for a while...")
        logger.info("Validating the {} dataset...".format(data_config['dataconfig']['dataname']))
        seeds = [(i, aug_name, config, sup_config, data_config, do_mem_study) for i in range(2)]
        results = []
        for i in range(len(seeds)):
            results.append(validate_data(*seeds[i]))
        # with Pool(processes=5) as p:
        #     results = p.starmap(validate_data, seeds)

        class1_mean_acc = mean([i[0][0] for i in results])
        class2_mean_acc = mean([i[0][1] for i in results])
        class3_mean_acc = mean([i[0][2] for i in results])
        class1_mean_f1 = mean([i[1][0] for i in results])
        class2_mean_f1 = mean([i[1][1] for i in results])
        class3_mean_f1 = mean([i[1][2] for i in results])

        if not data_config['dataconfig']['islarge']:
            leaner1 = 'Decision Tree'
            learner2 = 'Boosted Tree'
            learner3 = 'Gaussian Process'
        else:
            leaner1 = 'Log Reg'
            learner2 = 'SGD Classifier'
            learner3 = 'Passive Agressive'

        logger.info("Validation Complete!")
        logger.info("Augmentation: {}, Data:{}, Mean Tree Class. Acc.:{}, "
                    "Mean Boosted Tree Acc.:{}, Mean GP Acc.:{}\n".format(aug_name,
                                                                          data_name,
                                                                          class1_mean_acc,
                                                                          class2_mean_acc,
                                                                          class3_mean_acc,))
        logger.info(
            "Augmentation: {}, Data:{}, Mean Tree Class. F1:{}, "
            "Mean Boosted Tree F1:{}, Mean GP F1:{}, Train/Test Split: {}\n".format(
                aug_name,
                data_name,
                class1_mean_f1,
                class2_mean_f1,
                class3_mean_f1,
                train_test))

        if not do_mem_study:
            train_test = data_config['dataconfig']['trainperc']
            train_test_str = str(train_test)
            train_str = str(int(100 * train_test))

            if is_fixed_study:

                if 'ganconfig' in config.keys():
                    neurons = str(config['ganconfig']['gen']['layer_nodes']).replace(' ','')
                else:
                    neurons = str(None)

                if aug_name != 'VECGAN':
                    file_loc = './output/fixedParameters/' + aug_name + '_' +data_name + 'results_SPLIT_' + train_str + '_NEURONS_'+ neurons+'.txt'
                else:
                    file_loc = './output/fixedParameters/' + aug_name + '_'+ data_name + 'results' + \
                               config['augconfig']['gradmode'] +'_SPLIT_'+ train_str +'_NEURONS_'+ neurons + '.txt'

            else:

                if has_idn:

                    if aug_name != 'VECGAN':

                        file_loc = './output/' + aug_name + '/validation/' + data_name + 'results' + \
                                   '_SPLIT_'+ train_str +'_IDN_'+str(idn_perc)+'_STUDYRESULTS.txt'

                    else:

                        file_loc = './output/' + aug_name + '/validation/' + data_name + 'results' + \
                                   config['augconfig']['gradmode'] +'_SPLIT_'+ train_str +'_IDN_'+str(idn_perc)+'_STUDYRESULTS.txt'


                elif aug_name != 'VECGAN':
                    file_loc = './output/' + aug_name + '/validation/' + data_name + 'results_SPLIT_' + train_str + '_STUDYRESULTS.txt'
                else:
                    file_loc = './output/' + aug_name + '/validation/' + data_name + 'results' + \
                               config['augconfig']['gradmode'] +'_SPLIT_'+ train_str +'_STUDYRESULTS.txt'

            with open(file_loc, 'w+') as f:
                f.write("Augmentation: {}, Data:{}, Mean Tree Class. Acc.:{}, "
                        "Mean Boosted Tree Acc.:{}, Mean GP Acc.:{}, Train/Test Split: {}\n".format(aug_name,
                                                                                                    data_name,
                                                                                                    class1_mean_acc,
                                                                                                    class2_mean_acc,
                                                                                                    class3_mean_acc,
                                                                                                    train_test))
                f.write(
                    "Augmentation: {}, Data:{}, Mean Tree Class. F1:{}, "
                    "Mean Boosted Tree F1:{}, Mean GP F1:{}, Train/Test Split: {}\n".format(
                        aug_name,
                        data_name,
                        class1_mean_f1,
                        class2_mean_f1,
                        class3_mean_f1,
                        train_test))

                mean_acc = mean([class1_mean_acc,class2_mean_acc,class3_mean_acc])
                mean_f1 = mean([class1_mean_f1,class2_mean_f1,class3_mean_f1])

                f.write("Reporting Results for table-> Avg. Acc.: {:.1%}, Avg. f1: {:.1%}\n".format(mean_acc, mean_f1))

                f.close()

            if is_fixed_study:
                if aug_name != 'VECGAN':
                    file_loc = './output/fixedParameters/' + aug_name + data_name + 'results_SPLIT_' + train_str + '_NEURONS_'+ neurons+'.p'
                else:
                    file_loc = './output/fixedParameters/' + aug_name +  data_name + 'results' + \
                               config['augconfig']['gradmode'] +'_SPLIT_'+ train_str + '_NEURONS_' + neurons+ '.p'

                pickle.dump([class1_mean_acc, class2_mean_acc, class3_mean_acc], open(file_loc,'wb'))
            logger.info("Saving results...")


