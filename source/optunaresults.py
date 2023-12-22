"""
    Generates the results from the oputna database.
"""
import optuna
import argparse
import sys
import logging
import yaml
from statistics import mean
import numpy as np
import pickle
import copy

from colorlog import ColoredFormatter

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Config file input')

    parser.add_argument('--augconfig', type=str,
                        default='./config/augconfig/controlconfig.yml',
                        help='Config files for simulations (SMOTE, VANILLAGAN, WGAN, CTGAN, VECGAN)')
    parser.add_argument('--datasetconfig', type=str,
                        default='./config/dataconfig/pimaconfig.yml',
                        help='Dataset')
    parser.add_argument('--gradMode', type=str,
                        default='SGD',
                        help='Dataset')

    parser.add_argument('--trainPerc', type=str,
                        default=None,
                        help='Training Percentage')

    parser.add_argument('--idnPerc', type=str,
                        default=None,
                        help='IDN Percentage')

    args = parser.parse_args()
    aug_config = args.augconfig
    data_config = args.datasetconfig
    grad_mode = args.gradMode

    aug_file_name = copy.copy(aug_config)
    with open(aug_config, "r") as read_file:
        aug_config = yaml.load(read_file, Loader=yaml.FullLoader)

    with open(data_config, "r") as read_file:
        data_config = yaml.load(read_file, Loader=yaml.FullLoader)

    aug_name = aug_config['augconfig']['augname']
    data_name = data_config['dataconfig']['dataname']
    # Setup Optuna
    # Use optuna for parameter study
    logger.info("Setting up parameter study...")
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Sup. Loss

    with open('./output/HOSTS/host.txt') as f:
        lines = f.readlines()
    working_host = lines[0].split(' ')[0]

    directions = ["maximize"]

    if aug_name != 'VECGAN':
        study_name = 'Study' + aug_name + data_name
    else:
        study_name = 'Study' + aug_name + data_name + grad_mode

    train_perc = args.trainPerc
    idn_perc = args.idnPerc

    if idn_perc is not None and aug_config['augconfig']['has_idn']:
        if type(idn_perc) == str:
            if idn_perc.strip() == 'None':
                idn_perc = None

    if train_perc is not None and idn_perc is not None and aug_config['augconfig']['has_idn']:

        study_name = study_name + "_TRAIN_{}_IDN_{}".format(train_perc,idn_perc)
        directions = ["maximize"]


    # Check for mlp classifier

    fileconf = aug_file_name.split("/")

    sup_config = ""
    for idx,substring in enumerate(fileconf):
        if idx == (len(fileconf) - 1):
            sup_config = sup_config + "/sup" + substring
            if aug_name == 'VECGAN':
                sup_config = sup_config[:-10] + str(grad_mode).lower() + sup_config[-10:]
        elif idx == 0:
            sup_config = substring

        elif substring == "augconfig":
            sup_config = sup_config + "/supconfig"
        else:
            sup_config = sup_config + "/" + substring

    with open(sup_config, "r") as read_file:
        sup_config = yaml.load(read_file, Loader=yaml.FullLoader)

    if sup_config['supconfig']['use_mlp']:
        study_name = study_name + "_MLP_MODE_{}".format(sup_config['supconfig']['mlp']['mode'])
        directions = ["maximize"]
        if aug_name == 'CONTROL':

            if sup_config['supconfig']['vecganoverride']['override']:
                study_name = study_name + '_OVERRIDE_VECGAN_' + sup_config['supconfig']['vecganoverride']['mode']

    study = optuna.study.create_study(study_name=study_name,
                                      directions=directions,
                                      storage="postgresql://postgres@{}".format(working_host),
                                      load_if_exists=True)

    if aug_name != 'VECGAN':
        file_loc = './output/' + aug_name + '/validation/' + data_name + 'studyresults'+'.txt'
    else:
        file_loc = './output/' + aug_name + '/validation/' + data_name + 'studyresults' + grad_mode + '.txt'

    if train_perc is not None and idn_perc is not None and aug_config['augconfig']['has_idn']:
        file_loc = file_loc[:-4] + "_TRAIN_{}_IDN_{}".format(train_perc,idn_perc) + file_loc[-4:]

    if sup_config['supconfig']['use_mlp']:
        file_loc = file_loc[:-4] + "_MLP_MODE_{}".format(sup_config['supconfig']['mlp']['mode']) + file_loc[-4:]
        if aug_name == 'CONTROL':

            if sup_config['supconfig']['vecganoverride']['override']:
                file_loc = file_loc[:-4] + '_OVERRIDE_VECGAN_' + sup_config['supconfig']['vecganoverride']['mode'] + file_loc[-4:]

    logger.info("Got results from {} trials!".format(len(study.trials)))

    result_values = [i.values for i in study.best_trials]
    mean_trial_accs = [mean(i[0:3]) for i in result_values]
    hi_idx = int(np.argmax(mean_trial_accs))

    print("Got results from {} trials!\n".format(len(study.trials)))

    with open(file_loc, 'w+') as f:
        f.write("Got results from {} trials!\n".format(len(study.trials)))
        for idx, trial in enumerate(study.best_trials):
            f.write("For Trial {}... Values: {}, Params: {}\n".format(trial.number, trial.values, trial.params))
        f.write("Best Results (Average accuracy)\n")
        bt = study.best_trials
        f.write("For Trial {}... Values: {}, Params: {}\n".format(bt[hi_idx].number,
                                                                  bt[hi_idx].values, bt[hi_idx].params))

    if aug_name != 'VECGAN':
        pickle_loc = './output/' + aug_name + '/validation/' + data_name + 'studyparams' + '.p'
    else:
        pickle_loc = './output/' + aug_name + '/validation/' + data_name + 'studyparams' + grad_mode + '.p'

    if train_perc is not None and idn_perc is not None and aug_config['augconfig']['has_idn']:
        pickle_loc = pickle_loc[:-2] + "_TRAIN_{}_IDN_{}".format(train_perc,idn_perc) + pickle_loc[-2:]

    if sup_config['supconfig']['use_mlp']:
        pickle_loc = pickle_loc[:-2] + "_MLP_MODE_{}".format(sup_config['supconfig']['mlp']['mode']) + pickle_loc[-2:]

        if aug_name == 'CONTROL':

            if sup_config['supconfig']['vecganoverride']['override']:
                pickle_loc = pickle_loc[:-2] + '_OVERRIDE_VECGAN_' + sup_config['supconfig']['vecganoverride']['mode'] + pickle_loc[-2:]

    pickle.dump(bt[hi_idx].params,open(pickle_loc,'wb'))

    logger.info("Saved results!")
