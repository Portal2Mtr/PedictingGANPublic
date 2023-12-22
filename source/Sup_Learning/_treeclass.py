"""Tree management class

Class for managing tree classifiers

"""

import sys

import sklearn.utils

sys.path.append('./Augments/')
sys.path.append('./Datasets/')
import logging
import torch
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score,fbeta_score
from sklearn.linear_model import SGDClassifier,PassiveAggressiveClassifier

from _trxn_mlp import TrxnClassifier

# noinspection PyUnresolvedReferences
from _trainer_base import TrainerBase
# noinspection PyUnresolvedReferences
from _smote import SmoteTrainer
# noinspection PyUnresolvedReferences
from _vanilla_gan import VanillaGANTrainer
# noinspection PyUnresolvedReferences
from _wgan import WassGanTrainer
# noinspection PyUnresolvedReferences
from _vecgan import VecGanTrainer
# noinspection PyUnresolvedReferences
from _rcgan import RCGanTrainer
# # noinspection PyUnresolvedReferences
# from _ctgan import CTGanTrainer
# noinspection PyUnresolvedReferences
from _bitcoinlate2012 import BitcoinDataset
# noinspection PyUnresolvedReferences
# from _ctgan import CTGanTrainer
logger = logging.getLogger(__name__)
use_cuda = torch.cuda.is_available()
device = torch.device("cpu")


class TreeTrainer(TrainerBase):

    def __init__(self, sup_config, aug_config):
        super().__init__(seed=0)
        self.class1 = None
        self.class2 = None
        self.class3 = None
        self.mlp_class = None
        self.dataset = None
        self.learn_rate = None
        self.min_split_dt = None
        self.min_leaf_dt = None
        self.max_depth_dt = None
        torch.set_default_tensor_type('torch.DoubleTensor')
        self.sup_config = sup_config
        self.aug_config = aug_config
        self.large_dataset = False
        self.minibatch = 1000
        self.use_mlp = sup_config['supconfig']['use_mlp']

    def init_params(self):
        """
        Initialize the parameters for the tree classifiers
        Returns:

        """

        self.max_depth_dt = self.sup_config['supconfig']['maxdepthdt']
        self.min_leaf_dt = self.sup_config['supconfig']['minleafdt']
        self.min_split_dt = self.sup_config['supconfig']['minsplitdt']

        self.learn_rate = self.sup_config['supconfig']['learnrate']
        self.n_est = self.sup_config['supconfig']['n_est']
        self.max_depth_gb = self.sup_config['supconfig']['maxdepthgb']
        self.min_leaf_gb = self.sup_config['supconfig']['minleafgb']
        self.min_split_gb = self.sup_config['supconfig']['minsplitgb']

        self.svm_c = self.sup_config['supconfig']['svm_c']
        self.seed = self.sup_config['supconfig']['seed']
        self.alpha1 = self.sup_config['supconfig']['alpha1']
        self.alpha2 = self.sup_config['supconfig']['alpha2']
        self.alpha3 = self.sup_config['supconfig']['alpha3']
        self.passagglearn = self.sup_config['supconfig']['passagglearn']

    def init_data(self, trainPerc=None):
        """
        Initilaze the dataset used for training/testing
        Returns:

        """

        data_name = self.aug_config['augconfig']['dataset']
        self.dataset = self.get_dataset(data_name, trainPerc,
                                        self.aug_config['augconfig']['has_idn'],
                                        self.aug_config['augconfig']['idn_perc'])
        self.large_dataset = self.dataset.islarge

    def init_networks(self):
        """
        Wrapper for create_sup_learner
        Returns:

        """
        if not self.use_mlp:
            classes = self.create_sup_learner()
        else:
            classes = self.create_mlp()
        return [classes]

    def create_mlp(self):

        mlp_conf = self.sup_config['supconfig']['mlp']

        mlp_conf['n_inputs'] = self.dataset.num_features

        self.mlp_class = TrxnClassifier(mlp_conf)

        return self.mlp_class

    def create_sup_learner(self):
        """
        Creates the tree classifiers from sklearn
        Returns:

        """

        if not self.large_dataset:

            # Smaller classifiers for small datasets
            # Decision tree classifier
            self.class1 = DecisionTreeClassifier(random_state=self.seed)

            # Gradient Boosted Classifier
            self.class2 = GradientBoostingClassifier(random_state=self.seed,
                                                     learning_rate=self.learn_rate,
                                                     n_estimators=self.n_est)
            # Gaussian process classifier
            self.class3 = GaussianProcessClassifier(kernel=None, random_state=self.seed, n_jobs=-1)

        else:
            # Incremental classifiers for large datasets
            # Logistic regression
            self.class1 = SGDClassifier(
                loss='log',
                penalty='l2',
                shuffle=True,
                random_state=self.seed,
                n_jobs=-1,
                learning_rate='optimal',
                alpha=self.alpha1
            )
            # Single layer SGD
            self.class2 = SGDClassifier(
                loss='perceptron',
                penalty='l2',
                shuffle=True,
                random_state=self.seed,
                n_jobs=-1,
                learning_rate='optimal',
                alpha=self.alpha2
            )
            # Passive agressive
            self.class3 = PassiveAggressiveClassifier(
                C=self.passagglearn,
                shuffle=True,
                random_state=self.seed,
                loss='squared_hinge'
            )

        return [self.class1, self.class2, self.class3]

    def init_augment(self, trial=None, param=None, gen=None, disc=None):
        """
        Initializes the augmentation techniques
        Args:
            trial (): Optuna trial object
            param ():

        Returns:

        """
        # Init augmentation method
        # Smote doesn't need initialization
        aug_name = self.aug_config['augconfig']['augname']

        if aug_name == 'VANILLAGAN':
            self.augment = VanillaGANTrainer(self.aug_config)
            if trial is None:
                self.augment.init_networks(num_features=self.dataset.num_features)
                self.augment.load_training(get_loss=False)
            else:
                self.augment.init_networks(trial=trial, num_features=self.dataset.num_features)
                trial = self.augment.load_training(get_loss=False, trial=trial, gen=gen, disc=disc)

        elif aug_name == 'WGAN':
            self.augment = WassGanTrainer(self.aug_config)
            if trial is None:
                self.augment.init_networks(num_features=self.dataset.num_features)
                self.augment.load_training(get_loss=False)
            else:
                self.augment.init_networks(trial=trial, num_features=self.dataset.num_features)
                trial = self.augment.load_training(get_loss=False, trial=trial, gen=gen, disc=disc)

        elif aug_name == 'SMOTE':
            if param is not None:
                self.aug_config['neighbors'] = param
            self.augment = SmoteTrainer(self.aug_config)
            # No need to train SMOTE

        elif aug_name == 'VECGAN':
            self.augment = VecGanTrainer(self.aug_config)

            self.augment.init_networks(self.dataset.num_features, self.dataset.class_array)
            if trial is None:
                _ = self.augment.load_training(get_loss=False, get_discs=True)
            else:
                _ = self.augment.load_training(get_loss=False, get_discs=True, trial=trial, gen=gen, disc=disc)


        elif aug_name == 'RCGAN':
            self.augment = RCGanTrainer(self.aug_config)

            self.augment.init_networks(self.dataset.num_features,self.dataset.numClasses, self.dataset.class_array)

    def gen_synth(self):
        """
        Generate synthetic samples for the appropriate augmentation technique
        Returns:

        """
        logger.info("Generating Synthetic data!")
        train_synth_data, test_synth_data = self.augment.synth_samples(self.aug_config, self.dataset)
        self.dataset.append_train_test_synth(train_synth_data, test_synth_data,
                                             self.augment.config['augconfig']['augname'])

    def get_file_id(self):
        """
        Returns te file identifier for saving results
        Returns: File ID string

        """

        if isinstance(self.dataset, BitcoinDataset):
            reduce_size = self.dataset.config['dataconfig']['reducesize']
            file_id = 'BITCOIN'
            if reduce_size != 0:
                file_id += str(reduce_size)
            else:
                if self.dataset.config['dataconfig']['createLate2012']:
                    file_id += 'fullLate2012'

                if self.dataset.config['dataconfig']['createEntire']:
                    file_id += 'fullEntire'

        else:
            file_id = self.sup_config['supconfig']['dataset']

        file_data_id = '_' + self.aug_config['augconfig']['augname'] + file_id + '_'

        return file_data_id

    def conduct_training(self, trial=None, save_synth=False):
        """
        Conducts training for the tree classifiers
        Args:
            trial (): Optuna trial object

        Returns:

        """

        # Train/test loop for k-neighbors classifier
        if self.aug_config['augconfig']['augname'] != 'CONTROL':
            self.gen_synth()





        # Train with training dataset first

        logger.info("Starting training...")
        self.dataset.overwrite_synth_labels = True
        training_dataset = [(self.dataset[i]) for i in self.dataset.train_idxs]
        for idx, data in enumerate(training_dataset):
            temp_data = training_dataset[idx][0].tolist()
            temp_label = int(training_dataset[idx][1])
            training_dataset[idx] = [temp_data, temp_label]

        train_data = [training_dataset[i][0] for i in range(len(training_dataset))]
        train_labels = [training_dataset[i][1] for i in range(len(training_dataset))]

        self.dataset.overwrite_synth_labels = False
        test_data = [self.dataset[i] for i in self.dataset.test_idxs]

        for idx, data in enumerate(test_data):
            temp_data = test_data[idx][0].tolist()
            temp_label = test_data[idx][1]
            test_data[idx] = [temp_data, temp_label]


        if self.aug_config['augconfig']['augname'] == 'CONTROL':
            if self.sup_config['supconfig']['vecganoverride']['override']:

                test_data = [self.dataset[i] for i in self.dataset.test_idxs]

                true_labels = [test_data[i][1] for i in range(len(test_data))]
                remain_idxs = []
                for idx, label in enumerate(true_labels):
                    if type(label) != str:  # Synth label
                        remain_idxs.append(idx)

                    if type(label) == str:
                        continue # Remove synthetic data
                        # true_labels[idx] = int(label[0])
                        # remain_idxs.append(idx)

                # ETHFRAUD with mlp only!!!
                mode = self.sup_config['supconfig']['vecganoverride']['mode']
                file_name = self.sup_config['supconfig']['vecganoverride'][mode]
                training_data, _ = pickle.load(open(file_name,'rb'))
                self.dataset.append_train_test_synth(training_data,None)

                self.dataset.overwrite_synth_labels = True
                training_dataset = [(self.dataset[i]) for i in self.dataset.train_idxs]
                for idx, data in enumerate(training_dataset):
                    temp_data = training_dataset[idx][0].tolist()
                    temp_label = int(training_dataset[idx][1])
                    training_dataset[idx] = [temp_data, temp_label]


                train_data = [training_dataset[i][0] for i in range(len(training_dataset))]
                train_labels = [training_dataset[i][1] for i in range(len(training_dataset))]


                mlp_train = [(torch.tensor(data), label) for data, label in zip(train_data, train_labels)]
                data_loader = torch.utils.data.DataLoader(mlp_train,
                                                          shuffle=True,
                                                          batch_size=len(mlp_train))
                self.mlp_class.train_loop(data_loader)
                pred_labels = []

                for data, label in test_data:
                    pred_labels.append(self.mlp_class.pred_data(data))


                show_test_labels = np.array([test_data[i][1] for i in range(len(test_data))])

                show_class1_labels = np.array([pred_labels[i] for i in remain_idxs])
                acc_score = accuracy_score(show_test_labels, show_class1_labels)

                return acc_score,None, trial

        if self.large_dataset:
            # Minibatches for large datasets
            if self.use_mlp:
                mlp_train = [(torch.tensor(data), label) for data, label in zip(train_data, train_labels)]
                data_loader = torch.utils.data.DataLoader(mlp_train,
                                                          shuffle=True,
                                                          batch_size=len(mlp_train))
                self.mlp_class.train_loop(data_loader)

            else: # Use regular 3 classifiers
                classes = self.dataset.class_array
                for idx in range(0, len(train_data), self.minibatch):
                    batch_data = np.array(train_data[idx:idx+self.minibatch], dtype=np.float)
                    batch_labels = np.array(train_labels[idx:idx+self.minibatch])
                    batch_data, batch_labels = sklearn.utils.shuffle(batch_data, batch_labels)
                    self.class1.partial_fit(batch_data, batch_labels, classes=classes)
                    self.class2.partial_fit(batch_data, batch_labels, classes=classes)
                    self.class3.partial_fit(batch_data, batch_labels, classes=classes)

        else:

            train_data, train_labels = sklearn.utils.shuffle(train_data, train_labels,random_state=0)
            self.class1.fit(train_data, train_labels)
            self.class2.fit(train_data, train_labels)
            self.class3.fit(train_data, train_labels)

        if self.use_mlp:
            pred_labels = []

            for data, label in test_data:
                pred_labels.append(self.mlp_class.pred_data(data))

        else:
            class1_pred_labels = []
            class2_pred_labels = []
            class3_pred_labels = []
            for data, label in test_data:
                data = np.array(data)
                data = data.reshape(1, -1)
                class1_pred_labels.append(int(self.class1.predict(data)))
                class2_pred_labels.append(int(self.class2.predict(data)))
                class3_pred_labels.append(int(self.class3.predict(data)))

        true_labels = [test_data[i][1] for i in range(len(test_data))]
        remain_idxs = []
        for idx, label in enumerate(true_labels):
            if type(label) != str:  # Synth label
                remain_idxs.append(idx)

            if type(label) == str:
                continue # Remove synthetic data
                # true_labels[idx] = int(label[0])
                # remain_idxs.append(idx)

        show_test_labels = np.array([true_labels[i] for i in remain_idxs])

        if self.use_mlp:

            show_class1_labels = np.array([pred_labels[i] for i in remain_idxs])
            acc_score = accuracy_score(show_test_labels, show_class1_labels)
            class1_f1 = fbeta_score(show_test_labels, show_class1_labels, beta=0.5,average='weighted')
            print("Tree acc. score: {}".format(acc_score))
            print("Tree f1 score: {}".format(class1_f1))

            sys.exit()

            return acc_score,None, trial

        else:

            show_class1_labels = np.array([class1_pred_labels[i] for i in remain_idxs])
            show_class2_labels = np.array([class2_pred_labels[i] for i in remain_idxs])
            show_class3_labels = np.array([class3_pred_labels[i] for i in remain_idxs])
            acc_score_class1 = accuracy_score(show_test_labels, show_class1_labels)
            acc_score_class2 = accuracy_score(show_test_labels, show_class2_labels)
            acc_score_class3 = accuracy_score(show_test_labels, show_class3_labels)
            class1_f1 = fbeta_score(show_test_labels, show_class1_labels, beta=0.5,average='weighted')
            class2_f1 = fbeta_score(show_test_labels, show_class2_labels, beta=0.5,average='weighted')
            class3_f1 = fbeta_score(show_test_labels, show_class3_labels, beta=0.5,average='weighted')

            acc_scores = [acc_score_class1, acc_score_class2, acc_score_class3]
            f1_scores = [class1_f1, class2_f1, class3_f1]

            if trial is None:

                print("Tree acc. score: {}, Boost acc. score: {}, GP acc. score: {}".format(acc_score_class1,
                                                                                            acc_score_class2,
                                                                                            acc_score_class3))
                print("Tree f1 score: {}, Boost f1 score: {}, GP f1 score: {}".format(class1_f1, class2_f1, class3_f1))

                return acc_scores, f1_scores

            else:

                return acc_scores, f1_scores, trial