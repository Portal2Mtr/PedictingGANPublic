"""Vanilla GAN Class

Contains the torch implementation of the VANILLA GAN for data augmentation with batch processing.

"""
import sys

sys.path.append('./source/Augments/')
sys.path.append('./Datasets/')
import torch
import copy
import numpy as np
from sklearn.metrics import accuracy_score
import sklearn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import time
import pickle
import torch.nn
import optuna
from statistics import mean
# noinspection PyUnresolvedReferences
from _trainer_base import TrainerBase

from sklearn.model_selection import train_test_split

DEVICE = torch.device("cpu")
logger = logging.getLogger(__name__)


class VanillaGANTrainer(TrainerBase):
    """
    Class for managing training for the BASIC GAN.


    """

    def __init__(self, config):
        super().__init__(seed=config['augconfig']['seed'])
        torch.set_default_tensor_type('torch.DoubleTensor')
        self.config = config
        self.batch_size_div = self.config['ganconfig']['batchsizediv']
        self.batch_size = 0
        return

    def init_data(self, trainPerc=None):
        """
        Wrapper function for initializing the class dataset
        :return:
        :rtype:
        """
        data_name = self.config['augconfig']['dataset']
        self.dataset = self.get_dataset(data_name,trainPerc=trainPerc)
        self.num_features = self.dataset.num_features

    def init_networks(self, trial=None, num_features=None):
        """
        Initializes the MLP networks for training
        :param num_features: Number of features for dataset
        :return:
        """

        if num_features is None:
            self.create_networks(self.dataset.num_features, self.config)
        else:
            self.create_networks(num_features, self.config, trial)

    def conduct_training(self, trial=None,classes=None):
        """
        Conducts trainign for individual testing or parameter study.
        :param trial: Optuna trial
        :return:
        """

        if trial is None:
            gen_loss, disc_loss = self.train()
            self.save_models(gen_loss, disc_loss)
            # pickle.dump([gen_loss, disc_loss],
            #             open(self.config['outputconfig']['outputdir'] + '/VecGANtrainingloss.p', 'wb'))
        else:

            if classes is None:
                gen_loss, disc_loss = self.train(trial)
            else:
                # if self.config['augconfig']['has_idn'] or type(classes[0]) == list:
                #     gen_loss, disc_loss = self.train(trial,classes)
                # else:
                #     gen_loss, disc_loss = self.train(trial)

                gen_loss, disc_loss = self.train(trial, classes)

            return gen_loss, disc_loss, trial, self.generator, self.discriminator

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

        lr_gen = self.config['ganconfig']['gen']['genadam']
        lr_disc = self.config['ganconfig']['disc']['discadam']
        self.generator = VanillaGAN("Generator", num_features, config, lr=lr_gen, set_gen=True)
        self.discriminator = VanillaGAN("Discriminator", num_features, config, lr=lr_disc, set_disc=True)

    def train_step(self, real_data, labels):
        """
        Training step in pytorch for BASIC GAN.
        :param data: Batch input data
        :return:
        """
        self.discriminator.zero_grad()
        self.generator.zero_grad()
        # Random noise for generator to create synthetic data from
        noise = torch.rand(len(real_data), self.num_features)
        labels = labels.view(len(labels), 1)
        noise = torch.cat((noise, labels), 1)

        synth_sample = self.generator(noise)

        synth_sample = torch.cat((synth_sample,labels),1)

        # Evaluate with discriminator the validity of data samples

        real_label = self.discriminator(real_data)
        fake_label = self.discriminator(synth_sample)
        disc_loss = self.discriminator.loss(real_label, fake_label)
        disc_loss.backward(retain_graph=True)
        self.discriminator.optim.step()
        # Detach vars to remove disc grads
        real_label = real_label.detach()
        real_label.requires_grad = True
        fake_label = fake_label.detach()
        fake_label.requires_grad = True
        # Update generator

        gen_loss = self.generator.loss(real_label, fake_label)
        gen_loss.backward()
        self.generator.optim.step()

        # Return loss for plotting
        return gen_loss, disc_loss

    def train(self, trial=None, classes=None):
        """
        Wrapper function to manage entire training cycle of BASIC GAN.
        :param trial: Optuna trail
        :return:
        :rtype:
        """

        gan_train_data = [(self.dataset[i]) for i in self.dataset.train_idxs]
        for idx in range(len(gan_train_data)):
            temp = gan_train_data[idx][0].tolist()
            temp.append(gan_train_data[idx][1])
            gan_train_data[idx] = (torch.tensor(temp, requires_grad=True),gan_train_data[idx][1])

        self.batch_size = int(len(gan_train_data) / self.batch_size_div)
        data_loader = DataLoader(gan_train_data, batch_size=self.batch_size, shuffle=True)
        gan_train_config = self.config['augconfig']
        epochs = gan_train_config['epochs']

        gen_loss_array = []
        disc_loss_array = []

        self.generator.zero_grad()
        self.discriminator.zero_grad()
        start = time.time()
        mean_gen = 0
        mean_disc = 0
        update_disc_loss = []
        update_gen_loss = []
        for epoch in range(epochs):

            batch_gen_loss = []
            batch_disc_loss = []
            for i, data in enumerate(data_loader):
                real_data, labels = data
                gen_loss, disc_loss = self.train_step(real_data, labels)
                batch_gen_loss.append(gen_loss)
                batch_disc_loss.append(disc_loss)

            sum_gen = sum([float(i) for i in batch_gen_loss])
            sum_disc = sum([float(i) for i in batch_disc_loss])
            gen_loss_array.append(sum_gen)
            disc_loss_array.append(sum_disc)
            update_disc_loss.append(sum_disc)
            update_gen_loss.append(sum_gen)

            # Status Update
            if (epoch + 1) % 100 == 0:
                logger.info('Time for epoch {} is {} sec, Disc. Loss={} '
                            'Gen. loss={}'.format(epoch + 1,
                                                  time.time() - start, update_disc_loss[-1], update_gen_loss[-1]))
                start = time.time()

            if classes is not None:
                if (epoch+1) % 501 == 0 and (self.config['augconfig']['has_idn'] or type(classes[0]) != list) and trial is not None:
                    acc = self.early_report(classes)
                    trial.report(acc,epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    else:
                        logger.info("Progress looks good with {} mean accuracy...".format(acc))


        if trial is not None:
            return mean_gen, mean_disc
        else:

            return gen_loss_array, disc_loss_array

    def save_models(self, gen_loss, disc_loss, trial=None):
        """
        Saves the generated models from testing.
        :param gen_loss: Generator loss array
        :param disc_loss: Discriminator loss array
        :param trial: Oputna trial
        :return: Outputs pickle file with networks and loss arrays
        """

        if trial is None:
            output_config = self.config['outputconfig']
            torch.save(self.generator.state_dict(), output_config['outputdir'] + 'genweights'+str(self.seed)+'.p')
            torch.save(self.discriminator.state_dict(), output_config['outputdir'] + 'discweights'+str(self.seed)+'.p')
            pickle.dump([gen_loss, disc_loss], open(output_config['picklename'], 'wb'))
            logger.info("Saved Models!")

        # Skip saving for parameter study
        return trial

    def load_training(self, get_loss=True, trial=None,gen=None,disc=None):
        """
        Loads the training models and results from training for networks.
        :param get_loss: Optional recover the loss from training
        :param trial: Optuna trial
        :return:
        """

        if trial is None:
            output_config = self.config['outputconfig']

            self.generator.load_state_dict(torch.load(output_config['outputdir'] + 'genweights'+str(self.seed)+'.p'))
            self.discriminator.load_state_dict(torch.load(output_config['outputdir']+'discweights'+str(self.seed)+'.p'))
            gen_loss, disc_loss = pickle.load(open(output_config['picklename'], 'rb'))
            if get_loss:
                return gen_loss, disc_loss
        else:
            self.generator = gen
            self.discriminator = disc
            return trial

    def synth_samples(self, aug_config, total_dataset):
        """
        Sythesize samples for the BASIC GAN Training with class dataset.
        Args:
            aug_config (): Augmentation config
            total_dataset (): total dataset used for synthesis with supervised classifier

        Returns: Augmentated dataset with synthetic samples.

        """

        # Load training data and get num of samples to generate
        synth_samples = []
        class_labels = []
        num_features = total_dataset.num_features
        with torch.no_grad():
            for class_lab, num_gen in zip(total_dataset.class_array, total_dataset.num_gen_array):
                for i in range(num_gen):
                    noise = torch.rand(1, num_features)
                    noise = torch.cat((noise, torch.tensor(class_lab).view(1,1)), 1)
                    synth_sample = self.generator.forward(noise)
                    synth_samples.append(synth_sample[0])
                    class_labels.append(torch.tensor(class_lab))

        synth_train_X, synth_test_X, synth_train_Y, synth_test_Y = train_test_split(synth_samples,
                                                                                    class_labels,
                                                                                    train_size=total_dataset.train_split)

        train_synth_data = [(x,y) for x,y in zip(synth_train_X, synth_train_Y)]
        test_synth_data = [(x,y) for x,y in zip(synth_test_X, synth_test_Y)]

        return train_synth_data, test_synth_data

    def early_report(self,classes):
        train_data, test_data = self.synth_samples(None, self.dataset)

        early_data = copy.deepcopy(self.dataset)
        copy_classes = copy.deepcopy(classes)

        if type(classes[0]) != list:
            use_mlp = True
            mlp_class = copy_classes[0]
        else:
            use_mlp = False
            class1,class2,class3 = copy_classes[0]

        for data in train_data:
            early_data.all_data.append(data[0])
            early_data.all_labels.append("{}_{}".format(data[1].item(), early_data.synth_label))
            early_data.train_idxs.append(len(early_data.all_data) - 1)

        for data in test_data:
            early_data.all_data.append(data[0])
            early_data.all_labels.append("{}_{}".format(data[1].item(), early_data.synth_label))
            early_data.test_idxs.append(len(early_data.all_data) - 1)


        logger.info("Checking progress...")
        early_data.overwrite_synth_labels = True
        training_dataset = [(early_data[i]) for i in early_data.train_idxs]
        for idx, data in enumerate(training_dataset):
            temp_data = training_dataset[idx][0].tolist()
            temp_label = int(training_dataset[idx][1])
            training_dataset[idx] = [temp_data, temp_label]

        train_data = [training_dataset[i][0] for i in range(len(training_dataset))]
        train_labels = [training_dataset[i][1] for i in range(len(training_dataset))]

        early_data.overwrite_synth_labels = False
        test_data = [early_data[i] for i in early_data.test_idxs]

        if use_mlp:
            mlp_train = [(torch.tensor(data),label) for data,label in zip(train_data,train_labels)]
            data_loader = torch.utils.data.DataLoader(mlp_train,
                                                      shuffle=True,
                                                      batch_size=len(train_data)//4)
            mlp_class.train_loop(data_loader)

            pred_labels = []

            for data,label in test_data:
                pred_labels.append(mlp_class.pred_data(data))

            true_labels = [test_data[i][1] for i in range(len(test_data))]
            remain_idxs = []
            for idx, label in enumerate(true_labels):
                if type(label) != str: # Synth label
                    remain_idxs.append(idx)
                if type(label) == str:
                    # continue
                    true_labels[idx] = int(label[0])
                    remain_idxs.append(idx)

            show_test_labels = np.array([true_labels[i] for i in remain_idxs])

            show_class1_labels = np.array([pred_labels[i] for i in remain_idxs])
            acc_score = accuracy_score(show_test_labels, show_class1_labels)

            return acc_score

        # Report with classifiers (small only)

        for idx, data in enumerate(test_data):
            temp_data = test_data[idx][0].tolist()
            temp_label = test_data[idx][1]
            test_data[idx] = [temp_data, temp_label]

        train_data, train_labels = sklearn.utils.shuffle(train_data,train_labels)
        class1.fit(train_data, train_labels)
        class2.fit(train_data, train_labels)
        class3.fit(train_data, train_labels)

        class1_pred_labels = []
        class2_pred_labels = []
        class3_pred_labels = []
        for data, label in test_data:
            data = np.array(data)
            data = data.reshape(1, -1)
            class1_pred_labels.append(int(class1.predict(data)))
            class2_pred_labels.append(int(class2.predict(data)))
            class3_pred_labels.append(int(class3.predict(data)))

        true_labels = [test_data[i][1] for i in range(len(test_data))]
        remain_idxs = []
        for idx, label in enumerate(true_labels):
            if type(label) != str: # Synth label
                remain_idxs.append(idx)
            if type(label) == str:
                continue

        show_test_labels = np.array([true_labels[i] for i in remain_idxs])
        show_class1_labels = np.array([class1_pred_labels[i] for i in remain_idxs])
        show_class2_labels = np.array([class2_pred_labels[i] for i in remain_idxs])
        show_class3_labels = np.array([class3_pred_labels[i] for i in remain_idxs])
        acc_score_class1 = accuracy_score(show_test_labels, show_class1_labels)
        acc_score_class2 = accuracy_score(show_test_labels, show_class2_labels)
        acc_score_class3 = accuracy_score(show_test_labels, show_class3_labels)

        acc_scores = [acc_score_class1, acc_score_class2, acc_score_class3]
        return mean(acc_scores)



class VanillaGAN(torch.nn.Module):
    """
    Class for managing BASIC GAN networks.
    """

    def __init__(self, name, n_inputs, config,lr=None, set_gen=False, set_disc=False):
        super(VanillaGAN, self).__init__()
        self.aug_name = config['augconfig']['augname']
        # store the parameters of network
        gan_config = config['ganconfig']
        self.net_name = name
        self.is_gen = set_gen
        self.is_disc = set_disc
        self.n_inputs = n_inputs
        self.bin_loss = torch.nn.BCELoss()
        self.layers = []

        if self.is_gen:
            self.config = gan_config['gen']
            self.gen_layers()
        elif self.is_disc:
            self.config = gan_config['disc']
            self.disc_layers()

        self.inner_act = F.leaky_relu

        # self.inner_act = F.relu

        self.output_act = F.sigmoid

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.drop = torch.nn.Dropout(gan_config['dropout'])

    def __call__(self, x):
        return self.forward(x)

    def gen_layers(self):
        """
        Generator layers
        Returns:

        """

        self.layers = torch.nn.ModuleList()
        layer_weights = [self.n_inputs + 1]
        layer_weights.extend(self.config['layer_nodes'])
        layer_weights.append(self.n_inputs)

        # Declare layer-wise weights and biases
        for i in range(len(layer_weights) - 1):
            self.layers.append(
                torch.nn.Linear(layer_weights[i], layer_weights[i+1])
            )

    def disc_layers(self):
        """
        Discriminator layers
        Returns:

        """
        self.layers = torch.nn.ModuleList()
        layer_weights = [self.n_inputs + 1] # include class label
        layer_weights.extend(self.config['layer_nodes'])
        layer_weights.append(1)

        # Declare layer-wise weights and biases
        for i in range(len(layer_weights) - 1):
            self.layers.append(
                torch.nn.Linear(layer_weights[i], layer_weights[i + 1])
            )

    def forward(self, x):
        """
        Forward pass of the network
        Args:
            x (): Data input

        Returns:

        """

        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if i != len(self.layers)-1:
                out = self.inner_act(out)
                out = self.drop(out)

        y = self.output_act(out)

        return y

    def loss(self, y_true, y_pred):
        """
        Calculates loss for the BASIC GAN
        Args:
            y_true ():
            y_pred ():

        Returns:

        """

        if self.is_gen:
            return self.bin_loss(y_pred, torch.ones_like(y_pred))
        elif self.is_disc:
            real_loss = self.bin_loss(y_true, torch.ones_like(y_true))
            fake_loss = self.bin_loss(y_pred, torch.zeros_like(y_pred))
            total_loss = real_loss + fake_loss
            return total_loss

