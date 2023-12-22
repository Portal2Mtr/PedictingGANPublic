"""Wasserstein GAN  (WGAN)

Classes for managing training for the WGAN approach

"""

import sys
sys.path.append('./Augments/')
sys.path.append('./Datasets/')
import torch
import optuna
from torch.utils.data import DataLoader
import logging
import time
import pickle
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score
import copy
from statistics import mean
# noinspection PyUnresolvedReferences
from _vanilla_gan import VanillaGANTrainer

DEVICE = torch.device("cpu")
logger = logging.getLogger(__name__)


class WassGanTrainer(VanillaGANTrainer):

    def __init__(self, config):
        super().__init__(config)
        # Start basic training loop
        self.config = config
        return

    def conduct_training(self, trial=None, classes=None):
        """
        Conducts training for the WGAN approach
        Args:
            trial (): Optuna trial object

        Returns:

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
        Creates the gen/disc networks
        Args:
            numFeatures (): Number of dataset features
            config (): Dataset configuration file

        Returns:

        """
        self.generator = WassGan("Generator", num_features, config, set_gen=True)
        self.discriminator = WassGan("Discriminator", num_features, config, set_disc=True)

    def train_step(self, real_data, labels):

        # Random noise for generator to create synthetic data from.

        self.generator.disable_learn()
        self.discriminator.enable_learn()
        for i in range(self.discriminator.num_critic):
            self.discriminator.zero_grad()
            noise = torch.rand(len(real_data), self.num_features)
            labels = labels.view(len(labels), 1)
            noise = torch.cat((noise, labels), 1)
            synth_sample = self.generator(noise)

            synth_sample = torch.cat((synth_sample,labels),1)

            # Evaluate with discriminator the validity of data samples
            real_label = self.discriminator(real_data)
            fake_label = self.discriminator(synth_sample)
            fake_label *= -1

            disc_loss = self.discriminator.loss(real_label, fake_label)
            disc_loss = self.discriminator.backward(disc_loss)

        # Stop discriminator and update generator
        self.generator.enable_learn()
        self.discriminator.disable_learn()
        self.generator.zero_grad()

        # Train generator once
        noise = torch.rand(len(real_data), self.num_features)  # No batch processing
        labels = labels.view(len(labels), 1)
        noise = torch.cat((noise, labels), 1)

        synth_sample = self.generator(noise)
        synth_sample = torch.cat((synth_sample, labels),1)
        fake_label = self.discriminator(synth_sample)
        fake_label *= -1

        gen_loss = torch.mean(fake_label)
        self.generator.backward(gen_loss)
        self.generator.disable_learn()
        self.discriminator.enable_learn()

        return disc_loss, gen_loss

    def train(self, trial,classes=None):
        """
        Training loop for WGAN approach
        Args:
            trial (): Optuna trial object

        Returns:

        """

        # Train on minority samples only
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

        update_disc_loss = []
        update_gen_loss = []
        for epoch in range(epochs):

            self.generator.disable_learn()  # Stop for agd
            self.discriminator.enable_learn()
            batch_gen_loss = []
            batch_disc_loss = []

            for i, data in enumerate(data_loader):
                real_data, labels = data
                disc_loss, gen_loss = self.train_step(real_data, labels)

                batch_disc_loss.append(disc_loss)
                batch_gen_loss.append(gen_loss)

            sum_gen = sum([float(i) for i in batch_gen_loss])
            sum_disc = sum([float(i) for i in batch_disc_loss])
            update_disc_loss.append(sum_disc)

            if type(sum_gen) == int and len(update_gen_loss) > 0:
                update_gen_loss.append(update_gen_loss[-1])

            else:
                update_gen_loss.append(sum_gen)

            # Status Update
            if (epoch + 1) % 100 == 0:
                logger.info('Time for epoch {} is {} sec, Disc. Loss={} '
                            'Gen. loss={}'.format(epoch + 1,
                                                  time.time() - start, update_disc_loss[-1], update_gen_loss[-1]))
                start = time.time()

            if classes is not None:
                if (epoch+1) % 501 == 0 and trial is not None:
                    # if (epoch+1) % 501 == 0 and (self.config['augconfig']['has_idn'] or type(classes[0]) != list) and trial is not None:
                    acc = self.early_report(classes)
                    trial.report(acc,epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    else:
                        logger.info("Progress looks good with {} mean accuracy...".format(acc))

        if trial is not None:
            return sum_gen, sum_disc
        else:
            return gen_loss_array, disc_loss_array


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

class WassGan(torch.nn.Module):

    def __init__(self, name, n_inputs, config, set_gen=False, set_disc=False):

        super(WassGan, self).__init__()
        # store the parameters of network
        self.aug_name = config['augconfig']['augname']
        gan_config = config['ganconfig']
        self.net_name = name
        self.is_gen = set_gen
        self.is_disc = set_disc
        self.n_inputs = n_inputs
        self.layers = []

        if self.is_gen:
            self.config = gan_config['gen']
            self.gen_layers()
        elif self.is_disc:
            self.config = gan_config['disc']

            self.clipper = WeightClipper(self.config['clipval'])
            self.num_critic = self.config['num_critic']
            self.disc_layers()

        if self.config['inneract'] == 'relu':
            self.inner_act = torch.relu
        else:
            self.inner_act = torch.sigmoid

        if self.is_gen:
            self.output_act = torch.sigmoid

        self.optim = torch.optim.RMSprop(self.parameters(), lr=self.config['rmsrate'])

        self.drop = torch.nn.Dropout(gan_config['dropout'])

    def __call__(self, x):
        return self.forward(x)

    def gen_layers(self):
        """
        Initialize generator layers
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
        Initialize discriminator layers
        Returns:

        """
        self.layers = torch.nn.ModuleList()
        layer_weights = [self.n_inputs + 1]
        layer_weights.extend(self.config['layer_nodes'])
        layer_weights.append(1)

        # Declare layer-wise weights and biases
        for i in range(len(layer_weights) - 1):
            self.layers.append(
                torch.nn.Linear(layer_weights[i], layer_weights[i + 1])
            )

    def forward(self, data):
        """
        Forward pass for input data
        Args:
            data (): 

        Returns:

        """
        out = data

        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if i != len(self.layers)-1:
                out = self.inner_act(out)

                out = self.drop(out)

        if self.is_gen:
            final_output = self.output_act(out)

        else:
            final_output = out

        return final_output

    def loss(self, y_true, y_gen):
        """
        Wasserstein loss for discriminator
        Args:
            y_true (): True labels
            y_gen (): Synth labels

        Returns:

        """

        # Discriminator only, generator done in loop
        return torch.mean(y_true) + torch.mean(y_gen)

    def backward(self, loss):
        """
        Backward pass for network
        Args:
            loss (): Loss error

        Returns:

        """
        loss.backward(retain_graph=True)
        self.optim.step()
        self.zero_grad()
        if self.is_disc:
            # Clip weights from WGAN original paper
            self.apply(self.clipper)

        return loss

    def disable_learn(self):
        """
        Disables learning for current network
        Returns:

        """
        for param in self.parameters():
            param.requires_grad = False

    def enable_learn(self):
        """
        Enables learning for current network
        Returns:

        """
        for param in self.parameters():
            param.requires_grad = True


class WeightClipper(object):

    def __init__(self, value=5):
        self.value = value

    def __call__(self, module):
        # Filter the variables to get weights
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-self.value, self.value)
            module.weight.data = w

