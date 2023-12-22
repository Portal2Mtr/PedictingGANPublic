"""RCGAN Classes

Classes for managing the RCGAN experimental approach.

"""

import sys
import numpy.random
import sklearn.metrics
import torch
import optuna
import copy
from sklearn.metrics import accuracy_score
import numpy as np
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch import DoubleTensor, LongTensor
from torch.autograd import Variable
import logging
import time
import pickle
import math
import random
from statistics import mean
# noinspection PyUnresolvedReferences
from _trainer_base import TrainerBase
from sklearn.model_selection import train_test_split
sys.path.append('./source/Augments/')
sys.path.append('./source/Datasets/')


logger = logging.getLogger(__name__)
DEVICE = torch.device("cpu")


class RCGanTrainer(TrainerBase):

    def __init__(self, config):
        self.generator = None
        self.disc = None
        self.num_features = None
        super().__init__(seed=0)
        torch.set_default_tensor_type('torch.DoubleTensor')
        # Start basic training loop

        self.config = config
        self.batchdiv = self.config['augconfig']['minibatches']
        self.dataset = []
        self.recordEigen = self.config['augconfig']['recordEigen']
        self.eigenseed = self.config['augconfig']['eigenseed']
        self.batch_size_div = self.config['ganconfig']['batchsizediv']
        self.lambdaval = self.config['augconfig']['lambda_val']
        self.clust_steps = self.config['augconfig']['clust_steps']
        self.clusters = {}

        return

    def init_data(self,trainPerc=None):
        """
        Initializes dataset for augmentation
        Returns:

        """
        data_name = self.config['augconfig']['dataset']
        self.dataset = self.get_dataset(data_name, trainPerc,
                                        self.config['augconfig']['has_idn'],
                                        self.config['augconfig']['idn_perc'])
        self.num_features = self.dataset.num_features
        self.num_classes = self.dataset.numClasses

    def init_networks(self, num_features=None,num_classes=None, class_array=None):
        """
        Initializes VecGAN Networks
        Args:
            num_features (): Number of input features
            class_array (): Array of classes for dataset

        Returns:

        """
        if num_features is None and class_array is None:
            self.create_networks(self.dataset.num_features,self.dataset.numClasses, self.config)
        else:
            self.create_networks(num_features,num_classes, self.config)

    def conduct_training(self, trial=None,classes=None):
        """
        Conduct the training loop for VecGAN
        Args:
            trial (): Optuna Trial

        Returns: Pickle output file or return for parameter study

        """



        if trial is None:
            gen_loss, discs_loss_overall = self.train()
            self.save_models(gen_loss, discs_loss_overall)
            pickle.dump([gen_loss, discs_loss_overall],
                        open(self.config['outputconfig']['outputdir'] + '/VecGANtrainingloss.p', 'wb'))
        else:
            if classes is None:
                gen_loss, discs_loss_overall = self.train(trial)
            else:
                # if self.config['augconfig']['has_idn'] or type(classes[0]) == list:
                #     gen_loss, disc_loss = self.train(trial,classes)
                # else:
                #     gen_loss, disc_loss = self.train(trial)

                gen_loss, discs_loss_overall = self.train(trial, classes)
            return gen_loss, discs_loss_overall, trial, self.generator, self.disc

    def create_networks(self, num_features,num_classes, config):
        """
        Create neural networks for VecGAN
        Args:
            num_features (): Number of input features
            config (): Network config dict
            class_array (): Array of classes

        Returns:

        """

        self.generator = RCGAN("Generator", num_features,num_classes, config, set_gen=True)
        # Make multiple discs equal to number of dataset classes
        self.disc = RCGAN("Discriminator", num_features,num_classes, config, set_disc=True)
        self.labeller = RCGAN("Labeller", num_features,num_classes, config, set_label=True)

    def gan_train_step(self, real_data, curr_labels, curr_weights):

        # Create a conditional GAN (D-step)

        self.generator.disable_learn()
        self.disc.enable_learn()

        # Random noise for generator to create synthetic data from.
        noise = torch.rand(len(real_data), self.num_features)
        labels = torch.tensor([real_data[i][-1] for i in range(len(real_data))])
        labels = labels.view(len(labels), 1)
        noise = torch.cat((noise, labels), 1)
        # Train generator on real labels
        synth_sample = self.generator(noise)

        synth_sample = torch.cat((synth_sample, labels),1)

        # Evaluate with discriminator on the validity of data samples
        # Change out labels for real data with corrupted labels
        for i in range(len(real_data)):
            real_data[i][-1] = curr_labels[i]
        real_label = self.disc(real_data)

        for i in range(len(synth_sample)):
            synth_sample[i][-1] = curr_labels[i]
        fake_label = self.disc(synth_sample)

        disc_loss = self.disc.loss(real_label, fake_label)

        disc_loss = self.disc.backward(disc_loss)

        # Stop discriminator and update generator (G-step)

        self.disc.disable_learn()
        self.generator.enable_learn()

        # Train generator with auxilary classifier
        noise = torch.rand(len(real_data), self.num_features)  # No batch processing
        labels = labels.view(len(labels), 1)
        noise = torch.cat((noise, labels), 1)

        synth_sample = self.generator(noise)
        synth_sample = torch.cat((synth_sample, labels),1)
        fake_label = self.disc(synth_sample)

        gen_loss_1 = self.generator.loss(None,fake_label)
        label_out = self.labeller(synth_sample)
        labels = labels.view(len(labels))
        labels = labels.long()
        gen_loss_2 = self.lambdaval * self.labeller.loss(labels,label_out)
        gen_loss = gen_loss_1 + gen_loss_2
        self.generator.backward(gen_loss)

        return gen_loss, disc_loss

    def noise_step(self, data):

        # Generate weights for training from IDN matrices
        corr_labels = []
        label_weights = []
        clust_idxs = []
        labels = data[:,-1]
        for i, datasamp in enumerate(data):

            label = int(datasamp[-1].item())

            # Corrupt label
            relavent_probs = self.conf_mat[label,:]
            raw_probs = relavent_probs.tolist()
            temp = np.random.multinomial(1, raw_probs, size=1)

            # Adjust corruption weight
            label_weights.append(float(self.inv_conf_mat[np.where(temp==1)[1],label]))
            corr_labels.append(int(np.where(temp == 1)[1]))

        return corr_labels, label_weights


    def train(self, trial=None,classes=None):
        """
        Setup for VecGAN training loop
        Args:
            trial (): Optuna trial object

        Returns:

        """

        SEED = 0
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED)  # Used in sklearn

        # Train on data samples
        gan_train_data = [(self.dataset[i]) for i in self.dataset.train_idxs]
        for idx in range(len(gan_train_data)):
            temp = gan_train_data[idx][0].tolist()
            temp.append(gan_train_data[idx][1])
            gan_train_data[idx] = (idx, torch.tensor(temp, requires_grad=True))

        self.batch_size = int(len(gan_train_data) / self.batch_size_div)

        data_loader = DataLoader(gan_train_data, batch_size=self.batch_size, shuffle=True)

        gan_train_config = self.config['augconfig']
        epochs = gan_train_config['epochs']
        label_epochs = gan_train_config['label_epochs']
        gen_loss_array = []
        disc_loss_array = []
        start = time.time()

        for epoch in range(300):

            for i, new_tuple in enumerate(data_loader):

                data = new_tuple[1]

                out = self.labeller.forward(data)
                labels = data[:,-1].long()
                # labels = labels.view(len(labels),1)
                loss = self.labeller.loss(labels, out)
                self.labeller.backward(loss)

            if (epoch + 1) % 100 == 0:
                logger.info('Time for labelling epoch {} is {} sec'.format(epoch + 1,time.time() - start,loss))
                start = time.time()

        # Create confusion matrix

        preds = {}

        for i in range(self.num_classes):
            preds[i] = []

        temp_data = [(self.dataset[i]) for i in self.dataset.train_idxs]
        true_labels = [int(temp_data[idx][1]) for idx in range(len(temp_data))]

        gan_train_data = [(self.dataset[i]) for i in self.dataset.train_idxs]
        for idx in range(len(gan_train_data)):
            temp = gan_train_data[idx][0].tolist()
            temp.append(gan_train_data[idx][1])
            gan_train_data[idx] = torch.tensor(temp, requires_grad=True)

        with torch.no_grad():

            raws_list = []
            for i, new_tuple in enumerate(gan_train_data):
                data = new_tuple
                raws = torch.sigmoid(self.labeller(data))
                raw_list = raws.tolist()
                raws_list.append(raw_list)

            for label, entry in zip(true_labels,raws_list):

                scaled = [i/sum(entry) for i in entry]
                preds[label].append(scaled)

            mat_list = []
            for i in range(self.num_classes):
                temp = np.mean(preds[i],axis=0).tolist()
                preds[i] = temp
                mat_list.append(preds[i])

        conf_mat = np.array(mat_list)

        # average predictions for confusion matrix
        try:
            self.inv_conf_mat = np.linalg.inv(conf_mat)
            self.conf_mat = conf_mat
        except np.linalg.LinAlgError:
            # Unlikely to happen
            self.inv_conf_mat = conf_mat
            self.conf_mat = conf_mat

        self.disc.zero_grad()
        self.generator.zero_grad()
        self.labeller.disable_learn()
        start = time.time()

        update_disc_loss = []
        update_gen_loss = []
        logger.info("Starting labelling...")
        data = [i[1] for i in gan_train_data]
        np_data = np.array([i.detach().numpy() for i in data])

        # Modify batch size to accomadate minority dataset
        for epoch in range(epochs):
            self.generator.disable_learn()  # Stop for agd
            self.disc.enable_learn()
            batch_gen_loss = []
            batch_disc_loss = []
            # Train on each combo of data
            self.generator.zero_grad()
            self.disc.zero_grad()

            for i,new_tuple in enumerate(data_loader):

                idxs = new_tuple[0].tolist()
                data = new_tuple[1]
                curr_labels, curr_weights = self.noise_step(data)
                work_labels = curr_labels
                work_weights = curr_weights

                gen_loss, disc_loss = \
                    self.gan_train_step(data, work_labels, work_weights)
                batch_gen_loss.append(gen_loss)
                batch_disc_loss.append(disc_loss)

                if self.generator.grad_mode == 'BFA':
                    if self.generator.log_error:
                        self.generator.update_learning(float(gen_loss.detach()))
                    if self.generator.log_error:
                        self.disc.update_learning(float(disc_loss.detach()))

            sum_gen = sum([float(i) for i in batch_gen_loss])
            sum_disc = sum([float(i) for i in batch_disc_loss])

            update_disc_loss.append(sum_disc)
            update_gen_loss.append(sum_gen)
            gen_loss_array.append(sum_gen)
            disc_loss_array.append(sum_disc)

            if (epoch + 1) % 100 == 0:
                logger.info('Time for epoch {} is {} sec, Disc. Loss={}, Gen. loss={}'.format(epoch + 1,
                                                  time.time() - start, update_disc_loss[-1], update_gen_loss[-1] ))
                start = time.time()
                update_disc_loss = []
                update_gen_loss = []

            if classes is not None:
                if (epoch+1) % 501 == 0 and trial is not None:
                    # if (epoch+1) % 501 == 0 and (self.config['augconfig']['has_idn'] or type(classes[0]) != list) and trial is not None:
                    acc = self.early_report(classes)
                    trial.report(acc,epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    else:
                        logger.info("Progress looks good with {} mean accuracy...".format(acc))

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

    def save_models(self, gen_loss, disc_loss, trial=None):
        """
        Saves generator and discriminators
        Args:
            gen_loss (): Generator Loss
            disc_loss (): Discriminators losses
            trial (): Optuna trial

        Returns:

        """

        if trial is None:
            output_config = self.config['outputconfig']
            torch.save(self.generator.state_dict(), output_config['outputdir'] + 'genweights'+str(self.seed)+'.p')

            torch.save(self.disc.state_dict(), output_config['outputdir'] +
                       'discweights'+str(self.seed)+'.p')
            pickle.dump([gen_loss, disc_loss], open(output_config['picklename'], 'wb'))
            clust_name = output_config['picklename'][:2] + "_CLUSTERS_" + ".p"

            pickle.dump(self.clusters, open(clust_name, 'wb'))
        logger.info("Saved Models!")

    def load_training(self, get_loss=True, get_discs=False, trial=None, gen=None, disc=None):
        """
        Load training data and models
        Args:
            get_loss (): Get the training loss
            get_discs (): Get the discriminator models
            trial (): Optuna trial number for correct file

        Returns:

        """

        if trial is None:
            output_config = self.config['outputconfig']

            self.generator.load_state_dict(torch.load(output_config['outputdir'] + 'genweights'+str(self.seed)+'.p'))
            self.disc.load_state_dict(torch.load( output_config['outputdir'] + 'discweights'+str(self.seed)+'.p'))
            gen_loss, disc_loss_total = pickle.load(open(output_config['picklename'], 'rb'))
            clust_name = output_config['picklename'][:2] + "_CLUSTERS_" + ".p"

            self.clusters = pickle.load(open(clust_name, 'rb'))

            if get_loss:
                return gen_loss, disc_loss_total
            elif get_discs:
                return self.disc
        else:
            self.generator = gen
            self.disc = disc

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

        synth_train_X, synth_test_X, synth_train_Y, synth_test_Y = train_test_split(synth_samples, class_labels)

        train_synth_data = [(x,y) for x,y in zip(synth_train_X, synth_train_Y)]
        test_synth_data = [(x,y) for x,y in zip(synth_test_X, synth_test_Y)]

        return train_synth_data, test_synth_data


class RCGAN(torch.nn.Module):

    def __init__(self, name, n_inputs,n_classes, config, set_gen=False, set_disc=False,set_label=False):
        self.store_clusters = {}
        self.layers = []
        self.n_classes = n_classes
        super(RCGAN, self).__init__()
        self.aug_name = config['augconfig']['augname']
        # store the parameters of network
        self.aug_name = config['augconfig']['augname']
        gan_config = config['ganconfig']
        self.is_gen = set_gen
        self.is_disc = set_disc
        self.is_label = set_label
        self.n_inputs = n_inputs
        self.bin_loss = torch.nn.BCELoss()
        self.grad_mode = config['augconfig']['gradmode']
        self.cum_loss = []

        self.num_select = [0, 0]
        self.arm_cnt = 0
        self.loss_var = 1.0

        self.feed_action = 0
        self.error_log = []
        self.variance_log = []
        self.u_values = []
        self.v_values = []
        self.label_loss = torch.nn.CrossEntropyLoss()

        if self.is_gen:
            self.config = gan_config['gen']
            self.gen_layers()
        elif self.is_disc:
            self.config = gan_config['disc']
            self.disclayers()
        elif self.is_label:
            self.config = gan_config['label']
            self.label_layers()

        self.learn_rate = self.config['learnrate']


        if self.config['inneract'] == 'relu':
            self.inner_act = torch.relu
        else:
            self.inner_act = torch.sigmoid

        self.output_act = torch.sigmoid

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learn_rate)

        self.drop = torch.nn.Dropout(gan_config['dropout'])

    def __call__(self, data):
        return self.forward(data)

    def gen_layers(self):
        """
        Create generator layers
        Returns:

        """

        self.layers = torch.nn.ModuleList()
        layer_weights = [self.n_inputs + 1]  # Add classes and cluster idxs
        layer_weights.extend(self.config['layer_nodes'])
        layer_weights.append(self.n_inputs)

        # Declare layer-wise weights and biases
        for i in range(len(layer_weights) - 1):
            self.layers.append(
                torch.nn.Linear(layer_weights[i], layer_weights[i+1])
            )

    def disclayers(self):
        """
        Create discriminator layers
        Returns:

        """

        self.layers = torch.nn.ModuleList()
        layer_weights = [self.n_inputs]  # input classes
        layer_weights.extend(self.config['layer_nodes'])
        layer_weights.append(1)

        # Declare layer-wise weights and biases
        for i in range(len(layer_weights) - 1):
            self.layers.append(
                torch.nn.Linear(layer_weights[i], layer_weights[i + 1])
            )

    def label_layers(self):

        self.layers = torch.nn.ModuleList()
        layer_weights = [self.n_inputs]  # input classes
        # layer_weights.extend(self.config['layer_nodes'])
        layer_weights.append(self.n_classes)

        # Declare layer-wise weights and biases
        for i in range(len(layer_weights) - 1):
            self.layers.append(
                torch.nn.Linear(layer_weights[i], layer_weights[i + 1])
            )

    def forward(self, data):
        """
        Forward pass for processing data
        Args:
            x ():

        Returns:

        """

        if self.is_gen:
            out = data
            label = None
        elif self.is_disc:
            out = data[:,:-1]
            label = data[:,-1]
            label = label.view(len(label),1)
        elif self.is_label:
            if len(data.shape) == 1:
                out = data[:-1]
            else:
                out = data[:,:-1]
            label = None

        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if i != len(self.layers)-1:
                out = self.inner_act(out)

                if not self.is_label:
                    out = self.drop(out)

        if self.is_gen:
            final_output = self.output_act(out)
        elif self.is_disc:
            # Projection disc
            final_output_1 = out
            final_output_2 = label * out

            final_output = final_output_1 + final_output_2


            final_output = self.output_act(final_output)

        else:
            final_output = out

        return final_output

    def backward(self, loss):
        """
        Backward pass for gradient updates
        Args:
            loss (): Error loss

        Returns:

        """

        # Managed depending on learning mode
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.zero_grad()
        return loss

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
        elif self.is_label:

            return self.label_loss(y_pred,y_true)


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
