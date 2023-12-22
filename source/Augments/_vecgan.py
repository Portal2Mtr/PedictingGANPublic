"""VECGAN Classes

Classes for managing the VECGAN experimental approach.

"""

import sys
import numpy.random
import optuna
import torch
import sklearn
import logging
import time
import pickle
import copy
import math
import random
import numpy as np

from torch.autograd import grad
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance
from sklearn.metrics import accuracy_score
from statistics import mean
# noinspection PyUnresolvedReferences
from _trainer_base import TrainerBase
from sklearn.model_selection import train_test_split
sys.path.append('./source/Augments/')
sys.path.append('./source/Datasets/')

logger = logging.getLogger(__name__)
DEVICE = torch.device("cpu")

class VecGanTrainer(TrainerBase):

    def __init__(self, config):
        self.generator = None
        self.disc = None
        self.num_features = None
        super().__init__(seed=0)
        torch.set_default_tensor_type('torch.DoubleTensor')

        self.config = config
        self.batch_div = self.config['augconfig']['minibatches']
        self.dataset = []
        self.num_clust = self.config['ganconfig']['disc']['num_clust']
        self.batch_size_div = self.config['ganconfig']['batchsizediv']
        self.lambda_val = self.config['augconfig']['lambda_val']
        self.clust_steps = self.config['augconfig']['clust_steps']
        self.clusters = {}
        self.kmeans_store = None
        self.min_idxs = None
        self.dist_record = []

        return

    def init_data(self, trainPerc=None):
        """
        Initializes dataset for augmentation
        Returns:

        """
        data_name = self.config['augconfig']['dataset']
        self.dataset = self.get_dataset(data_name, trainPerc,
                                        self.config['augconfig']['has_idn'],
                                        self.config['augconfig']['idn_perc'])
        self.num_features = self.dataset.num_features

    def init_networks(self, num_features=None, class_array=None):
        """
        Initializes VecGAN Networks
        Args:
            num_features (): Number of input features
            class_array (): Array of classes for dataset

        Returns:

        """
        if num_features is None and class_array is None:
            self.create_networks(self.dataset.num_features, self.config)
        else:
            self.create_networks(num_features, self.config)

    def init_matrices(self):

        num_class = self.dataset.numClasses
        num_clust = self.num_clust

        for i in range(num_clust):
            self.clusters[i] = {
                'center': np.zeros((self.dataset.num_features,)),
                'trans_matrix': np.zeros((num_class,num_class)),
                'trans_matrix_inv': np.zeros((num_class,num_class)),
                'class_dict': {}
            }

        return

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

    def create_networks(self, num_features, config):
        """
        Create neural networks for VecGAN
        Args:
            num_features (): Number of input features
            config (): Network config dict
            class_array (): Array of classes

        Returns:

        """

        self.generator = VecGAN("Generator", num_features, config, set_gen=True)

        # Make multiple discs equal to number of dataset classes
        self.disc = VecGAN("Discriminator", num_features, config, set_disc=True)

    def gan_train_step(self, real_data, curr_labels, curr_weights, cluster_idxs):

        # Calculate cluster loss estimation for the discriminator
        min_idxs = self.min_idxs
        clust_centers = torch.tensor([self.clusters[i]['center'] for i in min_idxs], requires_grad=True)
        center_labels = self.disc(clust_centers)
        center_labels *= -1

        noise = torch.rand(len(real_data), self.num_features)
        labels = torch.tensor([real_data[i][-1] for i in range(len(real_data))])
        labels = labels.view(len(labels), 1)
        idxs = torch.tensor(cluster_idxs)
        idxs = idxs.view(len(idxs), 1)
        noise = torch.cat((noise, labels, idxs), 1)
        # Train generator on real labels
        synth_sample = self.generator(noise)
        synth_sample = torch.cat((synth_sample, labels), 1)
        # Evaluate with discriminator on the validity of data samples
        for i in range(len(real_data)):
            real_data[i][-1] = curr_labels[i]
        real_label = self.disc(real_data)  # Process real data once to make pytorch happy
        curr_weights = torch.tensor(curr_weights)
        curr_weights = curr_weights.view(len(curr_weights), 1)

        fake_label = self.disc(synth_sample)
        fake_label *= -1
        clust_loss = self.disc.loss(center_labels, real_label)
        disc_loss = self.disc.loss(real_label, fake_label)
        disc_loss = torch.multiply(disc_loss, torch.mean(curr_weights))
        clust_loss = self.disc.backward(clust_loss * self.lambda_val + disc_loss, store_loss=False)

        # Stop discriminator and update generator
        self.generator.enable_learn()
        self.disc.disable_learn()

        # Train generator with matrices regularization
        noise = torch.rand(len(real_data), self.num_features)  # No batch processing
        labels = labels.view(len(labels), 1)
        noise = torch.cat((noise, labels, idxs), 1)

        synth_sample = self.generator(noise)
        synth_sample = torch.cat((synth_sample, labels), 1)
        fake_label = self.disc(synth_sample)
        fake_label *= -1

        gen_loss = torch.mean(fake_label)
        self.generator.backward(gen_loss)
        self.generator.disable_learn()
        self.disc.enable_learn()

        return gen_loss, disc_loss, clust_loss

    def cluster_step(self, data, gen_weights=False):

        # Estimate wasserstein distances for all samples involved

        if not gen_weights:
            kmeans = sklearn.cluster.KMeans(n_clusters=self.num_clust, random_state=0, algorithm="auto")
            kmeans.fit_predict(data)
            self.kmeans_store = kmeans
            min_idxs = kmeans.labels_.tolist()
        else:
            kmeans = self.kmeans_store
            min_idxs = kmeans.predict(data)

        # Find kth minimum distances for each data sample from clusters

        num_classes = self.dataset.numClasses

        with torch.no_grad():
            # Generate wasserstein distance estimates for each samples

            # Update clusters based on k minimum distances
            clust_data = {}
            cluster_record = []

            for cluster_idx in range(kmeans.n_clusters):
                clust_data[cluster_idx] = []
                self.clusters[cluster_idx]['center'] = kmeans.cluster_centers_[cluster_idx, :]

            for sample, min_idx in zip(data, min_idxs):
                min_dist = wasserstein_distance(kmeans.cluster_centers_[min_idx], sample)
                clust_data[min_idx].append((sample, min_dist))
                cluster_record.append(min_idx)

            clust_error_total = 0.0
            if not gen_weights:
                self.min_idxs = min_idxs

                # Update cluster centers with mean minimum values
                for i in self.clusters.keys():
                    temp = [clust_data[i][j][1] for j in range(len(clust_data[i]))]
                    if len(temp) < 2:
                        continue

                    clust_error = sum(temp)

                    clust_error_total += clust_error

                # Update regularization matrices

                for generate_idx in self.clusters.keys():

                    if len(clust_data[generate_idx]) < 2:
                        continue

                    work_data = [clust_data[generate_idx][i][0] for i in range(len(clust_data[generate_idx]))]
                    class_dict = {}
                    for i in range(num_classes):
                        class_dict[i] = []

                    for idx,sample in enumerate(work_data):
                        class_dict[sample[-1]].append(idx)

                    count_dict = {}

                    for key,value in class_dict.items():
                        if len(value):
                            count_dict[key] = len(value)

                    # Update minimum counts for each class
                    # class_means = []
                    class_has_cnt = []
                    for i in range(num_classes):

                        if not class_dict[i]:
                            class_has_cnt.append(False)
                            continue

                        # class_means.append(np.mean([work_data[j][:-1] for j in range(len(class_dict[i]))],axis=0))
                        class_has_cnt.append(True)

                    # Sums
                    class_sums = []
                    for i in range(len(class_has_cnt)):
                        work_sum = 0
                        for j in range(len(class_has_cnt)):
                            if class_has_cnt[i] or class_has_cnt[j]:
                                L = [class_dict[i], class_dict[j]]
                                work_sum += min(len(x) for x in L if x is not None)

                        class_sums.append(work_sum)

                    # Individual probabilities
                    temp_mat = np.zeros((num_classes, num_classes))
                    for i in range(len(class_has_cnt)):
                        for j in range(len(class_has_cnt)):
                            if len(class_dict[i]) and len(class_dict[j]) and class_sums[i] > 0:
                                temp_mat[i][j] = min(len(class_dict[i]), len(class_dict[j])) / class_sums[i]

                    # Remove empty rows and columns and make map for reduced matrix
                    temp_mat = temp_mat[temp_mat != 0.0]
                    num_avail_classes = 0
                    class_dict = {}
                    for idx, i in enumerate(class_has_cnt):
                        if i:
                            class_dict[idx] = num_avail_classes
                            num_avail_classes += 1

                    temp_mat = temp_mat.reshape((num_avail_classes, num_avail_classes))

                    # Have probabilities from here on

                    # Update transition matrices based on the number of labels in each cluster
                    self.clusters[generate_idx]['trans_matrix'] = temp_mat
                    self.clusters[generate_idx]['class_dict'] = class_dict
                    self.clusters[generate_idx]['count_dict'] = count_dict

                    # Calculate individual sample distances for IDN noise
                    # clust_dists = 1.0/(np.linalg.norm(work_data - self.clusters[generate_idx]['center'],axis=1))
                    # clust_dists = scipy.special.softmax(clust_dists)
                    # self.dist_record.extend(clust_dists.tolist())

                    try:
                        inv_temp = np.linalg.inv(temp_mat)
                    except np.linalg.LinAlgError:
                        inv_temp = temp_mat
                        pass # Go with old inverse for now, next one shouldn't be a singular matrix
                    self.clusters[generate_idx]['trans_matrix_inv'] = inv_temp

            # Generate weights for training from IDN matrices
            corr_labels = []
            label_weights = []
            clust_idxs = []
            labels = [int(data[i][-1]) for i in range(len(data))]
            for i, (label, idx) in enumerate(zip(labels, cluster_record)):
                # Corrupt label
                clust_idxs.append(idx)
                if len(self.clusters[idx]['class_dict']) == 0:

                    # samp_weight = dist
                    # label_weights.append(samp_weight)
                    corr_labels.append(label)
                    label_weights.append(np.array([1.0]))
                    continue

                trans_label = self.clusters[idx]['class_dict'][labels[i]]
                relevant_probs = self.clusters[idx]['trans_matrix'][trans_label, :]

                # test_probs = dist / relevant_probs
                # test_probs = test_probs / sum(test_probs)
                temp = np.random.multinomial(1, relevant_probs, size=1)
                #     if sum(relevant_probs) == 1:
                #         label_weights.append(1.0)
                #         corr_labels.append(int(np.where(relevant_probs == 1)[1]))
                #     else: # Probably nan, can be case if single sample in matrix
                #         label_weights.append(1.0)
                #         corr_labels.append(label)
                #         # Continue, shouldn't happen often
                # else:
                # Adjust corruption weight
                label_weights.append(self.clusters[idx]['trans_matrix_inv'][np.where(temp == 1)[1], trans_label])
                corr_labels.append(int(np.where(temp == 1)[1]))

        return corr_labels, label_weights, clust_idxs, clust_error_total

    def train(self, trial=None, classes=None):
        """
        Setup for VecGAN training loop
        Args:
            classes:
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
            gan_train_data[idx] = (idx,torch.tensor(temp, requires_grad=True))

        self.batch_size = int(len(gan_train_data) / self.batch_size_div)

        data_loader = DataLoader(gan_train_data, batch_size=self.batch_size, shuffle=True)

        gan_train_config = self.config['augconfig']
        epochs = gan_train_config['epochs']
        gen_loss_array = []
        disc_loss_array = []
        start = time.time()

        self.disc.zero_grad()
        self.generator.zero_grad()
        self.init_matrices()
        start = time.time()

        update_disc_loss = []
        update_gen_loss = []
        update_clust_loss = []
        num_clust = 0
        clust_cnt = 0
        logger.info("Starting clustering...")
        data = [i[1] for i in gan_train_data]
        np_data = np.array([i.detach().numpy() for i in data])

        _, _,_, cluster_error = self.cluster_step(np_data,gen_weights=False)
        logger.info("Cluster error from kmeans was {}".format(cluster_error))
        final_weights = False
        # Modify batch size to accomadate minority dataset
        for epoch in range(epochs):
            self.generator.disable_learn()  # Stop for agd
            self.disc.enable_learn()
            batch_gen_loss = []
            batch_disc_loss = []
            batch_clust_loss = []
            # Train on each combo of data
            self.generator.zero_grad()
            self.disc.zero_grad()

            for i,new_tuple in enumerate(data_loader):

                idxs = new_tuple[0].tolist()
                data = new_tuple[1]
                curr_labels, curr_weights, cluster_idxs, _ = self.cluster_step(np_data[idxs], True)
                work_labels = curr_labels
                work_weights = curr_weights
                gen_loss, disc_loss, clust_loss = \
                    self.gan_train_step(data, work_labels, work_weights, cluster_idxs)

                batch_gen_loss.append(gen_loss)
                batch_disc_loss.append(disc_loss)
                batch_clust_loss.append(clust_loss)

                if self.generator.grad_mode == 'BFA':
                    if self.generator.log_error:
                        self.generator.update_learning(float(gen_loss.detach()))
                    if self.generator.log_error:
                        self.disc.update_learning(float(disc_loss.detach() + clust_loss.detach()))

            sum_gen = sum([float(i) for i in batch_gen_loss])
            sum_disc = sum([float(i) for i in batch_disc_loss])
            sum_clust = sum([float(i) for i in batch_clust_loss])

            update_disc_loss.append(sum_disc)
            update_gen_loss.append(sum_gen)
            update_clust_loss.append(sum_clust)
            gen_loss_array.append(sum_gen)
            disc_loss_array.append(sum_disc + sum_clust)

            if self.generator.grad_mode == 'BFA':
                with torch.no_grad():
                    try:
                        gen_loss_var = float(torch.var(torch.stack(batch_gen_loss)).to(DEVICE))
                    except RuntimeError:
                        gen_loss_var = 1.0

                    disc_loss_var = float(torch.var(torch.stack(batch_disc_loss)).to(DEVICE))
                    self.generator.update_history(gen_loss_var, epoch == 0)
                    self.disc.update_history(disc_loss_var, epoch == 0)

            if (epoch + 1) % 50 == 0:

                logger.info('Time for epoch {} is {} sec, Disc. Loss={}, Clust. Loss {} '
                            'Gen. loss={}'.format(epoch + 1,
                                                  time.time() - start, update_disc_loss[-1],
                                                  update_clust_loss[-1], update_gen_loss[-1]))
                start = time.time()
                update_disc_loss = []
                update_gen_loss = []
                update_clust_loss = []

                if self.generator.grad_mode == "BFA":
                    logger.info('Gen. Selection: {} Disc. Select: {}'.format(self.generator.num_select,
                                                                             self.disc.num_select))

            if classes is not None:
                if (epoch+1) % 501 == 0 and trial is not None:
                    # if (epoch+1) % 501 == 0 and (self.config['augconfig']['has_idn'] or type(classes[0]) != list) and trial is not None:
                    self.generator.store_clusters = copy.deepcopy(self.clusters)
                    self.generator.store_kmeans = copy.deepcopy(self.kmeans_store)

                    acc = self.early_report(classes)
                    trial.report(acc,epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    else:
                        logger.info("Progress looks good with {} mean accuracy...".format(acc))

        self.generator.store_clusters = self.clusters
        self.generator.store_kmeans = self.kmeans_store

        return gen_loss_array, disc_loss_array

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
            gen: Generator object to load training from
            disc: Discriminator object to load training from
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
        Synthesize data samples with VecGAN
        Args:
            aug_config (): Augmentation config
            total_dataset (): Dataset for synthesis from supervised learner
            min_label (): Minority label
        Returns:

        """
        # Load training data and get num of samples to generate
        synth_samples = []
        class_labels = []
        num_features = total_dataset.num_features
        clust_class_dicts = []
        clusters = self.generator.store_clusters
        for i in clusters.keys():
            clust_class_dicts.append(clusters[i]['class_dict'])

        sample_idxs = []
        for class_num in total_dataset.class_array:
            temp = []
            for idx, clust in enumerate(clust_class_dicts):
                if class_num in clust.keys():
                    temp.append(idx)

            sample_idxs.append(temp)

        # Assign test samples to each cluster
        test_data = [total_dataset.all_data[i] for i in total_dataset.test_idxs]
        test_labels = [total_dataset.all_labels[i] for i in total_dataset.test_idxs]

        with torch.no_grad():

            labeller = self.generator.store_kmeans
            for data, label in zip(test_data, test_labels):
                to_kmeans = np.append(data, label)
                clust_assign = labeller.predict(to_kmeans.reshape(1, -1))

                clust_assign = int(clust_assign)

                if 'count_dict' in clusters[clust_assign].keys():
                    if label in clusters[clust_assign]['count_dict'].keys():

                        clusters[clust_assign]['count_dict'][label] += 1
                    else:
                        clusters[clust_assign]['count_dict'][label] = 1

        class_weights = {}
        for class_num in total_dataset.class_array:
            class_cnt = []
            for clust in clusters.values():
                if 'count_dict' not in clust.keys():
                    class_cnt.append(0)
                    continue

                if class_num not in clust['count_dict'].keys():
                    class_cnt.append(0)
                else:
                    class_cnt.append(clust['count_dict'][class_num])

            class_sum = sum(class_cnt)
            try:
                class_scale = [class_cnt[i]/class_sum for i in range(len(class_cnt))]
            except ZeroDivisionError:
                class_scale =  [1 for i in range(len(class_cnt))]
            class_weights[class_num] = class_scale

        # Generate weights for sampling from each cluster with each class
        clust_list = [i for i in clusters.keys()]
        with torch.no_grad():
            for class_lab, num_gen in zip(total_dataset.class_array, total_dataset.num_gen_array):
                for i in range(num_gen):
                    noise = torch.rand(1, num_features)
                    # sel_idx = np.random.choice(clust_list) # Sample from clusters
                    sel_idx = np.random.choice(clust_list, p=class_weights[class_lab])  # Sample from clusters
                    app_idx = torch.tensor(sel_idx).view(1, 1)
                    noise = torch.cat((noise, torch.tensor(class_lab).view(1, 1), app_idx), 1)
                    synth_sample = self.generator.forward(noise)
                    synth_samples.append(synth_sample[0])
                    class_labels.append(torch.tensor(class_lab))

        synth_train_x, synth_test_x, synth_train_y, synth_test_y = train_test_split(synth_samples, class_labels,
                                                                                    train_size=0.8)

        train_synth_data = [(x, y) for x, y in zip(synth_train_x, synth_train_y)]
        test_synth_data = [(x, y) for x, y in zip(synth_test_x, synth_test_y)]

        return train_synth_data, test_synth_data


    def early_report(self, classes):
        """
        Reports to optuna to stop trial early for single classifier.
        Args:
            classes: Classifier array
        Returns:

        """
        train_data, test_data = self.synth_samples(None, self.dataset)
        early_data = copy.deepcopy(self.dataset)
        copy_classes = copy.deepcopy(classes)

        if type(classes[0]) != list:
            use_mlp = True
            mlp_class = copy_classes[0]
        else:
            use_mlp = False
            class1, class2, class3 = copy_classes[0]

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

            for data, label in test_data:
                pred_labels.append(mlp_class.pred_data(data))

            true_labels = [test_data[i][1] for i in range(len(test_data))]
            remain_idxs = []
            for idx, label in enumerate(true_labels):
                if type(label) != str:
                    remain_idxs.append(idx)
                if type(label) == str:  # Synth label
                    continue  # Remove synth samples from testing

            show_test_labels = np.array([true_labels[i] for i in remain_idxs])
            show_class1_labels = np.array([pred_labels[i] for i in remain_idxs])
            acc_score = accuracy_score(show_test_labels, show_class1_labels)
            return acc_score

        for idx, data in enumerate(test_data):
            temp_data = test_data[idx][0].tolist()
            temp_label = test_data[idx][1]
            test_data[idx] = [temp_data, temp_label]

        train_data, train_labels = sklearn.utils.shuffle(train_data, train_labels)
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
            if type(label) != str:  # Synth label
                remain_idxs.append(idx)

            if type(label) == str:
                continue  # Remove synth label from training

        show_test_labels = np.array([true_labels[i] for i in remain_idxs])
        show_class1_labels = np.array([class1_pred_labels[i] for i in remain_idxs])
        show_class2_labels = np.array([class2_pred_labels[i] for i in remain_idxs])
        show_class3_labels = np.array([class3_pred_labels[i] for i in remain_idxs])
        acc_score_class1 = accuracy_score(show_test_labels, show_class1_labels)
        acc_score_class2 = accuracy_score(show_test_labels, show_class2_labels)
        acc_score_class3 = accuracy_score(show_test_labels, show_class3_labels)

        acc_scores = [acc_score_class1, acc_score_class2, acc_score_class3]
        return mean(acc_scores)  # Single value reporting for optuna

class VecGAN(torch.nn.Module):

    def __init__(self, name, n_inputs, config, set_gen=False, set_disc=False):
        self.store_loss = None
        self.s_values = None
        self.log_error = None
        self.store_clusters = {}
        self.layers = []
        super(VecGAN, self).__init__()
        self.aug_name = config['augconfig']['augname']
        # store the parameters of network
        self.aug_name = config['augconfig']['augname']
        gan_config = config['ganconfig']
        self.is_gen = set_gen
        self.is_disc = set_disc
        self.n_inputs = n_inputs
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

        if self.is_gen:
            self.config = gan_config['gen']
            self.gen_layers()
        elif self.is_disc:
            self.config = gan_config['disc']
            self.disc_layers()

        self.num_log = self.config['numlog']
        self.num_update_incr = 0

        self.learn_rate = self.config['learnrate']

        if self.grad_mode == 'SGD':

            if self.config['inneract'] == 'relu':
                self.inner_act = torch.relu
            else:
                self.inner_act = torch.sigmoid

            if self.is_gen:
                self.output_act = torch.sigmoid

            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learn_rate)

        else:
            # Only tanh for feedback alignment
            self.inner_act = torch.sigmoid
            self.output_act = torch.sigmoid
            if self.is_gen:  # disc output is linear

                self.disable_learn_update = gan_config['gen']['disable_update']
            if self.is_disc:
                self.disable_learn_update = gan_config['disc']['disable_update']

        self.rank_cutoffs = []
        self.calc_cutoffs = True
        self.param_len = len([i for i in self.parameters()])
        self.gamma_alphas = [[] for _ in range(self.param_len)]
        self.gamma_betas = [[] for _ in range(self.param_len)]
        self.s_memory = [[] for _ in range(self.param_len)]
        self.s_cutoffs = [0 for _ in range(self.param_len)]
        self.s_pred_vals = [[[]] for j in range(self.param_len)]
        self.s_mem_means = [[] for j in range(self.param_len)]
        self.made_prediction = False
        self.var_mult = self.config['varmult']

        self.drop = torch.nn.Dropout(gan_config['dropout'])

    def update_learning(self, loss):
        """
        Updates the type of learning algorithm taken for learning
        Returns:

        """

        self.error_log.append(loss)
        if len(self.error_log) > self.num_log:
            # Remove old history
            self.error_log.pop(0)

            # Remove old matrix approximation history
        if len(self.s_memory[0][0]) > self.num_log:
            for idx, s_mem in enumerate(self.s_memory):
                if len(s_mem) != 0:
                    for jdx, item in enumerate(s_mem):
                        self.s_memory[idx][jdx].pop(0)

    def update_history(self, loss_var, first_epoch=False):
        """
        Update the bandit learner for selecting actions
        Args:
            first_epoch: Flag for if first epoch
            loss_var: Variance of past loss

        Returns:

        """

        self.log_error = False

        if first_epoch:
            if loss_var < self.loss_var:
                self.loss_var = loss_var

        if self.disable_learn_update:

            if self.made_prediction:
                rand_val = np.random.choice([0, 1])
                self.feed_action = rand_val

                if self.feed_action == 1:
                    return

        else:

            if loss_var < self.var_mult * self.loss_var:
                if loss_var < self.loss_var:
                    self.loss_var = loss_var
                if self.made_prediction:
                    self.feed_action = 1  # continue with 'good' matrix
                    # Wait until there is data for prediction

                self.log_error = False
                return

        self.feed_action = 0
        self.log_error = True

        # Try noise for SVD at minimum error
        for idx, s_vals in enumerate(self.s_values):
            if s_vals is not None:
                update_vector = s_vals[0].numpy()
                update_vector = update_vector[:self.s_cutoffs[idx]]
                for jdx, s_val in enumerate(update_vector):
                    self.s_memory[idx][jdx].append(s_val)

        self.num_update_incr += 1

        if self.num_update_incr >= self.num_log and not first_epoch and len(self.error_log) > 2:
            self.made_prediction = True
            self.s_pred_vals = [[[] for _ in range(self.s_cutoffs[idx])] for idx in range(self.param_len)]
            min_error_idx = (len(self.error_log) - 1) - np.argmin(self.error_log)
            with torch.no_grad():
                for idx in range(self.param_len):

                    if len(self.s_memory[idx]) == 0:
                        continue

                    s_mem_mean = self.s_mem_means[idx][0]
                    for jdx, s_mean in enumerate(s_mem_mean):  # Number of memory

                        diff_log = [abs(i - s_mean) for i in self.s_memory[idx][jdx]]
                        min_diff = diff_log[min_error_idx]
                        self.s_pred_vals[idx][jdx] = [torch.tensor(s_mem_mean[jdx]) +
                                                          torch.abs(torch.normal(torch.tensor(min_diff),
                                                                       self.loss_var))]

            self.s_mem_means = [[[] for i in range(self.s_cutoffs[idx])] for j in range(self.param_len)]
            for idx in range(self.param_len):
                if len(self.s_memory[idx]) == 0:
                    continue

                self.s_mem_means[idx] = [[] for _ in range(self.s_cutoffs[idx])]

                for jdx, s_vals in enumerate(self.s_memory[idx]):
                    s_mem_mean = mean(self.s_memory[idx][jdx])
                    self.s_mem_means[idx][jdx].append(s_mem_mean)

            self.num_update_incr = 0

    def calc_beta(self, gradient):

        beta = float(gradient.size()[0])/float(gradient.size()[1])

        self.rank_cutoffs.append(
            0.56*beta**3-0.95*beta**2+1.82*beta+1.43
        )

    def reformat_memories(self, idx, s_cutoff):

        self.s_cutoffs[idx] = s_cutoff
        self.s_memory[idx] = [[] for i in range(s_cutoff)]

        return

    def manage_grads(self, loss, store_loss=False):
        """
        Manages the calculation of gradient updates for VECGAN
        Args:
            loss: Error for grads

        Returns:

        """

        if store_loss:
            self.store_loss = loss
            return
        else:
            if self.store_loss is not None:
                self.store_loss = self.store_loss + loss
                loss = self.store_loss
                self.store_loss = None

        if self.grad_mode == 'BFA':
            # Do bayesian feedback alignment approach

            learning_rate = self.learn_rate
            virtual_grad = []  # operate in place
            if self.feed_action == 0:
                # Compute truncated svd

                self.num_select[0] += 1

                self.u_values = []
                self.v_values = []
                self.s_values = []

                for idx, param in enumerate(self.parameters()):
                    if len(param.shape) > 1:
                        s_values = []
                        gradient = grad(loss, param, retain_graph=True, allow_unused=True)[0]

                        # Inbuilt SVD
                        with torch.no_grad():
                            u, s, vh = torch.linalg.svd(gradient, full_matrices=False)

                        # Zero smaller values based on threshold from https://arxiv.org/abs/1305.5870

                        if self.calc_cutoffs:
                            self.calc_beta(gradient)

                        with torch.no_grad():
                            # if self.calc_cutoffs:
                            s_cutoff = min(len(s.shape), max(1, math.ceil(torch.median(s) * self.rank_cutoffs[idx])))
                            if s_cutoff != self.s_cutoffs[idx]:
                                self.reformat_memories(idx, s_cutoff)

                            s[self.s_cutoffs[idx]:] = 0
                            s_values.append(s.detach())
                            self.s_values.append(s_values)
                            if self.calc_cutoffs:
                                self.s_mem_means[idx] = [s[:self.s_cutoffs[idx]].tolist()]

                        v = vh.transpose(-2, -1).conj()
                        r1 = 0.1
                        grad_modified = torch.mm(torch.mm(u, torch.diag(s + r1)), v.t())
                        virtual_grad.append(grad_modified)
                        self.u_values.append(u.detach())
                        self.v_values.append(v.detach())
                    else:
                        virtual_grad.append(grad(loss, param, retain_graph=True)[0])
                        self.u_values.append(None)
                        self.v_values.append(None)
                        self.s_values.append(None)
                        if self.calc_cutoffs:
                            self.rank_cutoffs.append(None)

                if self.calc_cutoffs:  # Only compute on first try
                    self.calc_cutoffs = False

                with torch.no_grad():
                    for w, update in zip(self.parameters(), virtual_grad):
                        try:
                            w.data -= learning_rate * update.data
                        except RuntimeError:
                            continue  # Occasional matrix addition error

            elif self.feed_action == 1:

                # Construct a synthetic svd with added noise to eigen values
                self.num_select[1] += 1
                virtual_grad = []
                r1 = 0.1

                with torch.no_grad():
                    for idx, (u_value, s_values, v_values, param) in \
                            enumerate(zip(self.u_values, self.s_pred_vals, self.v_values, self.parameters())):
                        if u_value is not None:
                            u = u_value
                            s = s_values[0]
                            v = v_values
                            temp_tensor = torch.zeros(u.shape[1])
                            temp_tensor[:self.s_cutoffs[idx]] = s[0]
                            temp = torch.mm(u, torch.diag(temp_tensor + r1))
                            grad_modified = torch.mm(temp, v.t())
                            virtual_grad.append(grad_modified)

                        else:
                            rand_val = 0.1
                            update_val = rand_val * loss.detach().numpy()
                            update_vec = torch.full(param.shape, float(update_val))
                            virtual_grad.append(update_vec)

                with torch.no_grad():
                    for w, update in zip(self.parameters(), virtual_grad):
                        w.data -= learning_rate * update.data

        elif self.grad_mode == 'EDL':

            # Do error-driven learning approach
            learning_rate = self.learn_rate
            virtual_grad = []
            for param in self.parameters():
                if len(param.shape) > 1:
                    gradient = grad(loss, param, retain_graph=True)[0]

                    # Inbuilt SVD
                    # Inbuilt SVD
                    with torch.no_grad():
                        try:
                            u, s, vh = torch.linalg.svd(gradient, full_matrices=False)
                        except RuntimeError:
                            length = gradient.shape[0]
                            height = gradient.shape[1]
                            u, s, vh = torch.linalg.svd(gradient + 1e-4 * gradient.mean() * torch.rand(length, height),
                                                        full_matrices=False)

                    v = vh.transpose(-2, -1).conj()
                    r1 = 0.1
                    grad_modified = torch.mm(torch.mm(u, torch.diag(s + r1)), v.t())
                    virtual_grad.append(grad_modified)
                else:
                    virtual_grad.append(grad(loss, param, retain_graph=True)[0])

            with torch.no_grad():
                for w, update in zip(self.parameters(), virtual_grad):
                    w.data -= learning_rate * update.data
        elif self.grad_mode == 'SGD':
            # Standard stochastic gradient descent
            loss.backward(retain_graph=True)

            self.optimizer.step()

        return

    def __call__(self, data):
        return self.forward(data)

    def gen_layers(self):
        """
        Create generator layers
        Returns:

        """

        self.layers = torch.nn.ModuleList()
        layer_weights = [self.n_inputs + 2]  # Add classes and cluster idxs
        layer_weights.extend(self.config['layer_nodes'])
        layer_weights.append(self.n_inputs)

        # Declare layer-wise weights and biases
        for i in range(len(layer_weights) - 1):
            self.layers.append(
                torch.nn.Linear(layer_weights[i], layer_weights[i+1])
            )

    def disc_layers(self):
        """
        Create discriminator layers
        Returns:

        """

        self.layers = torch.nn.ModuleList()
        layer_weights = [self.n_inputs + 1]  # input classes
        layer_weights.extend(self.config['layer_nodes'])
        layer_weights.append(1)

        # Declare layer-wise weights and biases
        for i in range(len(layer_weights) - 1):
            self.layers.append(
                torch.nn.Linear(layer_weights[i], layer_weights[i + 1])
            )

    def forward(self, data):
        """
        Forward pass for processing data
        Args:
            data: Data input to MLP network

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
            # Linear
            final_output = out

        return final_output

    def backward(self, loss,store_loss=False):
        """
        Backward pass for gradient updates
        Args:
            loss (): Error loss

        Returns:

        """

        # Managed depending on learning mode
        self.manage_grads(loss,store_loss)
        if not store_loss:
            self.zero_grad()
        return loss

    def loss(self, pred, real):

        return torch.mean(pred) + torch.mean(real)

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
