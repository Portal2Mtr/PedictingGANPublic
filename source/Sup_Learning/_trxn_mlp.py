"""
Main classifier file for the blockchain network (MLP)
"""
import logging

import torch
import numpy as np
import copy
import time
import math
from statistics import mean
from torch.autograd import grad
import torch.nn.functional as F

import logging
logger = logging.getLogger()


class TrxnClassifier(torch.nn.Module):

    def __init__(self,conf):
        """
        Base learner for arbitrary inner worker problem (Classification, Regression, etc.)
        :param config:
        :param n_inputs:
        :param att_type:
        """
        super(TrxnClassifier, self).__init__()

        self.conf = conf
        self.grad_mode = self.conf['mode']
        self.attack_enabled = False
        self.n_inputs = self.conf['n_inputs']
        self.n_class = self.conf['num_cls']
        self.loss = torch.nn.MSELoss()
        self.layers = torch.nn.ModuleList()
        self.make_layers()
        self.mixed_attack = False
        self.param_shapes = []
        self.train_epochs = self.conf['train_epochs']

        if self.grad_mode == 'SGD':
            self.inner_act = torch.relu
        else:
            self.inner_act = torch.sigmoid

        self.outer_act = torch.sigmoid
        self.sgd_learn = self.conf['learn_alpha']
        self.optim = torch.optim.SGD(self.parameters(), lr=self.sgd_learn)

    def make_layers(self):
        """
        Create inner layers
        :return:
        """

        layer_weights = [self.n_inputs]
        layer_weights.extend(self.conf['layer_nodes'])
        layer_weights.append(1)

        # Declare layer-wise weights and biases
        for i in range(len(layer_weights) - 1):
            self.layers.append(
                torch.nn.Linear(layer_weights[i], layer_weights[i+1])
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

        y = self.outer_act(out)

        return y

    def backward(self, y_true, y_pred, rep_val=None):
        """
        Custom backward to store gradients for sharing.
        :return:
        """

        grad_error = False
        self.optim.zero_grad()

        loss = self.loss(y_true, y_pred)
        loss_val = loss.item()

        if self.grad_mode == 'EDL':
            # Do error-driven learning approach
            learning_rate = self.sgd_learn
            virtual_grad = []
            for param in self.parameters():
                if len(param.shape) > 1:
                    gradient = grad(loss, param, retain_graph=True)[0]

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
            self.optim.step()

        return grad_error, loss_val

    def train_loop(self,train_loader):

        start = time.time()
        for i in range(self.train_epochs):
            # Add small noise to introduce additional synthetic datapoints
            # Train on entire dataset then return gradients
            data, labels = next(iter(train_loader))
            y_pred = self.forward(data)
            # Calculate gradients and format for sharing
            y_act = labels.double().view(-1,1)
            _, loss = self.backward(y_pred, y_act, rep_val=None)
            if (i + 1) % 100 == 0:
                logger.info('Classifier time for epoch {} is {} sec w/ Loss {}'.format(i + 1, time.time() - start,loss))
                start = time.time()

    def pred_data(self, new_data):
        """
        Processes formatted transaction and predicts legality
        :param new_trxn: Formatted transaction from data_proc.py method
        :return:
        """


        with torch.no_grad():
            # Get percent confidence for transaction data
            new_data = torch.tensor(new_data)
            prediction = float(self.forward(new_data))

        return round(prediction)


