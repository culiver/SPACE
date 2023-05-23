import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset

from options import args_parser
from update import LocalUpdate, test_inference
from utils import get_dataset, average_weights, exp_details, save_json, get_json

import matplotlib
import matplotlib.pyplot as plt
import json
from utils import contribution_eval
from utilities import stable_sigmoid
from .central import Central

import itertools
import math
import time
import cvxpy as cp

from pydvl.value.shapley.gt import _constants
from pydvl.value.shapley.gt import num_samples_eps_delta
from pydvl.utils.numeric import random_subset_of_size
from pydvl.utils.status import Status
from numpy.typing import NDArray
from typing import TypeVar

T = TypeVar("T", NDArray[np.float_], float)

class GroupTesting(Central):
    def __init__(self, args, my_model, ckp):
        super().__init__(args, my_model, ckp)

    def get_contributions(self):
        save_json(self.args, 'contributions.json', self.contributions.tolist())
        save_json(self.args, 'contributions_k[{}]_T[{}].json'.format(self.args.k, self.args.T), self.contributions.tolist())
        self.ckp.write_log("Contribution of clients: {}".format(self.contributions))

    def train(self, user_shards=None):
        print('='*20)
        print('Start Calculate Group Testing Shapley value!')
        print('='*20)

        self.args.trainer = 'realshapley'
        self.subsets_info = get_json(self.args, 'subsets_info.json')
        self.subsets_info = {} if self.subsets_info is None else self.subsets_info
        self.subsets_info['num_users'] = self.args.num_users

        self.args.trainer = 'gt'

        eps = 0.001

        if self.args.gpu_id:
            torch.cuda.set_device(self.args.gpu_id)
        device = 'cuda' if self.args.gpu else 'cpu'

        def trainSubset(subset):
            self.model = copy.deepcopy(self.init_model)
            self.model.to(self.device)
            self.model.train()
            # copy weights
            global_weights = self.model.state_dict()

            for epoch in range(self.args.epochs):
                local_weights, local_losses = self.local_training(global_round=epoch, idxs_users=subset)

                # update global weights
                global_weights = average_weights(local_weights, weightings=self.client_data_ratio[subset])

                # update global weights
                self.model.load_state_dict(global_weights)

                loss_avg = (np.array(local_losses) * self.client_data_ratio[subset]).sum()

            # Test inference after completion of training
            test_acc, test_loss = test_inference(self.args, self.model, self.test_dataset)
            self.subsets_info[subsetName] = test_acc
            self.ckp.write_log("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            save_json(self.args, 'subsets_info.json', self.subsets_info)

        # set the max iteration number to O(n(logn)^2)
        if self.args.dataset == 'cifar':
            epsilon = 6
        elif self.args.dataset == 'mnist':
            epsilon = 3
        else:
            epsilon = 1
        delta = 0.05
        n_samples = num_samples_eps_delta(epsilon, delta, self.args.num_users, utility_range = 1-0)
        print("n_samples", n_samples)
        self.ckp.write_log("GT sampling iterations: {} with epsilon: {}".format(n_samples, epsilon))

        indices = [i for i in range(self.args.num_users)]

        #############################################################

        rng = np.random.default_rng()
        n = self.args.num_users
        const = _constants(n, 1, 1, 1)  # don't care about eps,delta,range

        betas = np.zeros(shape=(n_samples, n), dtype=np.int_)  # indicator vars
        uu = np.empty(n_samples)  # utilities

        for t in range(n_samples):
            k = rng.choice(const.kk, size=1, p=const.q).item()
            s = random_subset_of_size(indices, k)
            subsetName ='+'.join([str(elem) for elem in sorted(s)])
            if subsetName not in self.subsets_info:
                trainSubset(s)
            if self.args.u_trans:
                uu[t] = stable_sigmoid((self.subsets_info[subsetName]-self.args.T) * self.args.k) if (self.subsets_info[subsetName] != self.args.T) else 0.5    
            else:
                uu[t] = self.subsets_info[subsetName]
            betas[t, s] = 1

        #############################################################
            
        const = _constants(
            n=n,
            epsilon=epsilon,
            delta=delta,
            utility_range=1 - 0,
        )
        T = n_samples
        if T < const.T:
            print("n_samples of {T} are below the required {const.T} for the \n ε={epsilon:.02f} guarantee at δ={1 - delta:.02f} probability")

        # Matrix of estimated differences. See Eqs. (3) and (4) in the paper.
        C = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(i + 1, n):
                C[i, j] = np.dot(uu, betas[:, i] - betas[:, j])
        C *= const.Z / T

        subset = [i for i in range(self.args.num_users)]
        subsetName ='+'.join([str(elem) for elem in [i for i in range(self.args.num_users)]])
        if subsetName not in self.subsets_info:
            trainSubset(subset)

        total_utility = self.subsets_info[subsetName]
        if self.args.u_trans:
            total_utility = stable_sigmoid((total_utility-self.args.T) * self.args.k) if (total_utility != self.args.T) else 0.5    

        ###########################################################################
        # Solution of the constraint problem with cvxpy

        v = cp.Variable(n)
        constraints = [cp.sum(v) == total_utility]
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append(v[i] - v[j] <= epsilon + C[i, j])
                constraints.append(v[j] - v[i] <= epsilon - C[i, j])

        problem = cp.Problem(cp.Minimize(0), constraints)
        problem.solve()

        if problem.status != "optimal":
            values = (
                np.nan * np.ones_like(u.data.indices)
                if not hasattr(v.value, "__len__")
                else v.value
            )
            status = Status.Failed
        else:
            values = v.value
            status = Status.Converged

        self.contributions = values

        self.get_contributions()

    def test(self):
        pass