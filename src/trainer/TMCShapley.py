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

class TMCShapley(Central):
    def __init__(self, args, my_model, ckp):
        super().__init__(args, my_model, ckp)

        self.converge_min = max(30, self.args.num_users)
        self.max_percentage = 0.1
        if self.args.dataset == 'cifar':
            constant = 2
        elif self.args.dataset == 'mnist':
            constant = 10
        else:
            constant = 1

        # set the max iteration number to O(nlogn)
        self.max_number = min(
            2**self.args.num_users,
            constant * int(self.args.num_users * np.log10(self.args.num_users)),
        )
        self.last_k = 10
        self.converge_criteria = 0.05

    def get_contributions(self,contribution_records, fullset_metric):
        round_shapley_values = np.sum(contribution_records, 0) / len(
            contribution_records
        )
        assert len(round_shapley_values) == self.args.num_users

        round_marginal_gain = fullset_metric - 0
        round_shapley_value_dict = dict()
        for idx, value in enumerate(round_shapley_values):
            round_shapley_value_dict[idx] = float(value)

        
        shapley_values = self.normalize_shapley_values(
            round_shapley_value_dict, round_marginal_gain
        )

        self.contributions = [item for item in shapley_values.values()]
        
        save_json(self.args, 'contributions.json', self.contributions)
        save_json(self.args, 'contributions_k[{}]_T[{}].json'.format(self.args.k, self.args.T), self.contributions)
        self.ckp.write_log("Contribution of clients: {}".format(self.contributions))

    def train(self, user_shards=None):
        print('='*20)
        print('Start Calculate Truncated Monte Carlo Shapley value!')
        print('='*20)

        self.args.trainer = 'realshapley'
        self.subsets_info = get_json(self.args, 'subsets_info.json')
        self.subsets_info = {} if self.subsets_info is None else self.subsets_info
        self.subsets_info['num_users'] = self.args.num_users

        self.args.trainer = 'tmc'

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

        subset = [i for i in range(self.args.num_users)]
        subsetName ='+'.join([str(elem) for elem in [i for i in range(self.args.num_users)]])
        if subsetName not in self.subsets_info:
            trainSubset(subset)
        fullset_metric = self.subsets_info[subsetName]

        contribution_records: list = []
        t = 0
        while self.not_convergent(t, contribution_records):
            t = t + 1

            v: list = [0] * (self.args.num_users + 1)
            v[0] = 0 # last round metric
            marginal_contribution = [0 for i in range(self.args.num_users)]

            perturbed_indices = np.random.permutation([i for i in range(self.args.num_users)])

            for j in range(1, self.args.num_users + 1):
                subset = sorted(perturbed_indices[:j].tolist())

                subsetName = '{}'.format(subset[0])
                for item in subset[1:]:
                    subsetName += '+{}'.format(item)
                if abs(fullset_metric - v[j - 1]) >= eps:
                    if subsetName not in self.subsets_info:
                        trainSubset(subset)
                    v[j] = self.subsets_info[subsetName]
                else:
                    v[j] = v[j - 1]
                # update SV
                if self.args.u_trans:
                    v_j = stable_sigmoid((v[j]-self.args.T) * self.args.k) if (v[j] != self.args.T) else 0.5    
                    v_j_1 = stable_sigmoid((v[j-1]-self.args.T) * self.args.k) if (v[j-1] != self.args.T) else 0.5    
                    marginal_contribution[perturbed_indices[j - 1]] = v_j - v_j_1
                else:
                    marginal_contribution[perturbed_indices[j - 1]] = v[j] - v[j - 1]
            contribution_records.append(marginal_contribution)
        print("trauncated monte carlo finished!, run iterations: {}".format(t))
        self.ckp.write_log("TMC sampling iterations: {}".format(t))
        self.get_contributions(contribution_records, fullset_metric)

    def test(self):
        pass

    def not_convergent(self, index, contribution_records):
        if index >= self.max_number:
            return False
        return True
    
    def normalize_shapley_values(self, shapley_values: dict, marginal_gain: float) -> dict:
        sum_value: float = 0
        if marginal_gain >= 0:
            sum_value = sum([v for v in shapley_values.values() if v >= 0])
            if math.isclose(sum_value, 0):
                sum_value = 1e-9
        else:
            sum_value = sum([v for v in shapley_values.values() if v < 0])
            if math.isclose(sum_value, 0):
                sum_value = -1e-9

        return {k: marginal_gain * v / sum_value for k, v in shapley_values.items()}