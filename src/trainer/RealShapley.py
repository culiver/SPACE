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
from utils import contribution_eval, get_subsidy_realshap
from .central import Central

import itertools
import math

class RealShapley(Central):
    def __init__(self, args, my_model, ckp):
        super().__init__(args, my_model, ckp)

    def get_contributions(self):
        '''
        Shapley value calcuation implemented by pyDVL
        '''
        input_params = {'subsets_info':self.subsets_info}
        self.contributions = contribution_eval(input_params, solution_concept=self.args.solution_concept, u_trans=self.args.u_trans, k=self.args.k, T=self.args.T)
        
        save_json(self.args, 'contributions.json', self.contributions.tolist())
        save_json(self.args, 'contributions_k[{}]_T[{}].json'.format(self.args.k, self.args.T), self.contributions.tolist())
        self.ckp.write_log("Contribution of clients: {}".format(self.contributions))

    def train(self, user_shards=None):
        print('='*20)
        print('Start Calculate Real Shapley value!')
        print('='*20)

        self.pruning_accuracy = [[]]
        self.subsets_info = get_json(self.args, 'subsets_info.json')
        self.subsets_info = {} if self.subsets_info is None else self.subsets_info
        self.subsets_info['num_users'] = self.args.num_users


        if self.args.gpu_id:
            torch.cuda.set_device(self.args.gpu_id)
        device = 'cuda' if self.args.gpu else 'cpu'
        subsets = []
        for i in range(1, self.args.num_users+1):
            for subset in itertools.combinations([j for j in range(self.args.num_users)], i):
                subsets.append(subset)

        for subset in subsets:
            subset = sorted(subset)
            subsetName = '{}'.format(subset[0])
            for item in subset[1:]:
                subsetName += '+{}'.format(item)
            if subsetName in self.subsets_info:
                continue
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

        self.get_contributions()

    def test(self):
        pass

