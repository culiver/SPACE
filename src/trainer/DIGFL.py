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
# from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, get_json

import matplotlib
import matplotlib.pyplot as plt
import json
from .central import Central
from grad_utils import calc_grad, DIG_FL
from torch import nn

class DIGFL(Central):
    def __init__(self, args, my_model, ckp):
        super().__init__(args, my_model, ckp)
        self.criterion = nn.NLLLoss().to(self.device)
        self.all_round_contributions = []
        self.local_gradients = [[[ p.cpu() for p in self.model.parameters() if p.requires_grad ]] for i in range(self.args.num_users)]


    def get_current_contributions(self, local_weights, local_Omegas=None, Normalize=False, version='original'):
        # self implementation for adding hessian
        if version == 'self_implement':
            gradient_v = calc_grad(self.test_dataset, self.model, self.criterion, gpu=0)
            contribution_per_round = []
            for i, local_weight in enumerate(local_weights):
                delta = {}
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        delta[name] = param - local_weight[name].to(self.device)
                if local_Omegas is not None:
                    current_gradient = []

                score = 0
                for idx, (g, d) in enumerate(zip(gradient_v, delta.values())):
                    score += torch.sum(g.view(-1) * d.view(-1)).item() / self.args.num_users
                    if local_Omegas is not None:
                        local_Omegas[i][idx] = local_Omegas[i][idx].to(self.device)
                        score += - self.args.lr * torch.sum(g.view(-1).to(self.device) * local_Omegas[i][idx].reshape(-1).to(self.device)).item()
                        current_gradient.append((-1/self.args.num_users*d.to(self.device) - self.args.lr*local_Omegas[i][idx].to(self.device)).to('cpu'))

                if local_Omegas is not None:
                    self.local_gradients[i].append(current_gradient)

                contribution_per_round.append(max(score,0))

        # Original implementation from https://github.com/qmkakaxi/DIG_FL
        elif version == 'original':
            w_glob = copy.deepcopy(self.model.state_dict())
            contribution_per_round = DIG_FL(local_weights, w_glob, self.model, self.test_dataset, self.device)

        if Normalize:
            sum_contirbution = sum(contribution_per_round)+0.00001
            for j in range(len(contribution_per_round)):
                contribution_per_round[j]=contribution_per_round[j]/sum_contirbution
        
        return np.array(contribution_per_round)

    def train(self):
        if get_json(self.args, 'contributions.json') is not None and self.args.skip_train:
            return None

        # Set the model to train and send it to device.
        self.model = copy.deepcopy(self.init_model)
        self.model.to(self.device)
        self.model.train()
        self.model.load(
            self.ckp.get_path('model'),
            pre_train=self.args.pre_train,
            resume=self.args.resume,
            cpu=self.args.cpu
        )

        # copy weights
        global_weights = self.model.state_dict()

        # Training
        self.train_loss, self.train_accuracy = [], []
        val_acc_list, net_list = [], []
        cv_loss, cv_acc = [], []
        print_every = 1
        val_loss_pre, counter = 0, 0

        for epoch in tqdm(range(self.args.epochs)):
            idxs_users = np.array(range(self.args.num_users))
            if self.args.hvp:
                local_weights, local_losses, local_Omegas = self.local_training(global_round=epoch, idxs_users=idxs_users)
            else:
                local_weights, local_losses = self.local_training(global_round=epoch, idxs_users=idxs_users)
                local_Omegas = None

            per_round_contributions = self.get_current_contributions(local_weights, local_Omegas)
            self.all_round_contributions.append(per_round_contributions.tolist())

            # update global weights
            global_weights = average_weights(local_weights, weightings=self.client_data_ratio)

            # update global weights
            self.model.load_state_dict(global_weights)

            loss_avg = (np.array(local_losses) * self.client_data_ratio).sum()
            self.train_loss.append(loss_avg)

            test_acc, test_loss = test_inference(self.args, self.model, self.test_dataset)

            # print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                self.ckp.write_log(f' \nAvg Training Stats after {epoch+1} global rounds:')
                self.ckp.write_log(f'Training Loss : {np.mean(np.array(self.train_loss))}')
                self.ckp.write_log("|---- Test Accuracy: {:.2f} \n%".format(100*test_acc))

            self.ckp.add_log(test_acc)
            self.ckp.save(self.model, epoch, is_best=(self.ckp.log.index(max(self.ckp.log)) == epoch))

        # Test inference after completion of training
        test_acc, test_loss = test_inference(self.args, self.model, self.test_dataset)
        
        self.ckp.write_log(f' \n Results after {self.args.epochs} global rounds of training:')
        self.ckp.write_log("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
        self.reweight_info[self.current_iter]['normal']['accuracy'] = test_acc

        self.get_contributions()

    def test(self):
        pass

