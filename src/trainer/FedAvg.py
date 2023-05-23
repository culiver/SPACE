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
from utils import get_dataset, average_weights, exp_details

import matplotlib
import matplotlib.pyplot as plt
import json
from utils import get_json
from .central import Central

class FedAvg(Central):
    def __init__(self, args, my_model, ckp):
        super().__init__(args, my_model, ckp)

    def train(self):
        if get_json(self.args, 'contributions.json') is not None and self.args.skip_train:
            self.model.load(
                self.ckp.get_path('model'),
                pre_train=self.args.pre_train,
                resume=-1,
                cpu=self.args.cpu
            )
            self.get_contributions()
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
            local_weights, local_losses = self.local_training(global_round=epoch, idxs_users=idxs_users)

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

