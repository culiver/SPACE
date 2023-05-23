#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import torch
import time
import datetime
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, mnist_noniid_specified, cifar_noniid_specified
from sampling import cifar_iid, cifar_noniid, partition_CIFAR_dataset
# import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, NamedTuple, Optional, Sequence, Tuple

from pydvl.value.stopping import MaxUpdates
import cvxpy as cp
import math
import random

# Extended pydvl
from pydvl_extend.shapley.pruned import RelativePruning
from pydvl_extend.shapley import compute_shapley_values, ShapleyMode

from utilities import Utility_Func_cosine, Utility_Func_fid, Utility_Func_RealShap

try:
    import ujson as json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        import json

def get_save_path(args):
    args = vars(args)
    if args['dist'] == 4:
        required_attr = ['epochs', 'local_ep', 'iter_num', 'num_users', 'dist', 'alpha']
    else:
        required_attr = ['epochs', 'local_ep', 'iter_num', 'num_users', 'dist', 'num_Noniid', 'num_Mislabel']
    dirname = '../save/{}_{}'.format(args['trainer'].upper(), args['dataset'])
    for attr in required_attr:
        attrName = attr.replace('_', '')
        dirname += '_{}[{}]'.format(attrName, args[attr])
    return dirname

def save_json(args, filename, data):
    # Saving the accuracy of converged federated learning training:
    save_path = get_save_path(args)
    file_name = os.path.join(save_path, filename)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    # print(subsets_info)
    with open(file_name, 'w') as f:
        json.dump(data, f)

def get_json(args, filename):
    # Saving the accuracy of converged federated learning training:
    save_path = get_save_path(args)
    file_name = os.path.join(save_path, filename)
    if os.path.isfile(file_name):
        with open(file_name, newline='') as f:
            data = json.load(f)
        return data
    else:
        return None


def plot_loss_curve(args, reweight_info):
    plot_info = {'normal':{'color':'r'}, 'dynamic_reweight':{'color':'g'}, 'static_reweight':{'color':'b'}}

    plt.figure()
    
    for label, info in reweight_info.items():
        if 'test_losses' not in info:
            continue
        loss_curve = info['test_losses']
        x = list(range(len(loss_curve)))

        save_path = get_save_path(args)

        plt.plot(x, np.array(loss_curve), label=label, color=plot_info[label]['color'])

    plt.title('Comparison of Remove High, Remove Low and Remove Randomly')
    plt.ylabel('Average Loss')
    plt.xlabel('Epochs')

    plt.legend()
    img_name = os.path.join(save_path, 'loss_curve.png')
    plt.savefig(img_name)


def get_user_shards_cifar(num_users, num_Noniid, distribution):
    shards_bank = [[j for j in range(i*20, (i+1)*20)] for i in range(10)]
    assert num_users >= num_Noniid
    user_shards = []
    ptr = 0

    if distribution == 2:
        user_num_classes = []
        for i in range(num_users):
            if i >= (num_users - num_Noniid): # if it is a non-iid one
                user_num_classes.append(max(len(shards_bank)-i, 1))
            else: # if it is a normal one
                user_num_classes.append(len(shards_bank))

        for user_num_class in user_num_classes:
            temp_shard = []
            for i in range(user_num_class):
                idx = (i + ptr) % len(shards_bank)
                temp_shard.append(shards_bank[idx][0])
                shards_bank[idx] = shards_bank[idx][1:]
            user_shards.append(temp_shard)
            ptr = (ptr + user_num_class) % len(shards_bank)
    
    elif distribution == 3:
        # shards_per_class = max(20 // num_users, 1)
        shards_per_class = 1
        for i in range(num_users):
            remain_shards = shards_per_class * len(shards_bank)
            temp_shard = []
            if i < (num_users - num_Noniid): # if it is a normal one
                for j in range(len(shards_bank)):
                    # random without replacement
                    temp_shard += random.sample(shards_bank[j], shards_per_class)
                    shards_bank[j].remove(temp_shard[-1])
            else: # if it is a non-iid one
                j = 0
                while remain_shards: 
                    target_shard_length = len(shards_bank[ptr % len(shards_bank)])
                    if target_shard_length == 0:
                        ptr += 1
                        continue
                    selected_num = random.randint(1, min(remain_shards, target_shard_length))
                    selected_shards = random.choices(shards_bank[ptr % len(shards_bank)], k=selected_num)
                    shards_bank[ptr % len(shards_bank)] = [elem for elem in shards_bank[ptr % len(shards_bank)] if elem not in selected_shards]
                    remain_shards -= selected_num
                    temp_shard += selected_shards
                    ptr += 1
            user_shards.append(temp_shard)
    return user_shards


def get_user_shards_mnist(num_users, num_Noniid, distribution):
    shards_bank = [
        [i for i in range(0,19)],
        [i for i in range(20,42)],
        [i for i in range(43,62)],
        [i for i in range(63,82)],
        [i for i in range(83,101)],
        [i for i in range(102,120)],
        [i for i in range(121,139)],
        [i for i in range(140,160)],
        [i for i in range(161,180)],
        [i for i in range(181,200)],
    ]
    assert num_users > num_Noniid
    user_shards = []
    ptr = 0

    if distribution == 2:
        user_num_classes = []
        for i in range(num_users):
            if i >= (num_users - num_Noniid):
                user_num_classes.append(max(len(shards_bank)-i, 1))
            else:
                user_num_classes.append(len(shards_bank))

        for user_num_class in user_num_classes:
            temp_shard = []
            for i in range(user_num_class):
                idx = (i + ptr) % len(shards_bank)
                temp_shard.append(shards_bank[idx][0])
                shards_bank[idx] = shards_bank[idx][1:] + shards_bank[idx][:1]
            user_shards.append(temp_shard)
            ptr = (ptr + user_num_class) % len(shards_bank)
    
    elif distribution == 3:
        # shards_per_class = max(20 // num_users, 1)
        shards_per_class = 1
        for i in range(num_users):
            remain_shards = shards_per_class * len(shards_bank)
            temp_shard = []
            if i < (num_users - num_Noniid): # if it is a normal one
                for j in range(len(shards_bank)):
                    # random without replacement
                    temp_shard += random.sample(shards_bank[j], shards_per_class)
                    shards_bank[j].remove(temp_shard[-1])
            else: # if it is a non-iid one
                j = 0
                while remain_shards: 
                    target_shard_length = len(shards_bank[ptr % len(shards_bank)])
                    if target_shard_length == 0:
                        ptr += 1
                        continue
                    selected_num = random.randint(1, min(remain_shards, target_shard_length))
                    selected_shards = random.choices(shards_bank[ptr % len(shards_bank)], k=selected_num)
                    shards_bank[ptr % len(shards_bank)] = [elem for elem in shards_bank[ptr % len(shards_bank)] if elem not in selected_shards]
                    remain_shards -= selected_num
                    temp_shard += selected_shards
                    ptr += 1
            user_shards.append(temp_shard)
    return user_shards


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.dist == 4:
            alpha = args.alpha
            n_classes = 10
            n_clients = 100
            from numpy.random import dirichlet
            matrix = dirichlet([alpha] * n_classes, size=n_clients)
            while np.isnan(matrix).sum() > 0:
                matrix = dirichlet([alpha] * n_classes, size=n_clients)

            user_groups = partition_CIFAR_dataset(
                train_dataset,
                False, # Balance is False
                matrix,
                n_clients,
                n_classes,
                True,
            )
            user_groups = {k:v for k, v in user_groups.items() if k < args.num_users}

        elif args.dist == 2 or args.dist == 3:
            user_shards = get_user_shards_cifar(args.num_users, args.num_Noniid, args.dist)
            user_groups = cifar_noniid_specified(train_dataset, args.num_users, user_shards)
        elif args.dist == 1:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.dist == 2 or args.dist == 3:
            user_shards = get_user_shards_mnist(args.num_users, args.num_Noniid, args.dist)
            user_groups = mnist_noniid_specified(train_dataset, args.num_users, user_shards)
        elif args.dist == 1:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w, weightings=None):
    """
    Returns the average of the weights.
    """
    
    if weightings is not None:
        weightings[weightings < 0] = 0
        if weightings.sum() != 0:
            weightings = weightings / weightings.sum()
        else:
            weightings = [1/len(w) for i in range(len(w))]
    else:
        weightings = [1/len(w) for i in range(len(w))]
    
    print('Client Weightings: {}'.format(weightings))
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * weightings[0]
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * weightings[i]
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.dist:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = []
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            save_path = get_save_path(args)
            self.dir = os.path.join(save_path, args.save)
        else:
            self.dir = os.path.join(save_path, args.load)
            if os.path.exists(self.dir):
                self.log = json.load(self.get_path('fed_log.json'))['score']
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            # os.system('rm -rf ' + self.dir)
            os.system('rm ' + os.path.join(self.dir, 'log.txt'))
            os.system('rm ' + os.path.join(self.dir, 'config.txt'))
            os.system('rm ' + os.path.join(self.dir, 'fed_log.json'))
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        os.makedirs(self.get_path('results-{}'.format(args.dataset)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, model, epoch, is_best=False):
        model.save(self.get_path('model'), epoch, is_best=is_best)
        with open(self.get_path('fed_log.json'), 'w', newline='') as jsonfile:
            json.dump({'score':self.log}, jsonfile)

    def add_log(self, score):
        self.log.append(score)

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()



def contribution_eval(input_params, metric='cosine', budget=200, solution_concept='core', u_trans=False, k=10, T=1.0):
    epsilon = 1e-8
    if 'subsets_info' in input_params:
        utility = Utility_Func_RealShap(input_params['subsets_info'], u_trans=u_trans, k=k, T=T)
    elif 'serverMu' in input_params:
        utility = Utility_Func_fid(input_params['serverMu'], input_params['serverSigma'], input_params['clientMus'], input_params['clientSigmas'], input_params['client_data_nums'], u_trans=u_trans, k=k, T=T)
    else:
        utility = Utility_Func_cosine(input_params['serverPrototype'], input_params['clientPrototypes'], input_params['client_data_nums'], u_trans=u_trans, k=k, T=T)

    if 'shapley' in solution_concept:
        if 'exact' in solution_concept:
            values = compute_shapley_values(
                u=utility,
                mode=ShapleyMode.CombinatorialExact,
            )
        elif 'pruned' in solution_concept:
            values = compute_shapley_values(
                u=utility,
                mode=ShapleyMode.PrunedMontecarlo,
                n_jobs=1,
                done=MaxUpdates(1000),
                pruning=RelativePruning(u=utility, atol=0)
            )
        elif 'truncated' in solution_concept:
            values = compute_shapley_values(
                u=utility,
                mode=ShapleyMode.TruncatedMontecarlo,
                n_jobs=1,
                done=MaxUpdates(1000),
            )
        elif 'group' in solution_concept:
            values = compute_shapley_values(
                u=utility,
                mode=ShapleyMode.GroupTesting,
                n_jobs=1,
            )
        else:
            values = compute_shapley_values(
                u=utility,
                mode=ShapleyMode.PermutationMontecarlo,
                n_jobs=1,
                done=MaxUpdates(1000),
                pruning=RelativePruning(u=utility, atol=0.01)
            )
        return values.values
