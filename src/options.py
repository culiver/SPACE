#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # Aggregate Mode
    parser.add_argument('--trainer', type=str, default='fedavg', help='trainer name')

    # Contribution Calcuation
    # For current implementation 'montecarlo_shapley', 'pruned_montecarlo_shapley' may be the same, while pruned_montecarlo_shapley may more suitable for large scale cases.
    parser.add_argument('--solution_concept', type=str, default='exact_shapley', 
                        choices=['exact_shapley', 'montecarlo_shapley', 'pruned_montecarlo_shapley','truncated_montecarlo_shapley', 'group_testing'],
                        help='the solution concept used for allocating clients payoff')
    parser.add_argument('--u_trans', type=int, default=1,
                        help="Set 0 to use model performance as utility function \
                              Set 1 to use sigmoid to transform original utility function ")
    parser.add_argument('--k', type=float, default=10.0,
                        help="Growth rate of logistic function")
    parser.add_argument('--T', type=float, default=1.0,
                        help="Threshold of logistic function")

    parser.add_argument('--iter_num', type=int, default=1,
                        help="Number of time renew data distributions, which is used for obtaining the variance of reweighting.")
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of communication rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_ep_ka', type=int, default=10,
                        help="the number of local epochs used for knowledge amalgamation: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_ka', type=float, default=2e-4,
                        help='learning rate of ka')
    parser.add_argument('--lr_min_ka', type=float, default=1e-6,
                        help='minimun learning rate of ka')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')

    # other arguments
    parser.add_argument('--app', type=str, default='None', help="name \
                        of dataset")
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu_id', type=int, default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--gpu', type=bool, default=True, help="To use cuda")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--optimizer_ka', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--dist', type=int, default=1,
                        help="Control of data distribution. Set to 0 for all non-IID, Set to 1 for IID, Set to 2 for partial determined shards non-IID, Set to 3 for partial random shards non-IID, Set to 4 for Dirichlet distribution.")
    parser.add_argument('--num_Noniid', type=int, default=0,
                        help='Number of Non-iid client.')
    parser.add_argument('--num_Mislabel', type=int, default=0,
                        help='Number of Mislabeled client.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--hvp', type=int, default=0, help='Set to 1 for DIG-FL with hessian matrix')

    # Log specifications
    parser.add_argument('--reset', action='store_true',
                    help='reset the training')
    parser.add_argument('--save', type=str, default='checkpoint',
                        help='file name to save')
    parser.add_argument('--load', type=str, default='',
                        help='file name to load')
    parser.add_argument('--resume', type=int, default=0,
                        help='resume from specific checkpoint')
    parser.add_argument('--save_models', action='store_true',
                        help='save all intermediate models')
    parser.add_argument('--skip_train', action='store_true',
                        help='skip training')

    # Knowledge amalgamation
    parser.add_argument('--t_num', type=int, default=100,
                        help="number of teacher in amalgamation, which is G in paper")
    parser.add_argument('--ka_bs', type=int, default=16,
                        help="number of rounds of training")
    parser.add_argument('--ka_ep', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--ka_warmup_ep', type=int, default=3,
                        help="number of users: K")
    parser.add_argument('--kd_loss_weight', type=float, default=1,
                        help="review kd loss weight")

    # Clustered Sampling
    parser.add_argument('--sampling', type=str, default='random', help="Sampling methods")
    parser.add_argument('--sim_type', type=str, default='cosine', help="used for the clients distance")
    parser.add_argument('--n_SGD', type=int, default=100, help="The number of SGD run locally")
    parser.add_argument('--lr_sampling', type=float, default=0.05, help="The lr run locally")
    parser.add_argument('--decay', type=float, default=1.0, help="The number of SGD run locally")
    parser.add_argument('--alpha', type=float, default=0.001, help="alpha for dirichlet distribution")
    parser.add_argument('--n_sampled', type=int, default=10, help="number of clients selected")
                        

    # Hardware specifications
    parser.add_argument('--n_threads', type=int, default=20,
                        help='number of threads for data loading')
    parser.add_argument('--cpu', action='store_true',
                        help='use cpu only')
    parser.add_argument('--n_GPUs', type=int, default=1,
                        help='number of GPUs')
    parser.add_argument('--RPC', action='store_true',
                        help='Use RPC for parallel')

    args = parser.parse_args()
    return args
