#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from clustered_sampling.plots_func.Fig_MNIST_shard import plot_fig_alg2

dataset = "MNIST_shard"
sampling = "clustered_1"
# n_SGD = 50
n_SGD = 100
seed = 0
lr = 0.01
decay = 1.0
p = 0.1
mu = 0.0
n_iter_plot = 200


from clustered_sampling.plots_func.Fig_CIFAR10 import plot_fig_CIFAR10_alpha_effect_one

smooth = True
plot_fig_CIFAR10_alpha_effect_one('acc', n_SGD, p, mu, smooth)
plot_fig_CIFAR10_alpha_effect_one('loss', n_SGD, p, mu, smooth)
