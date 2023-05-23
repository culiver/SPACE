#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
# from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, get_save_path
import utils

from trainer.FedAvg import FedAvg
from trainer.KA import KA
from trainer.RealShapley import RealShapley
from trainer.DIGFL import DIGFL
from trainer.TMCShapley import TMCShapley
from trainer.GroupTesting import GroupTesting
import models
import json

import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from datetime import datetime
from trainer.central import run_trainer
import random


def run_ps():
    start_time = time.time()
    args = args_parser()
    exp_details(args)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    checkpoint = utils.checkpoint(args)

    _model = models.Model(args, checkpoint)
    if args.trainer == 'realshapley':
        t = RealShapley(args, _model, checkpoint)
    elif args.trainer == 'fedavg':
        t = FedAvg(args, _model, checkpoint)
    elif 'ka' in args.trainer:
        t = KA(args, _model, checkpoint)
    elif 'digfl' in args.trainer:
        t = DIGFL(args, _model, checkpoint)
    elif args.trainer == 'tmc':
        t = TMCShapley(args, _model, checkpoint)
    elif args.trainer == 'gt':
        t = GroupTesting(args, _model, checkpoint)
    for iter in range(args.iter_num):
        print('='*20)
        print('Iter {} Start!'.format(iter))
        print('='*20)
        t.renew_dataset(iter)
        t.train()
        if 'Reweighting' in args.app:
            t.train_with_reweight(dynamic='dynamic' in args.app)
        if 'sampling' in args.app:
            t.train_with_sampling()


    t.writer.close()
    # print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    checkpoint.write_log('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    

def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options=rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=0,  # infinite timeout
        _transports=["uv"],
     )
    if rank != 0:
        rpc.init_rpc(
            f"trainer{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        # trainer passively waiting for ps to kick off training iterations
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_ps()

    # block until all rpcs finish
    rpc.shutdown()

if __name__ == '__main__':
    args = args_parser()
    if args.RPC:
        world_size = args.num_users + 1
        print(world_size)
        mp.spawn(main, args=(world_size, ), nprocs=world_size, join=True)
    else:
        run_ps()
    
