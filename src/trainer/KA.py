import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from utils import get_dataset, average_weights, exp_details
from torch.utils.data import DataLoader, Dataset

import matplotlib
import matplotlib.pyplot as plt

from FKA import FKAModule
import json
from utils import get_json
from .central import Central
from warmup_scheduler import GradualWarmupScheduler

class KA(Central):
    def __init__(self, args, my_model, ckp):
        super().__init__(args, my_model, ckp)
    
    def amalgamate(self, test_dataset, all_local_weights, t_num=100):
        print_every = 1
        self.criterion_CE = nn.NLLLoss().to(self.device)
        self.criterion_L1 = nn.L1Loss().to(self.device)
        self.criterion_L1_none = nn.L1Loss(reduction='none').to(self.device)
        self.softmax = torch.nn.Softmax(dim=0)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        while len(all_local_weights) > 1:
            local_weights = all_local_weights[:t_num]
            all_local_weights = all_local_weights[t_num:]
            
            self.ka_loss = []
            fka_modules = nn.ModuleList([])
            if self.args.dataset == 'mnist':
                hidden_sizes = [10, 20]
                reduce_rate = 2
            else:
                if self.args.model == 'cnn':
                    hidden_sizes = [64, 128, 128, 256, 512] # Only distill cnn part
                    reduce_rate = 16
                elif self.args.model == 'cnnSimple':
                    hidden_sizes = [32, 64, 64] # Only distill cnn part
                    reduce_rate = 8

            for c in hidden_sizes: 
                fka_modules.append(FKAModule(channel_t=c, channel_s=c, channel_h=c//reduce_rate, n_teachers=len(local_weights)).to(self.device))
            # Dataset
            testloader = DataLoader(self.test_dataset, batch_size=self.args.ka_bs, shuffle=False)

            # Set optimizer for the local updates
            linear_scaled_lr = self.args.lr_ka * self.args.ka_bs / 16
            optimizer = torch.optim.Adam(list(self.model.parameters())+list(fka_modules.parameters()), lr=linear_scaled_lr,
                                        betas=(0.9, 0.999), eps=1e-8,)
                                        # weight_decay=1e-4)

            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.ka_ep - self.args.ka_warmup_ep, eta_min=self.args.lr_min_ka)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=self.args.ka_warmup_ep, after_scheduler=scheduler_cosine)
            scheduler.step()

            teacher_list = []
            for w in local_weights:
                teacher_list.append(copy.deepcopy(self.model).eval())
                teacher_list[-1].load_state_dict(w)

            self.model.eval()
            for iter in range(self.args.ka_ep):
                batch_loss = []
                # Make sure all param can be updated
                for name, param in self.model.named_parameters():
                    param.requires_grad = True

                for batch_idx, (images, labels) in enumerate(testloader):
                    losses = {}
                    images, labels = images.to(self.device), labels.to(self.device)

                    features_from_student, s_pred = self.model(images, is_feat = True, preact=True)

                    t_features_list = []
                    s_features_list = []
                    t_pred_list = []
                    for idx, teacher in enumerate(teacher_list):
                        with torch.no_grad():
                            t_features, t_pred = teacher(images, is_feat=True, preact=True)
                            t_features_list.append(t_features)
                            t_pred_list.append(t_pred) 

                    # Calculate the score of teachers and find best
                    all_logits = torch.stack(t_pred_list) # (t, b, c)
                    best_model_idx = torch.argmax(all_logits[:, torch.arange(s_pred.shape[0]), labels], dim=0) #(b)
                    weight_t = self.softmax(all_logits[:, torch.arange(s_pred.shape[0]), labels]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (t, b)

                    # stack t_pred_list to get (b, t, c) and select best model for each sample in batch as golden answer
                    t_pred_list = torch.stack(t_pred_list, dim=1)
                    golden_pred = t_pred_list[torch.arange(s_pred.shape[0]), best_model_idx]
                    features_from_teachers = []
                    # shape of features_from_teachers = (layer, t, b, c, h, w)
                    for layer in range(len(t_features_list[0])):
                        features_from_teachers.append([t_features_list[i][layer] for i in range(len(local_weights))])

                    L_feat, L_rec = 0., 0.
                    for i, (s_features, t_features, fka_module) in enumerate(zip(features_from_student, features_from_teachers, fka_modules)):
                        t_proj_features, t_recons_features, s_proj_features = fka_module(t_features, s_features)
                        t_proj_features = torch.stack(t_proj_features)
                        feat_dist = self.criterion_L1_none(s_proj_features.unsqueeze(0).expand_as(t_proj_features), t_proj_features)
                        L_feat += (feat_dist * weight_t.expand_as(feat_dist)).mean() / len(features_from_student)
                        L_rec += 0.05 * self.criterion_L1(torch.stack(t_recons_features), torch.stack(t_features))

                    L_kl = self.kl_loss(s_pred, golden_pred)
                    loss = L_kl + L_feat + L_rec

                    self.model.zero_grad()
                    self.ka_loss.append(loss.item())

                    loss.backward()
                    optimizer.step()

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| KA Epoch : {} | [{}/{} ({:.0f}%)]\L_kl: {:.6f} L_feat: {:.6f} L_rec: {:.6f}'.format(
                            iter, batch_idx * len(images),
                            len(testloader.dataset),
                            100. * batch_idx / len(testloader), L_kl.item(), L_feat.item(), L_rec.item()))

                test_acc, test_loss = test_inference(self.args, self.model, self.test_dataset)
                
                # print global training loss after every 'i' rounds
                if (iter+1) % print_every == 0:
                    self.ckp.write_log(f' \nAvg Training Stats after {iter+1} KA rounds:')
                    self.ckp.write_log(f'Training Loss : {np.mean(np.array(self.ka_loss))}')
                    self.ckp.write_log("|---- Test Accuracy: {:.2f}%\n".format(100*test_acc))

                self.ckp.add_log(test_acc)
                self.ckp.save(self.model, iter, is_best=(self.ckp.log.index(max(self.ckp.log)) == iter))

                scheduler.step()

            all_local_weights.append(self.model.state_dict()) 

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
        self.num_class = len(set([d[1] for d in self.test_dataset]))

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
        print_every = 2
        val_loss_pre, counter = 0, 0
        idxs_users = np.array(range(self.args.num_users))

        local_weights, local_losses = self.local_training(global_round=0, idxs_users=idxs_users, is_ka=True)
        
        # Knowledge Amalgamation
        self.amalgamate(self.test_dataset, local_weights, t_num=self.args.t_num)
        self.get_contributions()


    def test(self):
        pass



