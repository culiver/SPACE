#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from grad_utils import sum_grad, hvp

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, return_idxs=False):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.return_idxs = return_idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.return_idxs:
            return image.clone().detach(), torch.tensor(label), self.idxs[item]
        else:
            return image.clone().detach(), torch.tensor(label)

    def get_labels(self):
        labels = {}
        for i in self.idxs:
            image, label = self.dataset.__getitem__(i)
            labels[label] = labels.get(label, 0) + 1
        return labels

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, is_ka=False):
        self.args = args
        self.logger = logger
        self.dataset = dataset
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)
        self.is_ka = is_ka
        self.data_stats = {}

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        # idxs_train = idxs[:int(0.8*len(idxs))]
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        # idxs_test = idxs[int(0.9*len(idxs)):]
        idxs_train = idxs[:]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        # validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                          batch_size=int(len(idxs_val)/10), shuffle=False)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=int(len(idxs_test)/10), shuffle=False)
        validloader = None
        testloader = None
        return trainloader, validloader, testloader

    def data_split_by_forget(self):
        score_list = [(self.data_stats[key]['forget_score'], key) for key in self.data_stats]
        score_list.sort()

        normal_list = [pair[1] for pair in score_list[:int(len(score_list)*0.8)]]
        boundary_list = [pair[1] for pair in score_list[int(len(score_list)*0.8):]]

        trainloader_normal = DataLoader(DatasetSplit(self.dataset, normal_list),
                                 batch_size=self.args.local_bs, shuffle=True)
        trainloader_boundary = DataLoader(DatasetSplit(self.dataset, boundary_list),
                                 batch_size=self.args.local_bs, shuffle=True)
        return trainloader_normal, trainloader_boundary

    def calc_Omega(self, model, prev_grads, gpu=-1):
        hv_s = []
        if gpu == -1:
            model.to('cpu')
        else:
            model.to(self.device)
        sum_prev_grads = sum_grad(prev_grads)
        if gpu > -1:
            sum_prev_grads = [p.to(self.device) for p in sum_prev_grads ]
        hv_s.append(hvp(self.dataset, model, sum_prev_grads, self.criterion, gpu=gpu))
        torch.cuda.empty_cache()
        return sum_grad(hv_s)

    def update_weights(self, model, global_round):
        # if self.args.verbose:
        # print('Client labels:' + str(np.unique(np.array([d[1] for d in self.trainloader.dataset]))))
        # Set mode to train model
        model.train()
        model.to(self.device)
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        local_ep = self.args.local_ep_ka if self.is_ka else self.args.local_ep
        local_ep = local_ep * 2 if self.is_ka and len(self.dataset) < 500 else local_ep
        for iter in range(local_ep):
            batch_loss = []
            for batch_idx, data in enumerate(self.trainloader):
                images, labels = data

                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        result_weight = {k: v.cpu() for k, v in model.state_dict().items()}
        return result_weight, sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

    def computeLocalPrototype(self, model, num_class=10):
        """ Returns prototype.
        """
        new_model = model

        new_model.eval()

        numData = len(self.trainloader.dataset)
        feat_size = None

        prototypes = []
        class_features = {}
        label_nums = {}
        label_nums_list = []
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            features = new_model(images, is_feat=True, preact=False)[0][-1]
            features = features.view(features.size(0), -1)

            for i in range(features.shape[0]):
                label = labels[i].item()
                feature_vector = features[i].detach().cpu().numpy()
                feat_size = feature_vector.shape
                class_features[label] = class_features.get(label, np.zeros(feat_size)) + feature_vector
                label_nums[label] = label_nums.get(label, 0) + 1

        # Compute the mean feature vector for each class
        for label in range(num_class):
            if label in class_features:
                prototypes.append(class_features[label] / label_nums[label])
                label_nums_list.append(label_nums[label])
            else:
                prototypes.append(np.zeros(feat_size)) 
                label_nums_list.append(0)
        print(label_nums_list)
        torch.cuda.empty_cache()
        return np.array(prototypes), np.array(label_nums_list)

    def computeLocalDistribution(self, model, num_class=10):
        """ Returns prototype.
        """
        new_model = model
        new_model.eval()

        numData = len(self.trainloader.dataset)
        feat_size = None

        mu_list = []
        sigma_list = []
        label_nums_list = []

        class_features = {}
        label_nums = {}
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            features = new_model(images, is_feat=True, preact=False)[0][-1]
            features = features.view(features.size(0), -1)

            for i in range(features.shape[0]):
                label = labels[i].item()
                feature_vector = features[i].detach().cpu().numpy()
                feat_size = feature_vector.shape[0]
                class_features.setdefault(label, []).append(feature_vector)
                label_nums[label] = label_nums.get(label, 0) + 1

        # Compute the mean feature vector for each class
        for label in range(num_class):
            if label in class_features:
                mu_list.append(np.array(class_features[label]).mean(axis=0))
                sigma_list.append(np.cov(np.array(class_features[label]), rowvar=False))
                label_nums_list.append(label_nums[label])
            else:
                mu_list.append(np.zeros(feat_size)) 
                sigma_list.append(np.zeros((feat_size, feat_size))) 
                label_nums_list.append(0)
        return np.array(mu_list), np.array(sigma_list), np.array(label_nums_list)



def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    loss = loss/total
    return accuracy, loss
