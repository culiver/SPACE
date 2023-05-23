import os
import copy
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from update import LocalUpdate, test_inference
from utils import get_dataset, average_weights, exp_details
from utils import contribution_eval, save_json, plot_loss_curve, get_json
from tensorboardX import SummaryWriter

import torch.distributed.rpc as rpc
import itertools
import math
import random

def run_trainer(args, train_dataset, idxs, model, global_round, is_ka, local_gradient=None):
    writer = SummaryWriter('../logs/{}'.format(rpc.get_worker_info().name))
    local_model = LocalUpdate(args=args, dataset=train_dataset,
                            idxs=idxs, logger=writer, is_ka=is_ka)
    if args.hvp:
        Omega = local_model.calc_Omega(model, local_gradient)
    print("Data of client {}: {}, total {} samples.".format(rpc.get_worker_info().id-1, np.unique(np.array([d[1] for d in local_model.trainloader.dataset])), len(local_model.trainloader.dataset)))
    w, loss = local_model.update_weights(
        model=model, global_round=global_round)
    local_model.logger.close()

    for k, v in w.items():
        w[k] = v.cpu()
    
    torch.cuda.empty_cache()
    return_info = {'weight':w, 'loss':loss, 'data_stats':local_model.data_stats}

    if args.hvp:
        for i in range(len(Omega)):
            Omega[i] = Omega[i].cpu()
        return_info['Omega'] = Omega

    return return_info

class Central():
    def __init__(self, args, my_model, ckp):
        self.args = args
        self.model = copy.deepcopy(my_model)
        self.init_model = copy.deepcopy(my_model)
        self.ckp = ckp
        path_project = os.path.abspath('..')
        self.writer = SummaryWriter('../logs')
        self.device = 'cuda' if args.gpu else 'cpu'
        self.is_ka = self.args.trainer == 'ka'
        self.contributions_allIter = []

        prev_reweight_info = get_json(self.args, 'reweight_info.json')
        self.reweight_info = [{'normal':{}, 'dynamic_reweight':{}, 'static_reweight':{}}] if (prev_reweight_info is None) else prev_reweight_info
        
        prev_contribution = get_json(self.args, 'contributions.json')
        if prev_contribution is not None and self.args.skip_train:
            self.contributions = np.array(prev_contribution)

    '''
    ===================================
            Data Preprocessing
    ===================================
    '''
    def renew_dataset(self, iterNum):
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(self.args)
        self.change_data_label()
        save_json(self.args, 'user_groups_{}.json'.format(iterNum), {k:np.array(list(v)).astype(int).tolist() for k, v in self.user_groups.items()})
        self.current_iter = iterNum
        if len(self.reweight_info) < iterNum+1:
            self.reweight_info.append({'normal':{}, 'dynamic_reweight':{}, 'static_reweight':{}})
        self.client_data_num = np.array([len(self.user_groups[i]) for i in range(self.args.num_users)])
        self.client_data_ratio = self.client_data_num / self.client_data_num.sum()

    def change_data_label(self, ratio=0.5, num_classes=10):
        corrupt_clients = [i for i in range(self.args.num_users-self.args.num_Mislabel, self.args.num_users)]
        for client in corrupt_clients:
            user_group = self.user_groups[client].astype(int)
            indices_to_change = random.sample(user_group.tolist(), int(len(user_group)*ratio))
            noise = np.random.randint(num_classes-1, size=len(indices_to_change))
            if isinstance(self.train_dataset.targets, list):
                new_labels = (np.array(self.train_dataset.targets)[indices_to_change] + noise) % num_classes
                for idx, indice in enumerate(indices_to_change):
                    self.train_dataset.targets[indice] = new_labels[idx]
            else:
                new_labels = (self.train_dataset.targets[indices_to_change] + torch.from_numpy(noise)) % num_classes
                self.train_dataset.targets[indices_to_change] = new_labels

    '''
    ===================================
            Contribution Evaluation
    ===================================
    '''
    def computeServerPrototype(self, num_class=10):
        """ Returns prototype.
        """
        new_model = self.model
        new_model.eval()

        class_features = {}
        label_nums = {}
        prototypes = []

        device = 'cuda' if self.args.gpu else 'cpu'
        testloader = DataLoader(self.test_dataset, batch_size=128,
                                shuffle=False)

        numData = torch.tensor(len(testloader.dataset))

        for batch_idx, (images, labels) in enumerate(testloader):
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
            else:
                prototypes.append(np.zeros(feat_size))

        torch.cuda.empty_cache()

        return np.array(prototypes)

    def computeServerDistribution(self, num_class=10):
        """ Returns prototype.
        """
        new_model = self.model
        new_model.eval()

        class_features = {}
        label_nums = {}
        
        mu_list = []
        sigma_list = []

        testloader = DataLoader(self.test_dataset, batch_size=128,
                                shuffle=False)

        numData = torch.tensor(len(testloader.dataset))

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            features = new_model(images, is_feat=True, preact=False)[0][-1]
            features = features.view(features.size(0), -1)
            
            for i in range(features.shape[0]):
                label = labels[i].item()
                feature_vector = features[i].detach().cpu().numpy()
                feat_size = feature_vector.shape
                class_features.setdefault(label, []).append(feature_vector)
                label_nums.setdefault(label, 0) + 1

        # Compute the mean feature vector for each class
        for label in range(num_class):
            if label in class_features:
                mu_list.append(np.array(class_features[label]).mean(axis=0))
                sigma_list.append(np.cov(np.array(class_features[label]), rowvar=False))
            else:
                prototypes.append(np.zeros(feat_size))
                mu_list.append(np.zeros(feat_size)) 
                sigma_list.append(np.zeros((feat_size, feat_size))) 

        torch.cuda.empty_cache()

        return np.array(mu_list), np.array(sigma_list) 

    def get_current_contributions(self, distance='cosine', local_weights=None):
        if 'realshapley' in self.args.trainer:
            return copy.deepcopy(self.contributions)

        num_class = len(set([d[1] for d in self.test_dataset]))
        if distance == 'cosine':
            serverPrototype = self.computeServerPrototype(num_class=num_class)
            clientPrototypes = []
            client_data_nums = []
            for idx in range(self.args.num_users):
                local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                        idxs=self.user_groups[idx], logger=self.writer)

                prototype, local_labels_list = local_model.computeLocalPrototype(model=self.model, num_class=num_class)
                clientPrototypes.append(prototype)
                client_data_nums.append(local_labels_list)
            
            input_params = {'serverPrototype':serverPrototype, 'clientPrototypes':np.array(clientPrototypes), 'client_data_nums':np.array(client_data_nums)}
            return contribution_eval(input_params, solution_concept=self.args.solution_concept, u_trans=self.args.u_trans, k=self.args.k, T=self.args.T)
            
        elif distance == 'fid':
            serverMu, serverSigma = self.computeServerDistribution(num_class=num_class)
            clientMus = []
            clientSigmas = []
            client_data_nums = []
            for idx in range(self.args.num_users):
                local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                        idxs=self.user_groups[idx], logger=self.writer)

                mus, sigmas, local_labels_list = local_model.computeLocalDistribution(model=self.model, num_class=num_class)
                clientMus.append(mus)
                clientSigmas.append(sigmas)
                client_data_nums.append(local_labels_list)

            input_params = {'serverMu':serverMu, 'serverSigma':serverSigma, 'clientMus':np.array(clientMus), 'clientSigmas':np.array(clientSigmas), 'client_data_nums':np.array(client_data_nums)}
            return contribution_eval(input_params, solution_concept=self.args.solution_concept, u_trans=self.args.u_trans, k=self.args.k, T=self.args.T)

    def get_contributions(self, distance='cosine'):
        num_class = len(set([d[1] for d in self.test_dataset]))
        if 'digfl' in self.args.trainer:
            self.contributions = (np.array(self.all_round_contributions)).sum(axis=0)
        else:
            self.contributions = self.get_current_contributions(distance=distance)

        self.contributions_allIter.append(self.contributions.tolist())
        save_json(self.args, 'contributions.json', self.contributions.tolist())
        save_json(self.args, 'contributions_k[{}]_T[{}].json'.format(self.args.k, self.args.T), self.contributions.tolist())
        save_json(self.args, 'contributions_allIter.json', self.contributions_allIter)

        self.ckp.write_log("Contribution of clients: {}".format(self.contributions))
    
    def local_training(self, global_round, idxs_users, is_ka=False):
        local_weights, local_losses = [], []
        if self.args.hvp:
            local_Omegas = []
        self.ckp.write_log(f'\n | Global Training Round : {global_round+1} |\n')
        self.model.train()

        if self.args.RPC:
            ps_rref = rpc.RRef(self)
            futs = []
            temp_model = copy.deepcopy(self.model)
            temp_model.to('cpu')
            for idx in idxs_users:
                trainer = "trainer{}".format(idx+1)
                if self.args.hvp:
                    futs.append(
                        rpc.rpc_async(trainer, run_trainer, args=(self.args, self.train_dataset,
                                    self.user_groups[idx], temp_model, global_round, is_ka, self.local_gradients[idx]))
                    )
                else:
                    futs.append(
                        rpc.rpc_async(trainer, run_trainer, args=(self.args, self.train_dataset,
                                    self.user_groups[idx], temp_model, global_round, is_ka))
                    )
            results = torch.futures.wait_all(futs)
            for result in results:
                local_weight = copy.deepcopy(result['weight'])
                local_weights.append(local_weight)
                local_losses.append(copy.deepcopy(result['loss']))
                if self.args.hvp:
                    local_Omegas.append(copy.deepcopy(result['Omega']))
            torch.cuda.empty_cache()
            
        else:
            for idx in idxs_users:
                local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                        idxs=self.user_groups[idx], logger=self.writer, is_ka=is_ka)
                if self.args.hvp:
                    Omega = local_model.calc_Omega(copy.deepcopy(self.model), self.local_gradients[idx])
                    local_Omegas.append(Omega)
                print("Data of client {}: {}, total {} samples.".format(idx, np.unique(np.array([d[1] for d in local_model.trainloader.dataset])), len(local_model.trainloader.dataset)))
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(self.model), global_round=global_round)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
                torch.cuda.empty_cache()
        if self.args.hvp:
            return local_weights, local_losses, local_Omegas
        else:
            return local_weights, local_losses

    '''
    ===================================
            Application
    ===================================
    '''

    def train_with_reweight(self, train_normal=True, dynamic=True):
        train_normal = False if 'test_losses' in self.reweight_info[self.current_iter]['normal'] else True
        if train_normal:
            print('='*20)
            print('Start Training with FedAvg!')
            print('='*20)

            idxs_users = np.arange(self.args.num_users)

            self.model = copy.deepcopy(self.init_model)
            self.model.to(self.device)
            self.model.train()
            # copy weights
            global_weights = self.model.state_dict()

            for epoch in range(self.args.epochs):
                local_weights, local_losses = self.local_training(global_round=epoch, idxs_users=idxs_users)

                # update global weights
                global_weights = average_weights(local_weights, weightings=self.client_data_ratio)

                # update global weights
                self.model.load_state_dict(global_weights)

                loss_avg = (np.array(local_losses) * self.client_data_ratio).sum()
                test_acc, test_loss = test_inference(self.args, self.model, self.test_dataset)
                self.reweight_info[self.current_iter]['normal'].setdefault('test_losses', []).append(test_loss)
                self.reweight_info[self.current_iter]['normal'].setdefault('train_losses', []).append(loss_avg)

            # Test inference after completion of training
            test_acc, test_loss = test_inference(self.args, self.model, self.test_dataset)
            self.ckp.write_log("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            self.reweight_info[self.current_iter]['normal']['accuracy'] = test_acc

        print('='*20)
        print('Start Training with Reweighting!')
        print('='*20)
        reweight_type = 'dynamic_reweight' if dynamic else 'static_reweight'
        self.reweight_info[self.current_iter][reweight_type]['test_losses'] = []

        idxs_users = np.arange(self.args.num_users)

        self.model = copy.deepcopy(self.init_model)
        self.model.to(self.device)
        self.model.train()
        # copy weights
        global_weights = self.model.state_dict()

        for epoch in range(self.args.epochs):
            local_weights, local_losses = self.local_training(global_round=epoch, idxs_users=idxs_users)
        
            # update global weights
            per_round_contributions = self.get_current_contributions(local_weights=local_weights) if dynamic else self.contributions
            self.ckp.write_log('Per_round_contributions: {}'.format(per_round_contributions))
            global_weights = average_weights(local_weights, weightings=per_round_contributions)

            # update global weights
            self.model.load_state_dict(global_weights)

            loss_avg = (np.array(local_losses) * self.client_data_ratio).sum()
            test_acc, test_loss = test_inference(self.args, self.model, self.test_dataset)
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            self.reweight_info[self.current_iter][reweight_type].setdefault('test_losses', []).append(test_loss)
            self.reweight_info[self.current_iter][reweight_type].setdefault('train_losses', []).append(loss_avg)
    
        # Test inference after completion of training
        test_acc, test_loss = test_inference(self.args, self.model, self.test_dataset)
        self.ckp.write_log("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
        self.reweight_info[self.current_iter][reweight_type]['accuracy'] = test_acc
        save_json(self.args, 'reweight_info.json', self.reweight_info)

    def train_with_sampling(self):
        from clustered_sampling.py_func.hyperparams import get_file_name
        from clustered_sampling.py_func.read_db import get_dataloaders
        from clustered_sampling.py_func.hyperparams import get_hyperparams
        assert 'cifar' in self.args.dataset 
        force = False

        file_name = get_file_name(
            'CIFAR10_nbal_{}'.format(self.args.alpha), self.args.sampling, self.args.sim_type, self.args.seed, self.args.n_SGD, self.args.lr_sampling, self.args.decay, self.args.n_sampled/self.args.num_users, 0
        )
        n_iter, batch_size, meas_perf_period = get_hyperparams('CIFAR', self.args.n_SGD)
        mu = 0
        list_dls_train, list_dls_test = get_dataloaders(self.train_dataset, self.test_dataset, self.user_groups, batch_size)
        model_0 = copy.deepcopy(self.init_model)

        """FEDAVG with random sampling"""
        if self.args.sampling == "random" and (
            not os.path.exists(f"clustered_sampling/saved_exp_info/acc/{file_name}.pkl")
        ):

            from clustered_sampling.py_func.FedProx import FedProx_sampling_random

            FedProx_sampling_random(
                model_0,
                self.args.n_sampled,
                list_dls_train,
                list_dls_test,
                n_iter,
                self.args.n_SGD,
                self.args.lr_sampling,
                file_name,
                self.args.decay,
                meas_perf_period,
                mu,
                self.device,
            )


        """Run FEDAVG with clustered sampling"""
        if (self.args.sampling == "clustered_1" or self.args.sampling == "clustered_2" or self.args.sampling == "clustered_SPACE") and (
            not os.path.exists(f"clustered_sampling/saved_exp_info/acc/{file_name}.pkl") or force
        ):

            from clustered_sampling.py_func.FedProx import FedProx_clustered_sampling

            if self.args.sampling == 'clustered_SPACE':
                num_class = len(set([d[1] for d in self.test_dataset]))
                serverPrototype = self.computeServerPrototype(num_class=num_class)
                clientPrototypes = []
                for idx in range(self.args.num_users):
                    local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                            idxs=self.user_groups[idx], logger=self.writer)
                    prototype, local_labels_list = local_model.computeLocalPrototype(model=self.model, num_class=num_class)
                    clientPrototypes.append(prototype)
            else:
                clientPrototypes = None
            clientContributions = copy.deepcopy(self.contributions)
            clientContributions[clientContributions < 0] = 0
            clientContributions = clientContributions / clientContributions.sum()
            beta = 0 if self.args.alpha == 10.0 else 0.5

            FedProx_clustered_sampling(
                self.args.sampling,
                model_0,
                self.args.n_sampled,
                list_dls_train,
                list_dls_test,
                n_iter,
                self.args.n_SGD,
                self.args.lr_sampling,
                file_name,
                self.args.sim_type,
                0,
                self.args.decay,
                meas_perf_period,
                mu,
                self.device,
                prototypes=clientPrototypes,
                contributions=clientContributions,
                beta=beta,
            )


        """RUN FEDAVG with perfect sampling for MNIST-shard"""
        if (
            self.args.sampling == "perfect"
            and dataset == "MNIST_shard"
            and (not os.path.exists(f"clustered_sampling/saved_exp_info/acc/{file_name}.pkl") or force)
        ):

            from clustered_sampling.py_func.FedProx import FedProx_sampling_target

            FedProx_sampling_target(
                model_0,
                self.args.n_sampled,
                list_dls_train,
                list_dls_test,
                n_iter,
                self.args.n_SGD,
                self.args.lr_sampling,
                file_name,
                self.args.decay,
                mu,
                self.device,
            )


        """RUN FEDAVG with its original sampling scheme sampling clients uniformly"""
        if self.args.sampling == "FedAvg" and (
            not os.path.exists(f"clustered_sampling/saved_exp_info/acc/{file_name}.pkl") or force
        ):

            from clustered_sampling.py_func.FedProx import FedProx_FedAvg_sampling

            FedProx_FedAvg_sampling(
                model_0,
                self.args.n_sampled,
                list_dls_train,
                list_dls_test,
                n_iter,
                self.args.n_SGD,
                self.args.lr_sampling,
                file_name,
                self.args.decay,
                meas_perf_period,
                mu,
                self.device,
            )


