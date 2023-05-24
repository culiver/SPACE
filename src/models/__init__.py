import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P

class Model(nn.Module):
    def __init__(self, args, ckp=None):
        super(Model, self).__init__()
        print('Making model...')
        self.args = args
        self.cpu = args.cpu
        if self.cpu:
            self.device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module('models.{}_{}'.format(args.model.lower(), args.dataset.lower()))
        self.model = module.make_model(args).to(self.device)


    def forward(self, x, is_feat=False, preact=False):
        if self.training:
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x, is_feat, preact)
        else:
            return self.model(x, is_feat, preact)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {'map_location': self.device}

        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,
                    **kwargs
                )
            elif pre_train:
                print('Load the model from {}'.format(pre_train))
                load_from = torch.load(pre_train, **kwargs)
                
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if load_from:

            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    # Get the weight and bias tensors from the pretrained weight and the current module
                    weight = load_from[name + '.weight']
                    bias = load_from[name + '.bias']
                    
                    # Check if the number of channels in the weight tensor needs to be modified
                    if weight.shape != module.weight.shape:
                        # Calculate the number of extra channels in the weight tensor and the module
                        extra_channels = weight.shape[1] - module.weight.shape[1]
                        
                        # Slice the weight tensor to remove the extra channels
                        weight = weight[:module.weight.shape[0], :module.weight.shape[1], :, :]
                        
                        # Slice the bias tensor to remove the extra elements
                        bias = bias[:module.bias.shape[0]]
                        
                        # Update the weight and bias tensors in the pretrained weight
                        load_from[name + '.weight'] = weight
                        load_from[name + '.bias'] = bias
                    
                    # Load the modified weight and bias tensors into the module
                    module.weight.data.copy_(weight)
                    module.bias.data.copy_(bias)

            msg = self.model.load_state_dict(load_from, strict=False)
            print(f"Load model: {msg}")

