import argparse
import json
import numpy as np
from scipy.stats import pearsonr
import glob
import os
import matplotlib.pyplot as plt
import itertools

def checkString(s, keyword_list):
    result = True
    for k in keyword_list:
        result = result and (k in s)
    return result

def plot_acc_curve(args, acc_info, savename, dataset, scenario, axis, data_scenario_idx):
    color_list = ['r', 'g', 'b', 'orange']

    # plt.figure()
    x = list(range(0, 100, 20))


    for idx, label in enumerate(acc_info):
        #(5 * num_task, 5)
        acc_list = np.array(acc_info[label])
        if label == 'normal':
            num_task = len(args.m1.split('+'))
            acc_list = acc_list.reshape(num_task, 5, 5)
            acc_list = np.transpose(acc_list, (0, 2, 1))
            acc_list = acc_list.reshape(5*num_task, 5)
            means = np.mean(acc_list, axis=0) * 100
            normal_means = np.mean(acc_list, axis=0) * 100
        else:
            acc_list = np.transpose(acc_list, (1, 0))
            means = np.mean(acc_list, axis=0) * 100
            print(label, dataset, scenario, max(means-normal_means))
        stds = np.std(acc_list, axis=0) * 100

        axis[data_scenario_idx].plot(x, means, label=label, color=color_list[idx])
        axis[data_scenario_idx].fill_between(x=x, y1=means-stds, y2=means+stds, alpha=0.1, color=color_list[idx])
    

    dataset_name = 'CIFAR10' if dataset == 'cifar' else 'MNIST'
    scenario_name = 'Mislabel' if scenario=='mislabel' else 'Non-IID'

    axis[data_scenario_idx].set_title('{} {}'.format(dataset_name, scenario_name))
    axis[data_scenario_idx].set_xlabel('Percentage of {} clients (%)'.format(scenario_name))

    if data_scenario_idx == 0:
        axis[data_scenario_idx].legend()
        axis[data_scenario_idx].set_ylabel('Average Accuracy (%)')
    if data_scenario_idx == 3:
        plt.tight_layout(pad=0.5)
        plt.savefig(savename)


def main(args):
    os.makedirs(os.path.join(args.root, 'Reweighting_results'), exist_ok=True)
    dataset_list = ['mnist', 'cifar']
    scenario_list = ['noniid', 'mislabel']
    fig, axis = plt.subplots(1, 4, figsize=(16,4))
    name_dict = {'KA':'SPACE(static)', 'FEDAVG':'SPACE(dynamic)', 'DIGFL':'DIG-FL'}

    method_list = args.m1.split('+')
    for (dataset, scenario) in itertools.product(dataset_list, scenario_list):
        acc_info = {}
        for method in method_list:
            method = method.upper()
            print(method,dataset,scenario)
            c1_list = sorted(glob.glob(os.path.join(args.root, method+'*', 'reweight_info.json')))

            if scenario == 'noniid':
                keyword = 'Mislabel[0]' 
            elif scenario == 'mislabel':
                keyword = 'Noniid[0]'
            else:
                keyword = ''

            dataset_keyword = dataset

            c1_list = [c for c in c1_list if checkString(c, [dataset_keyword, keyword, '_iid[3]', 'iternum[5]'])]

            for file1_path in c1_list:
                with open(file1_path, 'r') as f:
                    reweight_info = json.load(f)
                    if 'accuracy' not in reweight_info[0]['normal']:
                        continue
                    acc_info.setdefault('normal', []).append([iteration['normal']['accuracy'] for iteration in reweight_info])
                    if method == 'KA':
                        acc_info.setdefault(name_dict[method], []).append([iteration['static_reweight']['accuracy'] for iteration in reweight_info])
                    else:
                        acc_info.setdefault(name_dict[method], []).append([iteration['dynamic_reweight']['accuracy'] for iteration in reweight_info])

        savename = os.path.join(args.root, 'Reweighting_results', 'Reweighting_all.png')
        plot_acc_curve(args, acc_info, savename, dataset, scenario, axis, data_scenario_idx=2*(dataset=='cifar') + 1*(scenario=='mislabel'))

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Calculate Pearson correlation coefficient from two JSON files')
    parser.add_argument('--m1', default='fedavg', type=str, help='Path to the first JSON file')
    parser.add_argument('--root', default='../save', type=str, help='Path to testing results')
    parser.add_argument('--dataset', default='mnist', type=str, help='Path to testing results')
    parser.add_argument('--scenario', choices=['noniid', 'mislabel', 'all'], default='noniid', type=str, help='Path to testing results')
    parser.add_argument('--num_users', default=10, type=int, help='Path to testing results')
    parser.add_argument('--rm_step', default=2, type=int, help='Path to testing results')
    args = parser.parse_args()

    main(args)