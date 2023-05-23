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
   

def main(args):
    os.makedirs(os.path.join(args.root, 'Contribution_results'), exist_ok=True)
    k_list = [100.0]
    if args.impact == 'k':
        T_list = ['user']
        k_list = [10.0, 100.0, 'inf']
    else:
        T_list = [1.0, 'nan', 'user']
        k_list = [100.0]
    dataset_list = ['mnist', 'cifar']
    # dataset_list = ['mnist']
    # dataset_list = ['cifar']
    scenario_list = ['noniid', 'mislabel']
    # scenario_list = ['noniid']
    # scenario_list = ['mislabel']
    
    for (dataset, scenario) in itertools.product(dataset_list, scenario_list):
        print("="*5 + str((dataset, scenario)) + "="*5)
        for T_idx, T in enumerate(T_list):
            for k_idx, k in enumerate(k_list):    
                list2 = []

                if T == 'user':
                    T_temp = 0.95 if dataset == 'mnist' else 0.5
                else:
                    T_temp = T

                c2_list = sorted(glob.glob(os.path.join(args.root, 'REALSHAPLEY*', 'contributions_*.json')))
                c2_list = [c for c in c2_list if checkString(c, ['k[{}]'.format(k), 'T[{}]'.format(T_temp)])]

                if scenario == 'noniid':
                    # keyword = 'Mislabel[0]' 
                    if dataset == 'mnist':
                        keyword = 'Noniid[8]'
                    else:
                        keyword = 'Noniid[4]' 
                elif scenario == 'mislabel':
                    # keyword = 'Noniid[0]'
                    if dataset == 'mnist':
                        keyword = 'Mislabel[8]'
                    else:
                        keyword = 'Mislabel[4]'
                else:
                    keyword = ''

                dataset_keyword = dataset

                c2_list = [c for c in c2_list if checkString(c, [dataset_keyword, keyword, '_iid[3]', 'iternum[1]'])]

                for file2_path in c2_list:
                    with open(file2_path, 'r') as f:
                        temp = np.array(json.load(f))
                        temp[temp<0] = 0
                        temp = (temp / temp.sum()).tolist() if temp.sum() > 0 else temp.tolist()
                        contribution = temp
                

                color_pool = ['red','blue','green','orange','purple','brown','pink','gray','cyan','magenta']
                T_temp = '0.9669' if T_temp == 'nan' else T_temp
                plt.scatter(np.arange(len(contribution)), contribution, c=color_pool[T_idx*len(k_list)+k_idx], label='k[{}]_T[{}]'.format(k, T_temp), marker='x')

        plt.legend()
        dataset_name = 'CIFAR10' if dataset == 'cifar' else 'MNIST'
        scenario_name = 'Mislabel' if scenario=='mislabel' else 'Non-IID'
        plt.xticks(range(len(contribution)))
        # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.title('{} {}'.format(dataset_name, scenario_name))
        plt.ylabel('Shapley value')
        plt.xlabel('Client index')
        plt.savefig(os.path.join(args.root, 'Contribution_results', 'Logistic_Impact_{}_{}_{}.png'.format(args.impact, dataset, scenario)))
        plt.close()

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Calculate Pearson correlation coefficient from two JSON files')
    parser.add_argument('--root', default='../save', type=str, help='Path to testing results')
    parser.add_argument('--dataset', default='mnist', type=str, help='Path to testing results')
    parser.add_argument('--scenario', choices=['noniid', 'mislabel', 'all'], default='noniid', type=str, help='Path to testing results')
    # parser.add_argument('--k', default='10.0', type=float, help='k')
    # parser.add_argument('--T', default='1.0', type=float, help='T')
    parser.add_argument('--impact', default='k', type=str, help='k')

    args = parser.parse_args()

    main(args)