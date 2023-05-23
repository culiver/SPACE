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
    T_list = ['user']
    k_list = [100.0]
    method_list = args.m1.split('+')
    dataset_list = ['mnist', 'cifar']
    scenario_list = ['noniid', 'mislabel']
    fig, axis = plt.subplots(len(method_list), 4, figsize=(15,15))
    
    for T in T_list:
        for k in k_list:
            print("="*5 + str((T, k)) + "="*5)
            for method_idx, method in enumerate(method_list):
                for (dataset, scenario) in itertools.product(dataset_list, scenario_list):
                    list1 = []
                    list2 = []

                    method = method.upper()
                    if T == 'user':
                        T_temp = 0.95 if dataset == 'mnist' else 0.5
                    else:
                        T_temp = T

                    if 'DIGFL' in method:
                        c1_list = sorted(glob.glob(os.path.join(args.root, method+'*', 'contributions.json')))
                    else:
                        c1_list = sorted(glob.glob(os.path.join(args.root, method+'_*', 'contributions_*.json')))
                        c1_list = [c for c in c1_list if checkString(c, ['k[{}]'.format(k), 'T[{}]'.format(T_temp)])]
                    c2_list = sorted(glob.glob(os.path.join(args.root, 'REALSHAPLEY*', 'contributions_*.json')))
                    c2_list = [c for c in c2_list if checkString(c, ['k[{}]'.format(k), 'T[{}]'.format(T_temp)])]

                    if scenario == 'noniid':
                        keyword = 'Mislabel[0]' 
                    elif scenario == 'mislabel':
                        keyword = 'Noniid[0]'
                    else:
                        keyword = ''

                    dataset_keyword = dataset

                    c1_list = [c for c in c1_list if checkString(c, [dataset_keyword, keyword, '_iid[3]', 'iternum[1]'])]
                    c2_list = [c for c in c2_list if checkString(c, [dataset_keyword, keyword, '_iid[3]', 'iternum[1]'])]

                    assert len(c1_list) == len(c2_list), (T, k, method, dataset, scenario, len(c1_list), len(c2_list))

                    abnormal_list = []
                    for file1_path in c1_list:
                        with open(file1_path, 'r') as f:
                            temp = np.array(json.load(f))
                            temp[temp<0] = 0
                            temp = (temp / temp.sum()).tolist() if temp.sum() > 0 else temp.tolist()
                            list1 += temp
                        
                        abnormal_num = max(int(os.path.basename(os.path.dirname(file1_path)).split('_')[7].split('[')[1].split(']')[0]), int(os.path.basename(os.path.dirname(file1_path)).split('_')[8].split('[')[1].split(']')[0]))
                        for i in range(len(temp)):
                            if len(temp) - i <= abnormal_num:
                                abnormal_list.append(True)
                            else:
                                abnormal_list.append(False)
                            

                    for file2_path in c2_list:
                        with open(file2_path, 'r') as f:
                            temp = np.array(json.load(f))
                            temp[temp<0] = 0
                            temp = (temp / temp.sum()).tolist() if temp.sum() > 0 else temp.tolist()
                            list2 += temp

                    array1 = np.array(list1)
                    array2 = np.array(list2)
                    abnormal_list = np.array(abnormal_list).astype(bool)

                    corr, _ = pearsonr(array1, array2)    
                    print(f"Method: {method}, Dataset: {dataset}, Scenario: {scenario}")
                    print(f"Pearson correlation coefficient: {corr:.4f}")

                    row_idx = 2*(dataset=='cifar') + 1*(scenario=='mislabel')
                    axis[method_idx, row_idx].scatter(array1[abnormal_list], array2[abnormal_list], c='red', label=scenario)
                    axis[method_idx, row_idx].scatter(array1[~abnormal_list], array2[~abnormal_list], c='blue', label='normal')

                    # Set the axis labels and title
                    if row_idx == 0:
                        if method == 'GP':
                            axis[method_idx, row_idx].set_ylabel("GT\nActual Shapley value")
                        elif method == 'FEDAVG':
                            axis[method_idx, row_idx].set_ylabel("SPACE(Avg)\nActual Shapley value")
                        elif method == 'KA':
                            axis[method_idx, row_idx].set_ylabel("SPACE\nActual Shapley value")
                        else:
                            axis[method_idx, row_idx].set_ylabel("{}\nActual Shapley value".format(method))
                    if method_idx == 0:
                        if dataset == 'cifar':
                            if scenario == 'noniid':
                                axis[method_idx, row_idx].set_title("CIFAR10 Non-IID")
                            elif scenario == 'mislabel':
                                axis[method_idx, row_idx].set_title("CIFAR10 Mislabel")
                        else:
                            if scenario == 'noniid':
                                axis[method_idx, row_idx].set_title("MNIST Non-IID")
                            elif scenario == 'mislabel':
                                axis[method_idx, row_idx].set_title("MNIST Mislabel")
                    if method_idx == len(method_list)-1:
                        axis[method_idx, row_idx].set_xlabel('Predicted Shapley value')
                        axis[method_idx, row_idx].legend(loc=4)
                    fig.tight_layout()

    plt.savefig(os.path.join(args.root, 'Contribution_results', 'Contribution_All_r.png'))

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Calculate Pearson correlation coefficient from two JSON files')
    parser.add_argument('--m1', default='fedavg', type=str, help='Path to the first JSON file')
    parser.add_argument('--root', default='../save', type=str, help='Path to testing results')
    parser.add_argument('--dataset', default='mnist', type=str, help='Path to testing results')
    parser.add_argument('--scenario', choices=['noniid', 'mislabel', 'all'], default='noniid', type=str, help='Path to testing results')
    parser.add_argument('--k', default='10.0', type=float, help='k')
    parser.add_argument('--T', default='1.0', type=float, help='T')

    args = parser.parse_args()

    main(args)