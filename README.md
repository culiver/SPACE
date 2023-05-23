# SPACE: Single-round Participant Amalgamation for Contribution Evaluation in Federated Learning

## Requirments
Install all the packages from requirments.txt

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist and Cifar.

## Quick Start
We provide shell scripts to run all the approaches mentioned in our paper, except for GTG-Shapley. For the implementation of GTG-Shapley, please refer to https://github.com/liuzelei13/GTG-Shapley.

Before running the commands, navigate to the source code directory src/:
```
cd src
```

### Contribution Evaluation
To calculate the actual Shapley value, run:
```
./RealShapley_Contribution_dist3.sh
```
For SPACE, run:
```
./KA_Contribution_dist3.sh
```
For SPACE(Avg), run:
```
./FedAvg_Contribution_dist3.sh
```
For DIG-FL, run:
```
./DIGFL_Contribution_dist3.sh
```
For TMC-Shapley, run:
```
./TMC_Contribution_dist3.sh
```
For Group Testing, run:
```
./GT_Contribution_dist3.sh
```
Note that the shell script runs all the local training sequentially. To run local training in parallel, add --RPC to the command.

To evaluate the Pearson Correlation Coefficient (PCC) between the estimated Shapley value and the actual Shapley value, run:
```
python3 plot_PCC_calculator.py --m1 gt+tmc+digfl+fedavg+ka --dataset mnist --scenario noniid
```
The result will be saved in save/Contribution_results/

### Client Reweighting
For SPACE, run:
```
./KA_Contribution_dist3.sh
```
For SPACE (Average), run:
```
./FedAvg_Contribution_dist3.sh
```
For DIG-FL, run:
```
./DIGFL_Contribution_dist3.sh
```
To plot the reweighting result, run:
```
python3 plot_reweighting.py --m1 digfl+fedavg+KA
```
The result will be saved in save/Reweighting_results/
### Client Selection
For the client selection task, run:
```
./KA_Selection.sh
```
Note that the default implementation amalgamates 100 clients simultaneously. If there is limited GPU memory capacity, you can change the --t_num parameter to adjust the number of teachers in knowledge amalgamation.

To plot the result, run:
```
python3 plots_paper.py
```
The results will be saved in src/clustered_sampling/plots/

