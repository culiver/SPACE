# T = 1, K = 10
# mnist non-iid
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 8 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 6 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 4 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 2 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 1.0
# mnist mislabel
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 8 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 6 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 4 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 2 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 1.0
# cifar non-iid
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 4 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 3 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 2 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 1 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 1.0
# cifar mislabel
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 4 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 3 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 2 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 1 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 1.0

# T = 1, K = 100
# mnist non-iid
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 8 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 6 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 4 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 2 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 1.0
# mnist mislabel
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 8 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 6 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 4 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 2 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 1.0
# cifar non-iid
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 4 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 3 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 2 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 1 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 1.0
# cifar mislabel
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 4 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 3 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 2 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 1 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 1.0

# T = 1, K = inf
# mnist non-iid
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 8 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 6 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 4 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 2 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 1.0
# mnist mislabel
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 8 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 6 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 4 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 1.0
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 2 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 1.0
# cifar non-iid
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 4 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 3 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 2 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 1 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 1.0
# cifar mislabel
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 4 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 3 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 2 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 1.0
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 1 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 1.0

# T = user-define, K = 10
# mnist non-iid
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 8 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 6 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 4 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 2 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 0.95
# mnist mislabel
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 8 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 6 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 4 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 2 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T 0.95
# cifar non-iid
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 4 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 3 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 2 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 1 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 0.5
# cifar mislabel
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 4 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 3 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 2 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 1 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T 0.5

# T = user-define, K = 100
# mnist non-iid
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 8 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 6 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 4 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 2 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 0.95
# mnist mislabel
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 8 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 6 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 4 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 2 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T 0.95
# cifar non-iid
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 4 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 3 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 2 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 1 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 0.5
# cifar mislabel
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 4 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 3 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 2 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 1 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T 0.5

# T = user-define, K = inf
# mnist non-iid
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 8 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 6 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 4 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 2 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 0.95
# mnist mislabel
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 8 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 6 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 4 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 0.95
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 2 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T 0.95
# cifar non-iid
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 4 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 3 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 2 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 1 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 0.5
# cifar mislabel
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 4 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 3 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 2 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 0.5
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 1 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T 0.5

# T = nan, K = 10
# mnist non-iid
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 8 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 6 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 4 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 2 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T nan
# mnist mislabel
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 8 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 6 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 4 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 2 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 10 --T nan
# cifar non-iid
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 4 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 3 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 2 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 1 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T nan
# cifar mislabel
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 4 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 3 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 2 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 1 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 10 --T nan

# T = nan, K = 100
# mnist non-iid
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 8 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 6 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 4 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 2 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T nan
# mnist mislabel
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 8 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 6 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 4 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 2 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k 100 --T nan
# cifar non-iid
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 4 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 3 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 2 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 1 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T nan
# cifar mislabel
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 4 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 3 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 2 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 1 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k 100 --T nan

# T = nan, K = inf
# mnist non-iid
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 8 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 6 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 4 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 2 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 0 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T nan
# mnist mislabel
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 8 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 6 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 4 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T nan
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --dist=3 --num_users 10 --num_Noniid 0 --num_Mislabel 2 --local_ep 1 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 1 --skip_train --k inf --T nan
# cifar non-iid
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 4 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 3 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 2 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 1 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 0 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T nan
# cifar mislabel
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 4 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 3 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 2 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T nan
python federated_main.py --model=cnn --dataset=cifar --gpu=0 --dist=3 --num_users 5 --num_Noniid 0 --num_Mislabel 1 --local_ep 2 --epochs 20 --verbose 0 --trainer ka5 --ka_ep=10 --local_ep_ka 2 --skip_train --k inf --T nan