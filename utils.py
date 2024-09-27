import numpy as np
import time
import GPUtil
import torch
import os
import random
import math
import pickle

def none_or_str(value):
    if value == 'None':
        return None
    return value

def grab_gpu(memory_limit=0.91, max_wait_time=600):
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        cuda_device_ids = GPUtil.getAvailable(order='memory', limit=len(GPUtil.getGPUs()), maxLoad=1.0, maxMemory=memory_limit)

        if cuda_device_ids:
            return str(cuda_device_ids[0])
        else:
            print("Waiting for available GPU...")
            time.sleep(10)

    raise RuntimeError("No GPU available within the maximum wait time.")

def create_iid_shards(idxs, num_shards, num_samples, num_classes, seed=1):
    np.random.seed(seed)
    data_distribution = np.random.choice(a=np.arange(0,num_shards), size=num_samples).astype(int)
    return {id:list(np.squeeze(np.argwhere((np.squeeze([data_distribution==id])==True)))) for id in range(num_shards)}

def create_imbalanced_shards(idxs, num_shards, num_samples, num_classes, skewness=0.8, seed=1):
    np.random.seed(seed)
    data_distribution = np.random.choice(a=np.arange(0,num_shards), size=num_samples, p=np.random.dirichlet(np.repeat(skewness, num_shards))).astype(int)
    return {id:list(np.squeeze(np.argwhere((np.squeeze([data_distribution==id])==True)))) for id in range(num_shards)}

def create_noniid_shards(idxs, num_shards, num_samples, num_classes, skewness=0.5, seed=1,):
    np.random.seed(seed)
    partitions = {}
    min_size = 0
    min_require_size = 10
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_shards)]
        for k in range(num_classes):
            idx_k = np.where(idxs==k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(skewness, num_shards))
            proportions = np.array([p * (len(idx_j) < num_samples / num_shards) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_shards):
            np.random.shuffle(idx_batch[j])
            partitions[j] = idx_batch[j]
    return partitions

def dir_balance(dataset, dataset_name, num_classes, num_users, alpha, data_dir, sample=None):
    """ for the fairness of annotation cost, each client has same number of samples
    """
    C = num_classes
    K = num_users
    alpha = alpha
    
    # Generate the set of clients dataset.
    clients_data = {}
    for i in range(K):
        clients_data[i] = []

    # Divide the dataset into each class of dataset.
    total_num = len(dataset)
    total_data = {}
    data_num = np.array([0 for _ in range(C)])
    for i in range(C):
        total_data[str(i)] = []
    for idx, data in enumerate(dataset):
        if dataset_name in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
            total_data[str(data[1][0])].append(idx)
            data_num[int(data[1][0])] += 1
        else:
            total_data[str(data[1])].append(idx)
            data_num[int(data[1])] += 1

    clients_data_num = {}
    for client in range(K):
        clients_data_num[client] = [0] * C
    
    # Distribute the data with the Dirichilet distribution.
    if sample is None:
        diri_dis = torch.distributions.dirichlet.Dirichlet(alpha * torch.ones(C))
        sample = torch.cat([diri_dis.sample().unsqueeze(0) for _ in range(K)], 0)

        # get balanced matrix
        rsum = sample.sum(1)
        csum = sample.sum(0)
        epsilon = min(1 , K / C, C / K) / 1000

        if alpha < 10:
            r, c = 1, K / C
            while (torch.any(rsum <= r - epsilon)) or (torch.any(csum <= c - epsilon)):
                sample /= sample.sum(0)
                sample /= sample.sum(1).unsqueeze(1)
                rsum = sample.sum(1)
                csum = sample.sum(0)
        else:
            r, c = C / K, 1
            while (torch.any(abs(rsum - r) >= epsilon)) or (torch.any(abs(csum - c) >= epsilon)):
                sample = sample / sample.sum(1).unsqueeze(1)
                sample /= sample.sum(0)
                rsum = sample.sum(1)
                csum = sample.sum(0)
        
    x = sample * torch.tensor(data_num)
    x = torch.ceil(x).long()
    x = torch.where(x <= 1, 0, x+1) if alpha < 10 else torch.where(x <= 1, 0, x)
    # print(x)
    
    print('Dataset total num', len(dataset))
    print('Total dataset class num', data_num)

    if alpha < 10:
        remain = np.inf
        nums = math.ceil(len(dataset) / K)
        i = 0
        while remain != 0:
            i += 1
            for client_idx in clients_data.keys():
                for cls in total_data.keys():
                    tmp_set = random.sample(total_data[cls], min(len(total_data[cls]), x[client_idx, int(cls)].item()))
                    
                    if len(clients_data[client_idx]) + len(tmp_set) > nums:
                        tmp_set = tmp_set[:nums-len(clients_data[client_idx])]

                    clients_data[client_idx] += tmp_set
                    clients_data_num[client_idx][int(cls)] += len(tmp_set)

                    total_data[cls] = list(set(total_data[cls])-set(tmp_set))   

            remain = sum([len(d) for _, d in total_data.items()])
            if i == 100:
                break
                
        # to make same number of samples for each client
        index = np.where(np.array([sum(clients_data_num[k]) for k in clients_data_num.keys()]) <= nums-1)[0]
        for client_idx in index:
            n = nums - len(clients_data[client_idx])
            add = 0
            for cls in total_data.keys():
                tmp_set = total_data[cls][:n-add]
                
                clients_data[client_idx] += tmp_set
                clients_data_num[client_idx][int(cls)] += len(tmp_set)
                total_data[cls] = list(set(total_data[cls])-set(tmp_set))  
                
                add += len(tmp_set)
    else:
        cumsum = x.T.cumsum(dim=1)
        for cls, data in total_data.items():
            cum = list(cumsum[int(cls)].numpy())
            tmp = np.split(np.array(data), cum)

            for client_idx in clients_data.keys():
                clients_data[client_idx] += list(tmp[client_idx])
                clients_data_num[client_idx][int(cls)] += len(list(tmp[client_idx]))

    print('clients_data_num', clients_data_num)
    print('clients_data_num', [sum(clients_data_num[k]) for k in clients_data_num.keys()])
    with open(os.path.join(data_dir, 'clients_data_num.pickle'), 'wb') as f:
        pickle.dump(clients_data_num, f)

    return clients_data, sample

def get_split_fn(name='iid', **split_fn_kwargs):
    if name == 'iid':
        return create_iid_shards
    elif name == 'noniid':
        return create_noniid_shards
    elif name == 'imbalanced':
        return create_imbalanced_shards
    elif name == 'dir_balance':
        return ...
    else:
        raise ValueError("Invalid name provided. Supported names are 'iid', 'noniid', 'imbalanced', and 'dir_balance'.")

import math

def get_learning_rate(initial_lr, current_round, total_rounds, decay_factor=0.5, num_decays=3):
    """
    Step-wise decay of learning rate.
    """
    # Determine the number of decays that should have occurred by the current round
    decay_step = total_rounds / (num_decays + 1)  # +1 to include the initial LR at round 0
    num_applied_decays = int(current_round / decay_step)
    
    # Calculate the new learning rate
    new_lr = initial_lr * (decay_factor ** num_applied_decays)
    
    return new_lr