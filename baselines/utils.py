import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import yaml

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_dir_if_not_exists(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

class DatasetSplit(Dataset):
    """Dataset split class for federated learning."""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label, self.idxs[item]  # Return index as well

def dir_partition(dataset, num_clients, alpha):
    """Partition the dataset using Dirichlet distribution."""
    num_classes = len(np.unique(np.array(dataset.targets)))
    client_data_dict = {i: [] for i in range(num_clients)}
    
    # Group indices by label
    label_indices = {i: [] for i in range(num_classes)}
    for idx, label in enumerate(dataset.targets):
        if isinstance(label, torch.Tensor):
            label = label.item()
        label_indices[label].append(idx)
    
    # Sample client proportions using Dirichlet distribution
    proportions = np.random.dirichlet(alpha * np.ones(num_clients), num_classes)
    
    # Assign samples to clients according to proportions
    for class_idx, class_proportions in enumerate(proportions):
        class_size = len(label_indices[class_idx])
        
        # Get the number of samples per client for this class
        num_samples_per_client = np.round(class_proportions * class_size).astype(int)
        # Adjust to ensure the sum matches the class size
        num_samples_per_client[-1] = class_size - np.sum(num_samples_per_client[:-1])
        
        # Shuffle the indices for this class
        class_indices = label_indices[class_idx].copy()
        random.shuffle(class_indices)
        
        # Assign indices to clients
        start_idx = 0
        for client_idx, num_samples in enumerate(num_samples_per_client):
            client_data_dict[client_idx].extend(
                class_indices[start_idx:start_idx + int(num_samples)]
            )
            start_idx += int(num_samples)
    
    # Ensure each client has the same amount of data
    max_len = max([len(client_data_dict[i]) for i in range(num_clients)])
    for client_idx in range(num_clients):
        current_len = len(client_data_dict[client_idx])
        if current_len < max_len:
            # Randomly sample additional data
            additional_indices = random.sample(
                sum([label_indices[c] for c in range(num_classes)], []),
                max_len - current_len
            )
            client_data_dict[client_idx].extend(additional_indices)
    
    return client_data_dict

def shard_partition(dataset, num_clients, num_classes_per_user):
    """Partition the dataset using shards."""
    num_classes = len(np.unique(np.array(dataset.targets)))
    client_data_dict = {i: [] for i in range(num_clients)}
    
    # Group indices by label
    label_indices = {i: [] for i in range(num_classes)}
    for idx, label in enumerate(dataset.targets):
        if isinstance(label, torch.Tensor):
            label = label.item()
        label_indices[label].append(idx)
    
    # Shuffle the labels
    label_order = list(range(num_classes))
    random.shuffle(label_order)
    
    # Assign shards to clients
    shards_per_class = num_clients * num_classes_per_user // num_classes
    for class_idx in label_order:
        class_indices = label_indices[class_idx]
        random.shuffle(class_indices)
        
        # Split the indices into shards
        shards = np.array_split(class_indices, shards_per_class)
        
        # Assign shards to clients
        for i, shard in enumerate(shards):
            client_idx = (i * num_classes) % num_clients
            client_data_dict[client_idx].extend(shard.tolist())
    
    return client_data_dict

def adjust_learning_rate(optimizer, epoch, args):
    """Adjust learning rate during training."""
    lr = args.lr
    if epoch >= args.num_rounds * 0.5:
        lr = args.lr * 0.1
    if epoch >= args.num_rounds * 0.75:
        lr = args.lr * 0.01
    
    for param_group in optimizer.parameters():
        param_group['lr'] = lr
    
    return lr


# Active Learning Strategies
class RandomSampling:
    @staticmethod
    def query(user_idx, label_idxs, unlabel_idxs, dataset, net_global, net_local, args, query_budget=None):
        """Randomly sample from unlabeled data."""
        if query_budget is None:
            query_budget = args.query_budget_per_round
            
        unlabel_idxs_list = list(unlabel_idxs)
        return random.sample(unlabel_idxs_list, min(query_budget, len(unlabel_idxs_list)))

class EntropySampling:
    @staticmethod
    def query(user_idx, label_idxs, unlabel_idxs, dataset, net_global, net_local, args, query_budget=None):
        """Sample based on prediction entropy."""
        if query_budget is None:
            query_budget = args.query_budget_per_round
            
        # Convert unlabel_idxs to list if it's a set
        unlabel_idxs_list = list(unlabel_idxs)
        
        if len(unlabel_idxs_list) <= query_budget:
            return unlabel_idxs_list
            
        # Choose model
        if args.query_model_mode == "global":
            net = net_global
        elif args.query_model_mode == "local_only":
            net = net_local
        else:
            raise ValueError("Invalid query_model_mode")
        
        # Create dataloader for unlabeled data
        unlabel_dataset = DatasetSplit(dataset, unlabel_idxs_list)
        unlabel_loader = DataLoader(unlabel_dataset, batch_size=64, shuffle=False)
        
        net.eval()
        uncertainties = []
        indices = []
        
        # Calculate entropy for each unlabeled sample
        with torch.no_grad():
            for data, _, idx in unlabel_loader:
                data = data.to(args.device)
                outputs, _ = net(data)
                prob = F.softmax(outputs, dim=1)
                entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
                
                uncertainties.extend(entropy.cpu().numpy())
                indices.extend(idx.numpy())
        
        # Select samples with highest entropy
        paired = list(zip(indices, uncertainties))
        paired.sort(key=lambda x: x[1], reverse=True)
        
        return [p[0] for p in paired[:min(query_budget, len(paired))]]


class CoreSet:
    @staticmethod
    def furthest_first(X, X_set, n):
        """Core-set selection using greedy furthest-first traversal."""
        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []
        for i in range(n):
            if np.max(min_dist) == float("inf") or len(idxs) >= m:
                break
                
            idx = min_dist.argmax()
            idxs.append(idx)
            
            if i == n - 1:
                break
                
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    @staticmethod
    def query(user_idx, label_idxs, unlabel_idxs, dataset, net_global, net_local, args, query_budget=None):
        """Sample using Core-set approach."""
        if query_budget is None:
            query_budget = args.query_budget_per_round
            
        # Convert sets to lists
        unlabel_idxs_list = list(unlabel_idxs)
        label_idxs_list = list(label_idxs)
        
        if len(unlabel_idxs_list) <= query_budget:
            return unlabel_idxs_list
            
        # Choose model
        if args.query_model_mode == "global":
            net = net_global
        elif args.query_model_mode == "local_only":
            net = net_local
        else:
            raise ValueError("Invalid query_model_mode")
        
        try:
            # Create dataloaders
            data_idxs = unlabel_idxs_list + label_idxs_list
            dataset_split = DatasetSplit(dataset, data_idxs)
            loader = DataLoader(dataset_split, batch_size=64, shuffle=False)
            
            net.eval()
            embeddings = []
            indices = []
            
            # Get embeddings for all samples
            with torch.no_grad():
                for data, _, idx in loader:
                    data = data.to(args.device)
                    _, embedding = net(data)
                    embeddings.append(embedding.cpu().numpy())
                    indices.append(idx.numpy())
            
            embeddings = np.vstack(embeddings)
            indices = np.concatenate(indices)
            
            # Map original indices to positions in the embeddings array
            idx_map = {idx: pos for pos, idx in enumerate(indices)}
            
            # Get embeddings for unlabeled and labeled data
            unlabel_embeddings = embeddings[:len(unlabel_idxs_list)]
            label_embeddings = embeddings[len(unlabel_idxs_list):]
            
            # Select samples using furthest-first
            chosen = CoreSet.furthest_first(
                unlabel_embeddings, 
                label_embeddings, 
                min(query_budget, len(unlabel_idxs_list))
            )
            
            return [unlabel_idxs_list[i] for i in chosen]
            
        except Exception as e:
            print(f"Error in CoreSet query: {e}")
            # Fallback to random sampling
            return random.sample(unlabel_idxs_list, min(query_budget, len(unlabel_idxs_list)))

class LoGo:
    @staticmethod
    def query(user_idx, label_idxs, unlabel_idxs, dataset, net_global, net_local, args, query_budget=None):
        """Sample using LoGo algorithm."""
        if query_budget is None:
            query_budget = args.query_budget_per_round
            
        # Convert unlabel_idxs and label_idxs to lists if they're sets
        unlabel_idxs_list = list(unlabel_idxs)
        label_idxs_list = list(label_idxs)
        
        if len(unlabel_idxs_list) <= query_budget:
            return unlabel_idxs_list
        
        # Ensure we have both models
        if net_global is None or net_local is None:
            raise ValueError("LoGo requires both global and local models")
        
        # Create dataloader for unlabeled data
        unlabel_dataset = DatasetSplit(dataset, unlabel_idxs_list)
        unlabel_loader = DataLoader(unlabel_dataset, batch_size=64, shuffle=False)
        
        # Macro Step: Clustering with Local-Only Model
        net_local.eval()
        
        # Calculate gradient embeddings
        grad_embeddings = []
        indices = []
        
        try:
            with torch.no_grad():
                for data, _, idx in unlabel_loader:
                    data = data.to(args.device)
                    outputs, embedding = net_local(data)
                    
                    # Get predicted labels
                    _, pred_labels = torch.max(outputs, 1)
                    
                    # Calculate the gradient embedding for each sample
                    for i in range(len(data)):
                        # Get the predicted class for the sample
                        pred_class = pred_labels[i].item()
                        emb = embedding[i]
                        
                        # Calculate the gradient of loss with respect to the weight of the predicted class
                        # This is a simplified version of the gradient embedding
                        prob = F.softmax(outputs[i:i+1], dim=1)[0]
                        grad = emb * (1 - prob[pred_class])
                        
                        grad_embeddings.append(grad.cpu().numpy())
                        indices.append(idx[i].item())
            
            grad_embeddings = np.array(grad_embeddings)
            
            # Make sure we have enough samples for K-means
            n_clusters = min(query_budget, len(unlabel_idxs_list))
            if len(grad_embeddings) < n_clusters:
                print(f"Warning: Not enough samples for K-means (have {len(grad_embeddings)}, need {n_clusters})")
                # Fall back to random sampling
                return random.sample(unlabel_idxs_list, min(query_budget, len(unlabel_idxs_list)))
            
            # Perform K-Means clustering on gradient embeddings
            kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed).fit(grad_embeddings)
            
            # Create clusters
            clusters = {i: [] for i in range(n_clusters)}
            for i, cluster_idx in enumerate(kmeans.labels_):
                clusters[cluster_idx].append(indices[i])
            
            # Micro Step: Cluster-wise Sampling with Global Model
            net_global.eval()
            selected_samples = []
            
            for cluster_idx, cluster_samples in clusters.items():
                if not cluster_samples:
                    continue
                    
                # Create dataloader for cluster samples
                cluster_dataset = DatasetSplit(dataset, cluster_samples)
                cluster_loader = DataLoader(cluster_dataset, batch_size=64, shuffle=False)
                
                # Calculate entropy for each sample in the cluster
                uncertainties = []
                cluster_indices = []
                
                with torch.no_grad():
                    for data, _, idx in cluster_loader:
                        data = data.to(args.device)
                        outputs, _ = net_global(data)
                        prob = F.softmax(outputs, dim=1)
                        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
                        
                        uncertainties.extend(entropy.cpu().numpy())
                        cluster_indices.extend(idx.numpy())
                
                # Select the most uncertain sample from the cluster
                if cluster_indices:
                    most_uncertain_idx = cluster_indices[np.argmax(uncertainties)]
                    selected_samples.append(most_uncertain_idx)
            
            return selected_samples
            
        except Exception as e:
            print(f"Error in LoGo query: {e}")
            # Fall back to random sampling
            return random.sample(unlabel_idxs_list, min(query_budget, len(unlabel_idxs_list)))

# Map of active learning strategies
AL_STRATEGIES = {
    'random': RandomSampling,
    'entropy': EntropySampling,
    'coreset': CoreSet,
    'logo': LoGo
}

def get_al_strategy(strategy_name):
    """Get active learning strategy by name."""
    if strategy_name.lower() in AL_STRATEGIES:
        return AL_STRATEGIES[strategy_name.lower()]
    else:
        raise ValueError(f"Unknown active learning strategy: {strategy_name}")

# Implementation of CNN4Conv model
class CNN4Conv(nn.Module):
    def __init__(self, in_channels, num_classes, img_size=32):
        super(CNN4Conv, self).__init__()
        hidden_size = 64
        
        if img_size == 32:
            self.emb_dim = hidden_size * 2 * 2
        elif img_size == 28:
            self.emb_dim = hidden_size
        else:
            raise ValueError(f"Unsupported image size: {img_size}")
            
        self.features = nn.Sequential(
            self.conv3x3(in_channels, hidden_size),
            self.conv3x3(hidden_size, hidden_size),
            self.conv3x3(hidden_size, hidden_size),
            self.conv3x3(hidden_size, hidden_size)
        )

        self.linear = nn.Linear(self.emb_dim, num_classes)
        self.linear.bias.data.fill_(0)

    def conv3x3(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
            nn.BatchNorm2d(out_channels, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        features = self.features(x)
        features = features.view((features.size(0), -1))
        logits = self.linear(features)
        
        return logits, features
    
    def get_embedding_dim(self):
        return self.emb_dim
