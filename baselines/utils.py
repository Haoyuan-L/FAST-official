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

def dir_partition(dataset, num_clients, alpha=1.0, seed=0, dir=False):
    """
    Partition the dataset using Dirichlet distribution.
    
    Args:
        dataset: The dataset to partition
        num_clients: Number of clients
        alpha: Concentration parameter for Dirichlet distribution (lower means more heterogeneity)
        equal_samples: If True, ensure each client gets approximately the same number of samples
        
    Returns:
        Dictionary mapping client IDs to list of data indices
    """
    if dir:
        return noniid_partition(dataset, num_clients, alpha)
    else:
        return iid_partition(dataset, num_clients, seed)

def iid_partition(dataset, num_clients, seed):
    """
    Creates an IID partition of the dataset across num_clients.
    """
    # Determine total number of samples
    total_num = len(dataset)

    # Reproducible randomness
    np.random.seed(seed)

    # Randomly assign each sample index to a client
    assignments = np.random.choice(num_clients, size=total_num)

    # Build the partition dict
    client_data_dict = {i: [] for i in range(num_clients)}
    for idx, client_id in enumerate(assignments):
        client_data_dict[int(client_id)].append(idx)

    return client_data_dict


def noniid_partition(dataset, num_clients, alpha):
    # Number of classes and total samples
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    else:
        # fallback if dataset.labels used
        targets = dataset.labels
    # ensure targets is a list or array
    labels_arr = np.array([t.item() if isinstance(t, torch.Tensor) else t for t in targets])
    num_classes = len(np.unique(labels_arr))
    total_num = len(labels_arr)

    # Group indices by label
    label_indices = {i: [] for i in range(num_classes)}
    for idx, label in enumerate(labels_arr):
        label_indices[int(label)].append(idx)

    # Per-client batch containers and size budget
    client_batches = [[] for _ in range(num_clients)]
    per_client_budget = math.ceil(total_num / num_clients)
    min_require_size = 10
    min_size = 0

    # Repeat splitting until every client has at least min_require_size samples
    while min_size < min_require_size:
        # reset batches
        client_batches = [[] for _ in range(num_clients)]
        for cls in range(num_classes):
            idx_list = label_indices[cls].copy()
            random.shuffle(idx_list)

            # draw Dirichlet proportions for this class across clients
            props = np.random.dirichlet(alpha * np.ones(num_clients))
            # zero out proportions for any client already full
            props = np.array([p * (len(client_batches[i]) < per_client_budget)
                              for i, p in enumerate(props)])
            props = props / props.sum()

            # determine cut points and split indices
            cuts = (np.cumsum(props) * len(idx_list)).astype(int)[:-1]
            splits = np.split(np.array(idx_list), cuts)

            # assign splits to clients
            for i in range(num_clients):
                client_batches[i].extend(splits[i].tolist())

        # check the smallest batch size
        min_size = min(len(batch) for batch in client_batches)
        # if too small, retry the random draw

    # build and return dict mapping client -> indices
    client_data_dict = {i: client_batches[i] for i in range(num_clients)}
    return client_data_dict

def dir_partition_balanced(dataset, num_clients, alpha):
    num_classes = len(np.unique(np.array(dataset.targets)))
    client_data_dict = {i: [] for i in range(num_clients)}
    
    # Calculate target samples per client
    total_samples = len(dataset)
    target_samples_per_client = total_samples // num_clients
    
    print(f"Total dataset size: {total_samples}")
    print(f"Target samples per client: {target_samples_per_client}")
    
    # Group indices by label
    label_indices = {i: [] for i in range(num_classes)}
    for idx, label in enumerate(dataset.targets):
        if isinstance(label, torch.Tensor):
            label = label.item()
        label_indices[label].append(idx)
    
    # Print class distribution
    for c in range(num_classes):
        print(f"Class {c}: {len(label_indices[c])} samples")
    
    # Sample client proportions using Dirichlet distribution
    proportions = np.random.dirichlet(alpha * np.ones(num_clients), num_classes)
    
    # First pass: distribute according to Dirichlet proportions
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
            num_samples = int(num_samples)
            if start_idx + num_samples <= len(class_indices):
                client_data_dict[client_idx].extend(
                    class_indices[start_idx:start_idx + num_samples]
                )
                start_idx += num_samples
    
    # Second pass: ensure each client has exactly target_samples_per_client samples
    # First, check how many samples each client has
    client_sample_counts = {i: len(samples) for i, samples in client_data_dict.items()}
    
    # For clients with too few samples, randomly sample from other clients or remaining data
    remaining_indices = []
    for c in range(num_classes):
        remaining_indices.extend(label_indices[c])
    
    # First distribute from clients with excess samples
    for client_idx in range(num_clients):
        if client_sample_counts[client_idx] > target_samples_per_client:
            # This client has too many samples, remove some
            excess = client_sample_counts[client_idx] - target_samples_per_client
            excess_indices = random.sample(client_data_dict[client_idx], excess)
            client_data_dict[client_idx] = [idx for idx in client_data_dict[client_idx] if idx not in excess_indices]
            remaining_indices.extend(excess_indices)
            client_sample_counts[client_idx] = target_samples_per_client
    
    # Remove indices that are already assigned to clients
    for client_idx in range(num_clients):
        for idx in client_data_dict[client_idx]:
            if idx in remaining_indices:
                remaining_indices.remove(idx)
    
    # Now add samples to clients with too few
    for client_idx in range(num_clients):
        if client_sample_counts[client_idx] < target_samples_per_client:
            # This client needs more samples
            needed = target_samples_per_client - client_sample_counts[client_idx]
            if needed > 0:
                additional_indices = random.sample(remaining_indices, min(needed, len(remaining_indices)))
                client_data_dict[client_idx].extend(additional_indices)
                for idx in additional_indices:
                    remaining_indices.remove(idx)
                client_sample_counts[client_idx] += len(additional_indices)
    
    # Print final distribution
    print("\nFinal client data distribution:")
    client_class_counts = {i: {c: 0 for c in range(num_classes)} for i in range(num_clients)}
    
    for client_idx, indices in client_data_dict.items():
        for idx in indices:
            label = dataset.targets[idx].item() if isinstance(dataset.targets[idx], torch.Tensor) else dataset.targets[idx]
            client_class_counts[client_idx][label] += 1
        
        print(f"Client {client_idx}: {len(indices)} samples, class distribution: {client_class_counts[client_idx]}")
    
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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer_names = []
        last_planes = self.in_planes
        strides = [1, 2, 2, 2]
        planes_list = [64, 128, 256, 512]

        for idx, num_block in enumerate(num_blocks):
            if num_block > 0:
                planes = planes_list[idx]
                stride = strides[idx]
                layer = self._make_layer(block, planes, num_block, stride)
                layer_name = f'layer{idx + 1}'
                setattr(self, layer_name, layer)
                self.layer_names.append(layer_name)
                last_planes = planes * block.expansion

        self.linear = nn.Linear(last_planes, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer_name in self.layer_names:
            out = getattr(self, layer_name)(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)

        features = out
        logits   = self.linear(features)

        return logits, features

def ResNet8(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)
