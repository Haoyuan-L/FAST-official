import torch
import contextlib
from torch.utils.data import DataLoader
from medmnist.dataset import PathMNIST, DermaMNIST
from torchvision.datasets import CIFAR10, SVHN
import torchvision.transforms as transforms
import numpy as np
from utils import *
import open_clip
import faiss
from collections import Counter
import time

def get_transforms(dataset_name, augmentation=True):
    """Returns the appropriate transformations based on the dataset name."""
    
    MEAN = {'mnist': (0.1307,), 'fmnist': (0.5,), 'emnist': (0.5,), 'svhn': [0.4376821, 0.4437697, 0.47280442], 
            'cifar10': [0.485, 0.456, 0.406], 'cifar100': [0.507, 0.487, 0.441], 'pathmnist': (0.5,), 
            'octmnist': (0.5,), 'organamnist': (0.5,), 'dermamnist': (0.5,), 'bloodmnist': (0.5,)}
    STD = {'mnist': (0.3081,), 'fmnist': (0.5,), 'emnist': (0.5,), 'svhn': [0.19803012, 0.20101562, 0.19703614], 
           'cifar10': [0.229, 0.224, 0.225], 'cifar100': [0.267, 0.256, 0.276], 'pathmnist': (0.5,),
           'octmnist': (0.5,), 'organamnist': (0.5,), 'dermamnist': (0.5,), 'bloodmnist': (0.5,)}

    if augmentation:
        data_transform = [transforms.RandomHorizontalFlip(), 
			              transforms.ToTensor(), 
			              transforms.Normalize(mean=MEAN[dataset_name], std=STD[dataset_name])]
    else:
        data_transform = [transforms.ToTensor(), 
						  transforms.Normalize(mean=MEAN[dataset_name], std=STD[dataset_name])]

    return transforms.Compose(data_transform)

def get_embeddings(dataset, model, device, batch_size=64):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            # Obtain image embeddings
            features = model.encode_image(images)
            embeddings.append(features.cpu())
            labels.extend(targets.numpy())
    embeddings = torch.cat(embeddings)
    labels = np.array(labels)
    return embeddings, labels

# Calculate logit (probability distribution over classes)
def get_logits_from_knn(k, indices, labeled_labels, num_classes):
    logits = []
    for neighbor_indices in indices:
        # Get the labels of the k-nearest neighbors
        neighbor_labels = labeled_labels[neighbor_indices]
        # Count occurrences of each label
        label_count = Counter(neighbor_labels)
        # Create a probability distribution (logit format) over all classes
        logit = np.zeros(num_classes)
        for label, count in label_count.items():
            logit[label] = count / k  # Fraction of neighbors that belong to the class
        logits.append(logit)
    return np.array(logits)

def compute_entropy(logits):
    probs = logits / np.sum(logits, axis=1, keepdims=True)
    # Compute entropy for each sample
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    return entropy

def get_data(dataset_name="cifar10", id=0, num_clients=10, return_eval_ds=False, batch_size=128, 
             split=None, alpha=None, num_workers=4, seed=0, data_dir="./data"):
    
    # load the OpenCLIP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-SigLIP-512', 'webli')
    model.to(device)
    model.eval()

    # Choose dataset based on the provided name
    if dataset_name.lower() == "cifar10":
        with contextlib.redirect_stdout(None):
            train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=get_transforms(dataset_name, augmentation=True))
            train_dataset_for_embeddings = CIFAR10(root=data_dir, train=True, download=True, transform=preprocess)
        with contextlib.redirect_stdout(None):
            test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=get_transforms(dataset_name, augmentation=False))
        num_classes = 10

    elif dataset_name.lower() == "svhn":
        with contextlib.redirect_stdout(None):
            train_dataset = SVHN(root=data_dir, split='train', download=True, transform=get_transforms(dataset_name, augmentation=True))
            train_dataset_for_embeddings = SVHN(root=data_dir, split='train', download=True, transform=preprocess)
        with contextlib.redirect_stdout(None):
            test_dataset = SVHN(root=data_dir, split='test', download=True, transform=get_transforms(dataset_name, augmentation=False))
        num_classes = 10

    elif dataset_name.lower() == "pathmnist":
        with contextlib.redirect_stdout(None):
            train_dataset = PathMNIST(root=data_dir, train=True, download=True, transform=get_transforms(dataset_name, augmentation=True))
            train_dataset_for_embeddings = PathMNIST(root=data_dir, train=True, download=True, transform=preprocess)
        with contextlib.redirect_stdout(None):
            test_dataset = PathMNIST(root=data_dir, train=False, download=True, transform=get_transforms(dataset_name, augmentation=False))
        num_classes = 9

    elif dataset_name.lower() == "dermamnist":
        with contextlib.redirect_stdout(None):
            train_dataset = DermaMNIST(root=data_dir, train=True, split="train", download=True, transform=get_transforms(dataset_name, augmentation=True))
            train_dataset_for_embeddings = DermaMNIST(root=data_dir, train=True, split="train", download=True, transform=preprocess)
        with contextlib.redirect_stdout(None):
            test_dataset = DermaMNIST(root=data_dir, train=False, split="test", download=True, transform=get_transforms(dataset_name, augmentation=False))
        num_classes = 7
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
 
    if os.path.exists(f"{dataset_name}_labels.npy"):
        all_labels = np.load(f"{dataset_name}_labels.npy")
    else:
        # balancely select 1% of the data as the initial labeled training set, and the rest as the unlabeled pool
        initial_labeled_ratio = 0.01
        np.random.seed(1)
        total_samples = len(train_dataset)
        num_labeled_samples = int(total_samples * initial_labeled_ratio)
        num_per_class = num_labeled_samples // num_classes
        
        labels = np.array(train_dataset.targets)
        indices_per_class = {i: np.where(labels == i)[0] for i in range(num_classes)}
        labeled_indices = []
        for idx in range(num_classes):
            class_indices = indices_per_class[idx]
            selected_indices = np.random.choice(class_indices, num_per_class, replace=False)
            labeled_indices.extend(selected_indices)
        labeled_indices = labeled_indices[:num_labeled_samples]
        all_indices = set(range(total_samples))
        unlabeled_indices = list(all_indices - set(labeled_indices))

        labeled_subset = torch.utils.data.Subset(train_dataset_for_embeddings, labeled_indices)
        unlabeled_subset = torch.utils.data.Subset(train_dataset_for_embeddings, unlabeled_indices)

        # Utilize SIGLIP encoder to encode all the training data
        labeled_embeddings, labeled_labels = get_embeddings(labeled_subset, model, device, batch_size)
        unlabeled_embeddings, unlabled_ground_truth = get_embeddings(unlabeled_subset, model, device, batch_size)

        # Apply FAISS-KNN to embedings for data labeling
        labeled_embeddings = labeled_embeddings.numpy().astype('float32')
        unlabeled_embeddings = unlabeled_embeddings.numpy().astype('float32')

        index = faiss.IndexFlatL2(labeled_embeddings.shape[1])
        index.add(labeled_embeddings)
        k = 10
        distances, indices = index.search(unlabeled_embeddings, k)
        logits = get_logits_from_knn(k, indices, labeled_labels, num_classes)
        predicted_labels = np.array(np.argmax(logits, axis=1))
        unlabeled_ground_truth = np.array(unlabled_ground_truth)
        corrects = np.sum(predicted_labels == unlabeled_ground_truth)
        labeling_acc = corrects / len(unlabeled_ground_truth)
        print(f"Labeling Accuracy: {labeling_acc * 100:.2f}%")

        all_labels = np.zeros(len(train_dataset), dtype=int)
        all_labels[labeled_indices] = labeled_labels
        all_labels[unlabeled_indices] = predicted_labels

        # select the most uncertain samples for manual labeling (Oracle)
        query_ratio = 0.05
        entropy = compute_entropy(logits)
        num_query_samples = int(query_ratio * len(unlabeled_indices))
        uncertainty_order = np.argsort(-entropy)
        uncertain_indices = uncertainty_order[:num_query_samples]
        uncertain_sample_indices = np.array(unlabeled_indices)[uncertain_indices]
        oracle_annotation_labels = unlabeled_ground_truth[uncertain_indices]
        all_labels[uncertain_sample_indices] = oracle_annotation_labels
        np.save(f"{dataset_name}_labels.npy", all_labels)

        # labeling accuracy after first AL round
        updated_unlabeled_labels = all_labels[unlabeled_indices]
        corrects = np.sum(updated_unlabeled_labels == unlabeled_ground_truth)
        new_labeling_acc = corrects / len(unlabeled_ground_truth)
        print(f"Labeling Accuracy after first AL round: {new_labeling_acc * 100:.2f}%")
    
    train_dataset.targets = all_labels.tolist()
    # Return evaluation dataset if required
    if return_eval_ds:
        eval_loader = DataLoader(test_dataset, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)
        num_samples = len(test_dataset)
        return eval_loader, num_classes, num_samples
    else:
        if split == "dir_balance":
            # Call dir_balance function
            clients_data, sample = dir_balance(
                dataset=train_dataset,
                dataset_name=dataset_name,
                num_classes=num_classes,
                num_users=num_clients,
                alpha=alpha,
                data_dir=data_dir,
                sample=None
            )
            # Get the train indices for the specific client
            train_indices = clients_data[int(id)]
            # Create a subset of the train dataset for the client
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        else:
            split_fn = get_split_fn(split)
            # Split data into client-specific subsets
            train_indices = split_fn(idxs=train_dataset.targets, num_shards=num_clients,
                                    num_samples=len(train_dataset), num_classes=num_classes, seed=seed)[int(id)]
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        num_samples = len(train_indices)

        return train_loader, num_classes, num_samples