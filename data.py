import torch
import torch.nn.functional as F
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


def extract_labels(dataset):
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    elif hasattr(dataset, 'target'):
        labels = dataset.target
    elif hasattr(dataset, 'label'):
        labels = dataset.label
    else:
        raise AttributeError("Dataset does not have a known attribute for labels")

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)
    elif isinstance(labels, np.ndarray):
        pass
    else:
        labels = np.array(labels)
    return labels

def compute_entropy(logits):
    probs = logits / np.sum(logits, axis=1, keepdims=True)
    # Compute entropy for each sample
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    return entropy

def compute_norm(logits):
    predicted_labels = np.argmax(logits, axis=1) 
    # One-hot encode the predicted labels
    one_hot_predicted = np.zeros_like(logits) 
    one_hot_predicted[np.arange(len(predicted_labels)), predicted_labels] = 1.0
    # Compute L2 norm between logits and one-hot encoded predictions
    uncertainty = np.linalg.norm(logits - one_hot_predicted, axis=1) 
    return uncertainty

def compute_least_confidence(logits):
    max_probs = np.max(logits, axis=1)
    # Least confidence uncertainty
    uncertainty = 1.0 - max_probs
    return uncertainty

def compute_smallest_margin(logits):
    # Sort logits in descending order
    sorted_logits = np.sort(logits, axis=1)[:, ::-1]  
    margin = sorted_logits[:, 0] - sorted_logits[:, 1]
    # Smallest margin uncertainty
    uncertainty = 1.0 - margin
    return uncertainty

def compute_largest_margin(logits):
    # Maximum and minimum logits per sample
    max_logits = np.max(logits, axis=1)
    min_logits = np.min(logits, axis=1)

    # Compute margin
    margin = max_logits - min_logits
    # Largest margin uncertainty
    uncertainty = 1.0 - margin
    return uncertainty

def get_embeddings(dataset, model, device, fname, lname, batch_size=64, save_path=None):
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

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(embeddings, save_path+fname)
        np.save(save_path+lname, labels)
    return embeddings, labels

def get_logits_from_knn(k, indices, labeled_labels, num_classes):
    logits = []
    for neighbor_indices in indices:
        neighbor_labels = labeled_labels[neighbor_indices]
        # Debugging statements
        print(f"Neighbor Labels Shape: {neighbor_labels.shape}")
        print(f"Neighbor Labels Sample: {neighbor_labels[:5]}")
        
        if neighbor_labels.ndim > 1:
            neighbor_labels = np.argmax(neighbor_labels, axis=1)
        else:
            neighbor_labels = neighbor_labels.flatten()
        neighbor_labels = neighbor_labels.astype(int)
        label_count = Counter(neighbor_labels)
        # Create a probability distribution (logit format) over all classes
        logit = np.zeros(num_classes)
        for label, count in label_count.items():
            logit[label] = count / k # Fraction of neighbors that belong to the class
        logits.append(logit)
    return np.array(logits)

def get_data(dataset_name="cifar10", id=0, num_clients=10, return_eval_ds=False, batch_size=128, 
             split=None, alpha=None, num_workers=4, seed=0, data_dir="./data", class_aware=False, uncertainty="norm"):

    # load the OpenCLIP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-SigLIP-512', 'webli')
    model.to(device)
    model.eval()
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
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
            train_dataset = PathMNIST(root=data_dir, split="train", download=True, transform=get_transforms(dataset_name, augmentation=True))
            train_dataset_for_embeddings = PathMNIST(root=data_dir, split="train", download=True, transform=preprocess)
        with contextlib.redirect_stdout(None):
            test_dataset = PathMNIST(root=data_dir, split="test", download=True, transform=get_transforms(dataset_name, augmentation=False))
        num_classes = 9

    elif dataset_name.lower() == "dermamnist":
        with contextlib.redirect_stdout(None):
            train_dataset = DermaMNIST(root=data_dir, split="train", download=True, transform=get_transforms(dataset_name, augmentation=True))
            train_dataset_for_embeddings = DermaMNIST(root=data_dir, split="train", download=True, transform=preprocess)
        with contextlib.redirect_stdout(None):
            test_dataset = DermaMNIST(root=data_dir, split="test", download=True, transform=get_transforms(dataset_name, augmentation=False))
        num_classes = 7
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
 
    if os.path.exists(f"{dataset_name}_{uncertainty}_labels.npy"):
        all_labels = np.load(f"{dataset_name}_{uncertainty}_labels.npy")
    else:
        # balancely select 1% of the data as the initial labeled training set, and the rest as the unlabeled pool
        initial_labeled_ratio = 0.01
        np.random.seed(1)
        total_samples = len(train_dataset)
        num_labeled_samples = int(total_samples * initial_labeled_ratio)
        num_per_class = num_labeled_samples // num_classes
        
        try:
            labels = extract_labels(train_dataset)
        except AttributeError as e:
            raise AttributeError(f"Failed to extract labels for dataset '{dataset_name}': {str(e)}")
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
        # Load the embeddings if they exist
        save_path = "./embeddings/"
        labeled_embeddings_fname = f"{dataset_name}_embeddings_labeled.pt"
        unlabeled_embeddings_fname = f"{dataset_name}_embeddings_unlabeled.pt"
        labeled_labels_fname = f"{dataset_name}_labeled_labels.npy"
        unlabeled_labels_fname = f"{dataset_name}_unlabeled_labels.npy"

        if os.path.exists(save_path + labeled_embeddings_fname):
            labeled_embeddings = torch.load(save_path + labeled_embeddings_fname)
            labeled_labels = np.load(save_path + labeled_labels_fname)
        else:
            labeled_embeddings, labeled_labels = get_embeddings(
                labeled_subset, model, device, fname=labeled_embeddings_fname, lname=labeled_labels_fname, batch_size=batch_size, save_path=save_path
            )
        if os.path.exists(save_path + unlabeled_embeddings_fname):
            unlabeled_embeddings = torch.load(save_path + unlabeled_embeddings_fname)
            unlabled_ground_truth = np.load(save_path + unlabeled_labels_fname)
        else:
            unlabeled_embeddings, unlabled_ground_truth = get_embeddings(
                unlabeled_subset, model, device, fname=unlabeled_embeddings_fname, lname=unlabeled_labels_fname, batch_size=batch_size, save_path=save_path
            )

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

        # load uncertainty method
        if uncertainty == "norm":
            uncertainty_func = compute_norm
        elif uncertainty == "entropy":
            uncertainty_func = compute_entropy
        elif uncertainty == "least_confidence":
            uncertainty_func = compute_least_confidence
        elif uncertainty == "smallest_margin":
            uncertainty_func = compute_smallest_margin
        elif uncertainty == "largest_margin":
            uncertainty_func = compute_largest_margin
        else:
            raise ValueError(f"Unknown uncertainty method: {uncertainty}")

        # select the most uncertain samples for manual labeling (Oracle)
        query_ratio = 0.05
        uncertainty_score = uncertainty_func(logits)
        num_query_samples = int(query_ratio * len(unlabeled_indices))
        class_aware = True  # Set this to False for class-agnostic sampling

        if class_aware:
            predicted_labels = np.argmax(logits, axis=1)
            num_classes = logits.shape[1]
            num_query_samples_per_class = num_query_samples // num_classes
            uncertain_sample_indices = []

            for c in range(num_classes):
                # Get indices of samples predicted to belong to class c
                class_member_mask = predicted_labels == c
                cls_indices = np.where(class_member_mask)[0]
                cls_uncertainty = uncertainty_score[cls_indices]
                cls_uncertainty_order = np.argsort(-cls_uncertainty)
                num_samples = min(len(cls_uncertainty_order), num_query_samples_per_class)
                cls_uncertain_indices = cls_indices[cls_uncertainty_order[:num_samples]]
                uncertain_sample_indices.extend(cls_uncertain_indices.tolist())

            # Handle any remaining samples to meet the total query quota
            total_selected = len(uncertain_sample_indices)
            if total_selected < num_query_samples:
                remaining_samples = num_query_samples - total_selected
                selected_set = set(uncertain_sample_indices)
                remaining_indices = [i for i in range(len(unlabeled_indices)) if i not in selected_set]
                remaining_uncertainty = uncertainty_score[remaining_indices]
                remaining_uncertainty_order = np.argsort(-remaining_uncertainty)
                additional_uncertain_indices = [remaining_indices[i] for i in remaining_uncertainty_order[:remaining_samples]]
                uncertain_sample_indices.extend(additional_uncertain_indices)
            elif total_selected > num_query_samples:
                uncertain_sample_indices = uncertain_sample_indices[:num_query_samples]

            # Map back to dataset indices
            uncertain_sample_dataset_indices = np.array(unlabeled_indices)[uncertain_sample_indices]
            oracle_annotation_labels = unlabeled_ground_truth[uncertain_sample_indices]
            all_labels[uncertain_sample_dataset_indices] = oracle_annotation_labels
        else:
            uncertainty_order = np.argsort(-uncertainty_score)
            uncertain_indices = uncertainty_order[:num_query_samples]
            uncertain_sample_indices = np.array(unlabeled_indices)[uncertain_indices]
            oracle_annotation_labels = unlabeled_ground_truth[uncertain_indices]
            all_labels[uncertain_sample_indices] = oracle_annotation_labels

        np.save(f"{dataset_name}_{uncertainty}_labels.npy", all_labels)

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
