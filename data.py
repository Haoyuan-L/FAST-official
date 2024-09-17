import torch
import contextlib
from torch.utils.data import DataLoader
from medmnist.dataset import PathMNIST, DermaMNIST
from torchvision.datasets import CIFAR10, SVHN
import torchvision.transforms as transforms
import numpy as np
from utils import *


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

def get_data(dataset_name="cifar10", id=0, num_clients=10, return_eval_ds=False, batch_size=128, 
             split=None, alpha=None, num_workers=4, seed=0, data_dir="./data"):
    
    # Choose dataset based on the provided name
    if dataset_name.lower() == "cifar10":
        with contextlib.redirect_stdout(None):
            train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=get_transforms(dataset_name, augmentation=True))
        with contextlib.redirect_stdout(None):
            test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=get_transforms(dataset_name, augmentation=False))
        num_classes = 10

    elif dataset_name.lower() == "svhn":
        with contextlib.redirect_stdout(None):
            train_dataset = SVHN(root=data_dir, split='train', download=True, transform=get_transforms(dataset_name, augmentation=True))
        with contextlib.redirect_stdout(None):
            test_dataset = SVHN(root=data_dir, split='test', download=True, transform=get_transforms(dataset_name, augmentation=False))
        num_classes = 10
    elif dataset_name.lower() == "pathmnist":
        with contextlib.redirect_stdout(None):
            train_dataset = PathMNIST(root=data_dir, train=True, download=True, transform=get_transforms(dataset_name, augmentation=True))
        with contextlib.redirect_stdout(None):
            test_dataset = PathMNIST(root=data_dir, train=False, download=True, transform=get_transforms(dataset_name, augmentation=False))
        num_classes = 9
    elif dataset_name.lower() == "dermamnist":
        with contextlib.redirect_stdout(None):
            train_dataset = DermaMNIST(root=data_dir, train=True, split="train", download=True, transform=get_transforms(dataset_name, augmentation=True))
        with contextlib.redirect_stdout(None):
            test_dataset = DermaMNIST(root=data_dir, train=False, split="test", download=True, transform=get_transforms(dataset_name, augmentation=False))
        num_classes = 7
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

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