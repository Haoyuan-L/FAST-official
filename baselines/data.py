import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets
import medmnist
from medmnist import INFO

def get_dataset(args):
    """Get dataset based on configuration."""
    MEAN = {
        'mnist': (0.1307,), 
        'fmnist': (0.5,), 
        'emnist': (0.5,), 
        'svhn': [0.4376821, 0.4437697, 0.47280442], 
        'cifar10': [0.485, 0.456, 0.406], 
        'cifar100': [0.507, 0.487, 0.441], 
        'pathmnist': (0.5,), 
        'octmnist': (0.5,), 
        'organamnist': (0.5,), 
        'dermamnist': (0.5,), 
        'bloodmnist': (0.5,)
    }
    
    STD = {
        'mnist': (0.3081,), 
        'fmnist': (0.5,), 
        'emnist': (0.5,), 
        'svhn': [0.19803012, 0.20101562, 0.19703614], 
        'cifar10': [0.229, 0.224, 0.225], 
        'cifar100': [0.267, 0.256, 0.276], 
        'pathmnist': (0.5,),
        'octmnist': (0.5,), 
        'organamnist': (0.5,), 
        'dermamnist': (0.5,), 
        'bloodmnist': (0.5,)
    }
    
    # Define data transformations
    noaug = [
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN[args.dataset], std=STD[args.dataset])
    ]
    
    weakaug = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN[args.dataset], std=STD[args.dataset])
    ]
    
    trans_noaug = transforms.Compose(noaug)
    trans_weakaug = transforms.Compose(weakaug)
    
    # Create dataset directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    print(f'Loading Dataset: {args.dataset}')
    
    # Standard benchmark datasets
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST(args.data_dir, train=True, download=True, transform=trans_weakaug)
        dataset_query = datasets.MNIST(args.data_dir, train=True, download=True, transform=trans_noaug)
        dataset_test = datasets.MNIST(args.data_dir, train=False, download=True, transform=trans_noaug)
    
    elif args.dataset == "fmnist":
        dataset_train = datasets.FashionMNIST(args.data_dir, download=True, train=True, transform=trans_weakaug)
        dataset_query = datasets.FashionMNIST(args.data_dir, download=True, train=True, transform=trans_noaug)
        dataset_test = datasets.FashionMNIST(args.data_dir, download=True, train=False, transform=trans_noaug)

    elif args.dataset == 'emnist':
        dataset_train = datasets.EMNIST(args.data_dir, split='byclass', train=True, download=True, transform=trans_weakaug)
        dataset_query = datasets.EMNIST(args.data_dir, split='byclass', train=True, download=True, transform=trans_noaug)
        dataset_test = datasets.EMNIST(args.data_dir, split='byclass', train=False, download=True, transform=trans_noaug)

    elif args.dataset == 'svhn':
        dataset_train = datasets.SVHN(args.data_dir, 'train', download=True, transform=trans_weakaug)
        dataset_query = datasets.SVHN(args.data_dir, 'train', download=True, transform=trans_noaug)
        dataset_test = datasets.SVHN(args.data_dir, 'test', download=True, transform=trans_noaug)
            
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=trans_weakaug)
        dataset_query = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=trans_noaug)
        dataset_test = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=trans_noaug)
            
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(args.data_dir, train=True, download=True, transform=trans_weakaug)
        dataset_query = datasets.CIFAR100(args.data_dir, train=True, download=True, transform=trans_noaug)
        dataset_test = datasets.CIFAR100(args.data_dir, train=False, download=True, transform=trans_noaug)

    # Medical benchmark datasets
    elif args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
        DataClass = getattr(medmnist, INFO[args.dataset]['python_class'])
        
        dataset_train = DataClass(download=True, split='train', transform=trans_weakaug)
        dataset_query = DataClass(download=True, split='train', transform=trans_noaug)
        dataset_test = DataClass(download=True, split='test', transform=trans_noaug)
        
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')
    
    # Set dataset specific attributes
    if args.dataset in ['mnist', 'fmnist', 'emnist', 'cifar10', 'cifar100']:
        dataset_train.targets = torch.tensor(dataset_train.targets)
        dataset_query.targets = torch.tensor(dataset_query.targets)
        dataset_test.targets = torch.tensor(dataset_test.targets)
    elif args.dataset == 'svhn':
        dataset_train.targets = torch.tensor(dataset_train.labels)
        dataset_query.targets = torch.tensor(dataset_query.labels)
        dataset_test.targets = torch.tensor(dataset_test.labels)
    elif args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
        dataset_train.targets = torch.tensor(dataset_train.labels).squeeze()
        dataset_query.targets = torch.tensor(dataset_query.labels).squeeze()
        dataset_test.targets = torch.tensor(dataset_test.labels).squeeze()
    
    # Get number of classes and input channels
    if args.dataset == 'mnist':
        args.num_classes = 10
        args.in_channels = 1
        args.img_size = 28
    elif args.dataset == 'fmnist':
        args.num_classes = 10
        args.in_channels = 1
        args.img_size = 28
    elif args.dataset == 'emnist':
        args.num_classes = 62
        args.in_channels = 1
        args.img_size = 28
    elif args.dataset == 'svhn':
        args.num_classes = 10
        args.in_channels = 3
        args.img_size = 32
    elif args.dataset == 'cifar10':
        args.num_classes = 10
        args.in_channels = 3
        args.img_size = 32
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.in_channels = 3
        args.img_size = 32
    elif args.dataset == 'pathmnist':
        args.num_classes = len(INFO[args.dataset]['label'])
        args.in_channels = INFO[args.dataset]['n_channels']
        args.img_size = 28
    elif args.dataset == 'octmnist':
        args.num_classes = len(INFO[args.dataset]['label'])
        args.in_channels = INFO[args.dataset]['n_channels']
        args.img_size = 28
    elif args.dataset == 'organamnist':
        args.num_classes = len(INFO[args.dataset]['label'])
        args.in_channels = INFO[args.dataset]['n_channels']
        args.img_size = 28
    elif args.dataset == 'dermamnist':
        args.num_classes = len(INFO[args.dataset]['label'])
        args.in_channels = INFO[args.dataset]['n_channels']
        args.img_size = 28
    elif args.dataset == 'bloodmnist':
        args.num_classes = len(INFO[args.dataset]['label'])
        args.in_channels = INFO[args.dataset]['n_channels']
        args.img_size = 28
    
    # Calculate total data size and budgets
    args.total_data = len(dataset_train)
    args.initial_budget_size = int(args.total_data * args.initial_budget / args.num_clients)  # Per client
    args.query_budget_per_round = int(args.total_data * args.query_budget)  # Total budget
    
    print(f"Total training data: {args.total_data}")
    print(f"Initial budget per client: {args.initial_budget_size}")
    print(f"Query budget per round (total): {args.query_budget_per_round}")
    print(f"Query budget per client: {args.query_budget_per_round // args.num_clients}")
    
    return dataset_train, dataset_query, dataset_test
