import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import contextlib
from torch.utils.data import DataLoader, TensorDataset
from medmnist.dataset import PathMNIST, DermaMNIST
from torchvision.datasets import CIFAR10, SVHN, CIFAR100
import torchvision.transforms as transforms
import numpy as np
from utils import *
import open_clip
import faiss
from collections import Counter
import time
from scipy.stats import mode

def get_transforms(dataset_name, augmentation=True):
    """Returns the appropriate transformations based on the dataset name."""
    
    MEAN = {'mnist': (0.1307,), 'fmnist': (0.5,), 'emnist': (0.5,), 'svhn': [0.4376821, 0.4437697, 0.47280442], 
            'cifar10': [0.485, 0.456, 0.406], 'cifar100': [0.507, 0.487, 0.441], 'pathmnist': (0.5,), 
            'octmnist': (0.5,), 'organamnist': (0.5,), 'dermamnist': (0.5,), 'bloodmnist': (0.5,), 'tiny-imagenet': [0.480, 0.448, 0.398]}
    STD = {'mnist': (0.3081,), 'fmnist': (0.5,), 'emnist': (0.5,), 'svhn': [0.19803012, 0.20101562, 0.19703614], 
           'cifar10': [0.229, 0.224, 0.225], 'cifar100': [0.267, 0.256, 0.276], 'pathmnist': (0.5,),
           'octmnist': (0.5,), 'organamnist': (0.5,), 'dermamnist': (0.5,), 'bloodmnist': (0.5,), 'tiny-imagenet': [0.277, 0.269, 0.282]}

    if augmentation:
        if dataset_name.lower() == "tiny-imagenet":
            data_transform = [
                            #transforms.RandomCrop(64, padding=4),
                            transforms.RandomHorizontalFlip(), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=MEAN[dataset_name], std=STD[dataset_name])]
        elif dataset_name.lower() in ["cifar10", "cifar100", "svhn"]:
            data_transform = [
                            #transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=MEAN[dataset_name], std=STD[dataset_name])]
        else:
            data_transform = [
                            #transforms.RandomCrop(28, padding=4),
                            transforms.RandomHorizontalFlip(), 
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
    labels = labels.flatten()
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
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            # Check if the model has 'encode_image' method
            if hasattr(model, 'encode_image'):
                embeddings = model.encode_image(images).float()
            else:
                embeddings = model(images).float()
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)
    all_labels = all_labels.cpu().numpy()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(all_embeddings, save_path+fname)
        torch.save(all_labels, save_path+lname)
    return all_embeddings, all_labels

def get_labels_from_knn(k, indices, labeled_labels, num_classes):
    logits = []
    for neighbor_indices in indices:
        neighbor_labels = labeled_labels[neighbor_indices]

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

def get_min_distance_logits(unlabeled_embeddings, labeled_embeddings, labeled_labels, num_classes):
    """
    Calculate logits based on the minimum distance from each unlabeled sample to each class.
    """
    min_distances = np.full((unlabeled_embeddings.shape[0], num_classes), np.inf, dtype=np.float32)
    
    for cls in range(num_classes):
        # Get embeddings of the current class
        cls_indices = np.where(labeled_labels == cls)[0]
        cls_embeddings = labeled_embeddings[cls_indices]
        
        if cls_embeddings.shape[0] == 0:
            raise ValueError(f"No samples found for class {cls}")
        
        # Build FAISS index for the current class
        index = faiss.IndexFlatL2(labeled_embeddings.shape[1])
        index.add(cls_embeddings.astype('float32'))
        
        # Perform 1-NN search to find the nearest sample in this class
        distances, _ = index.search(unlabeled_embeddings.astype('float32'), 1)
        
        # Take the square root of squared L2 distances to get actual Euclidean distances
        min_distances[:, cls] = np.sqrt(distances).flatten()
    
    return min_distances

def get_average_distance_logits(unlabeled_embeddings, labeled_embeddings, labeled_labels, num_classes):
    """
    Calculate logits based on the average distance from each unlabeled sample to each class.
    """
    average_distances = np.zeros((unlabeled_embeddings.shape[0], num_classes), dtype=np.float32)
    
    for cls in range(num_classes):
        cls_indices = np.where(labeled_labels == cls)[0]
        cls_embeddings = labeled_embeddings[cls_indices]
        if cls_embeddings.shape[0] == 0:
            raise ValueError(f"No samples found for class {cls}")
        
        # Compute the differences between each unlabeled sample and all labeled samples in the class
        # Shape: (N, M, D)
        diff = unlabeled_embeddings[:, np.newaxis, :] - cls_embeddings[np.newaxis, :, :]
        sq_distances = np.sum(diff ** 2, axis=2)  # Shape: (N, M)
        distances = np.sqrt(sq_distances)  # Shape: (N, M)
        
        # Compute the average distance for each unlabeled sample to this class
        average_distances[:, cls] = np.mean(distances, axis=1)
    
    return average_distances

def get_average_cosine_similarity_logits(unlabeled_embeddings, labeled_embeddings, labeled_labels, num_classes):
    """
    Calculation of logits based on the average cosine similarity from each unlabeled sample to each class.
    """

    labeled_labels = labeled_labels.flatten()
    # Normalize embeddings
    unlabeled_norm = unlabeled_embeddings / (np.linalg.norm(unlabeled_embeddings, axis=1, keepdims=True) + 1e-10)
    labeled_norm = labeled_embeddings / (np.linalg.norm(labeled_embeddings, axis=1, keepdims=True) + 1e-10)
    
    average_similarities = np.zeros((unlabeled_embeddings.shape[0], num_classes), dtype=np.float32)
    
    for cls in range(num_classes):
        cls_embeddings = labeled_norm[labeled_labels == cls]
        if cls_embeddings.shape[0] == 0:
            raise ValueError(f"No samples found for class {cls}")
        
        # Compute cosine similarities
        similarities = np.dot(unlabeled_norm, cls_embeddings.T)  # Shape: (N, M_cls)
        
        # Compute average similarity
        average_similarities[:, cls] = similarities.mean(axis=1)
    
    return average_similarities

def train_linear_classifier(labeled_embeddings, labeled_labels, num_classes, device, epochs=20, batch_size=128, learning_rate=1e-3):
    """
    Trains a linear classifier on the labeled embeddings.
    """
    # Convert to torch tensors
    embeddings = torch.tensor(labeled_embeddings, dtype=torch.float32)
    labels = torch.tensor(labeled_labels, dtype=torch.long)
    dataset = CustomTensorDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = embeddings.shape[1]
    model = nn.Linear(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_embeddings, batch_labels in dataloader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_embeddings.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    model.eval()
    return model

def get_linear_classifier_logits(model, unlabeled_embeddings, device, batch_size=128):
    """
    Obtains logits from the linear classifier for unlabeled embeddings.
    """
    embeddings = torch.tensor(unlabeled_embeddings, dtype=torch.float32)
    dataset = TensorDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    logits_list = []
    with torch.no_grad():
        for (batch_embeddings,) in dataloader:
            batch_embeddings = batch_embeddings.to(device)
            outputs = model(batch_embeddings)
            logits_list.append(outputs.cpu().numpy())
    logits = np.vstack(logits_list)
    return logits

def get_data(dataset_name="cifar10", id=0, num_clients=10, return_eval_ds=False, batch_size=128, embed_input=False, encoder="SigLIP",
             split=None, alpha=None, num_workers=4, seed=0, data_dir="./data", class_aware=False, uncertainty="norm", active_oracle=True, 
             budget=0.1, initial_only=False, initial_with_random=False):

    # load the OpenCLIP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if encoder == "SigLIP":
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-SigLIP-512', 'webli')
    elif encoder == "CLIP":
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', 'claion2b_s34b_b88klip')
    elif encoder == "EvaCLIP":
        model, _, preprocess = open_clip.create_model_and_transforms('EVA02-B-16', 'merged2b_s8b_b131k')
    elif encoder == "DINOv2":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # Define the preprocessing transforms
        preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    model.to(device)
    model.eval()
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # Path and file names for embeddings and labels
    save_path = "./embeddings/"
    # Define filenames for embeddings and labels
    train_embeddings_fname = f"{dataset_name}_embeddings_train.pt"
    train_labels_fname = f"{dataset_name}_train_labels.pt"
    
    # Choose dataset based on the provided name
    if dataset_name.lower() == "cifar10":
        with contextlib.redirect_stdout(None):
            train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=get_transforms(dataset_name, augmentation=True))
            train_dataset_for_embeddings = CIFAR10(root=data_dir, train=True, download=True, transform=preprocess)
            test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=get_transforms(dataset_name, augmentation=False))
            test_dataset_for_embeddings = CIFAR10(root=data_dir, train=False, download=True, transform=preprocess)
        num_classes = 10

    elif dataset_name.lower() == "svhn":
        with contextlib.redirect_stdout(None):
            train_dataset = SVHN(root=data_dir, split='train', download=True, transform=get_transforms(dataset_name, augmentation=True))
            train_dataset_for_embeddings = SVHN(root=data_dir, split='train', download=True, transform=preprocess)
            test_dataset = SVHN(root=data_dir, split='test', download=True, transform=get_transforms(dataset_name, augmentation=False))
            test_dataset_for_embeddings = SVHN(root=data_dir, split='test', download=True, transform=preprocess)
        num_classes = 10

    elif dataset_name.lower() == "pathmnist":
        with contextlib.redirect_stdout(None):
            train_dataset = PathMNIST(root=data_dir, split="train", download=True, transform=get_transforms(dataset_name, augmentation=True))
            train_dataset_for_embeddings = PathMNIST(root=data_dir, split="train", download=True, transform=preprocess)
            test_dataset = PathMNIST(root=data_dir, split="test", download=True, transform=get_transforms(dataset_name, augmentation=False))
            test_dataset_for_embeddings = PathMNIST(root=data_dir, split="test", download=True, transform=preprocess)
        num_classes = 9

    elif dataset_name.lower() == "dermamnist":
        with contextlib.redirect_stdout(None):
            train_dataset = DermaMNIST(root=data_dir, split="train", download=True, transform=get_transforms(dataset_name, augmentation=True))
            train_dataset_for_embeddings = DermaMNIST(root=data_dir, split="train", download=True, transform=preprocess)
            test_dataset = DermaMNIST(root=data_dir, split="test", download=True, transform=get_transforms(dataset_name, augmentation=False))
            test_dataset_for_embeddings = DermaMNIST(root=data_dir, split="test", download=True, transform=preprocess)
        num_classes = 7

    elif dataset_name.lower() == "cifar100":
        with contextlib.redirect_stdout(None):
            train_dataset = CIFAR100(root=data_dir, train=True, download=True, transform=get_transforms(dataset_name, augmentation=True))
            train_dataset_for_embeddings = CIFAR100(root=data_dir, train=True, download=True, transform=preprocess)
            test_dataset = CIFAR100(root=data_dir, train=False, download=True, transform=get_transforms(dataset_name, augmentation=False))
            test_dataset_for_embeddings = CIFAR100(root=data_dir, train=False, download=True, transform=preprocess)
        num_classes = 100
    
    elif dataset_name.lower() == "tiny-imagenet":
        with contextlib.redirect_stdout(None):
            train_dataset = TinyImageNet(root=data_dir, split="train", download=True, transform=get_transforms(dataset_name, augmentation=True))
            train_dataset_for_embeddings = TinyImageNet(root=data_dir, split="train", download=True, transform=preprocess)
            test_dataset = TinyImageNet(root=data_dir, split="val", download=True, transform=get_transforms(dataset_name, augmentation=False))
            test_dataset_for_embeddings = TinyImageNet(root=data_dir, split="val", download=True, transform=preprocess)
        num_classes = 200
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
 
    if initial_only and os.path.exists(f"{dataset_name}_initial_data.pt"):
        pass
    elif initial_with_random and os.path.exists(f"{dataset_name}_initial_with_random_data.pt"):
        pass
    elif os.path.exists(f"{dataset_name}_{uncertainty}_balance-{class_aware}_budget{budget}_labels.npy"):
        all_labels = np.load(f"{dataset_name}_{uncertainty}_balance-{class_aware}_budget{budget}_labels.npy")
    elif active_oracle == False and os.path.exists(f"{dataset_name}_None_labels.npy"):
        all_labels = np.load(f"{dataset_name}_None_labels.npy")
    else:
        # balancely select 1% of the data as the initial labeled training set, and the rest as the unlabeled pool
        initial_labeled_ratio = 0.01
        np.random.seed(seed)
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

        # initial_only training data
        initial_data = [train_dataset[i][0] for i in labeled_indices]
        initial_labels = [labels[i] for i in labeled_indices]
        torch.save(initial_data, f"{dataset_name}_initial_data.pt")
        torch.save(initial_labels, f"{dataset_name}_initial_labels.pt")
    
        initial_with_random = int(total_samples * budget)
        add_indices = np.random.choice(unlabeled_indices, initial_with_random, replace=False)
        # Extract additional data and labels
        additional_data = [train_dataset[i][0] for i in add_indices]
        additional_labels = [labels[i] for i in add_indices]
        updated_data = initial_data + additional_data
        updated_labels = initial_labels + additional_labels
        torch.save(updated_data, f"{dataset_name}_initial_with_random_data.pt")
        torch.save(updated_labels, f"{dataset_name}_initial_with_random_labels.pt")

        # Utilize SIGLIP encoder to encode all the training data
        # Load the embeddings if they exist

        if os.path.exists(os.path.join(save_path, train_embeddings_fname)):
            train_embeddings = torch.load(os.path.join(save_path, train_embeddings_fname))
            train_labels = torch.load(os.path.join(save_path, train_labels_fname))
        else:
            train_embeddings, train_labels = get_embeddings(
                train_dataset_for_embeddings, model, device, fname=train_embeddings_fname, lname=train_labels_fname, batch_size=batch_size, save_path=save_path
            )

        # Extract and save embeddings for initial data
        if not os.path.exists(f"{dataset_name}_initial_embeddings.pt"):
            initial_embeddings = train_embeddings[labeled_indices]
            torch.save(initial_embeddings, f"{dataset_name}_initial_embeddings.pt")
            # Extract and save embeddings for initial with random additional data
            initial_with_random_indices = labeled_indices + list(add_indices)
            initial_with_random_embeddings = train_embeddings[initial_with_random_indices]
            torch.save(initial_with_random_embeddings, f"{dataset_name}_initial_with_random_embeddings.pt")

        # Use indices to extract labeled and unlabeled embeddings and labels
        labeled_embeddings = train_embeddings[labeled_indices]
        labeled_labels = train_labels[labeled_indices]
        unlabeled_embeddings = train_embeddings[unlabeled_indices]
        unlabeled_ground_truth = train_labels[unlabeled_indices]

        # Apply FAISS-KNN to embeddings for data labeling
        labeled_embeddings_np = labeled_embeddings.numpy().astype('float32')
        unlabeled_embeddings_np = unlabeled_embeddings.numpy().astype('float32')
        index = faiss.IndexFlatL2(labeled_embeddings_np.shape[1])
        index.add(labeled_embeddings_np)
        k = 10
        distances, indices = index.search(unlabeled_embeddings_np, k)

        # Get the pseudo-labels for the unlabeled data via majority voting
        # label_frac = get_labels_from_knn(k, indices, labeled_labels, num_classes)
        # predicted_labels = np.array(np.argmax(label_frac, axis=1))
        neighbor_labels = labeled_labels[indices]
        predicted_labels, _ = mode(neighbor_labels, axis=1)
        predicted_labels = predicted_labels.flatten()

        # Evaluate labeling accuracy
        unlabeled_ground_truth = unlabeled_ground_truth.squeeze()
        corrects = np.sum(predicted_labels == unlabeled_ground_truth)
        labeling_acc = corrects / len(unlabeled_ground_truth)
        print(f"Labeling Accuracy: {labeling_acc * 100:.2f}%")

        # Create all_labels array with both labeled and pseudo-labeled data
        all_labels = np.zeros(len(train_dataset), dtype=int)
        all_labels[labeled_indices] = labeled_labels.flatten()
        all_labels[unlabeled_indices] = predicted_labels
        # save the pseudo labels
        np.save(f"{dataset_name}_None_labels.npy", all_labels)

        # 1.Calculate logits based on the minimum distance from each unlabeled sample to each class
        #logits = get_min_distance_logits(unlabeled_embeddings_np, labeled_embeddings_np, labeled_labels, num_classes)

        # 2. Based on average L2 distance
        #logits = get_average_distance_logits(unlabeled_embeddings_np, labeled_embeddings_np, labeled_labels, num_classes)

        # 3. Based on average cosine similarity
        logits = get_average_cosine_similarity_logits(unlabeled_embeddings_np, labeled_embeddings_np, labeled_labels, num_classes)

        # 4. Based on linear classifier
#        weak_labels = np.load(f"{dataset_name}_None_labels.npy")
#        weak_labels = weak_labels.flatten()
#        linear_model_fp = f"{dataset_name}_linear_classifier.pth"
#        if os.path.exists(linear_model_fp):
#            input_dim = labeled_embeddings_np.shape[1]
#            linear_model = nn.Linear(input_dim, num_classes).to(device)
#            linear_model.load_state_dict(torch.load(linear_model_fp, map_location=device))
#            linear_model.eval()
#        else:
#            linear_model = train_linear_classifier(labeled_embeddings_np, weak_labels, num_classes=num_classes, device=device)
#            torch.save(linear_model.state_dict(), linear_model_fp)
#        logits = get_linear_classifier_logits(linear_model, unlabeled_embeddings_np, device)

        if active_oracle:
            # Load uncertainty method
            if uncertainty.lower() == "norm":
                uncertainty_func = compute_norm
            elif uncertainty.lower() == "entropy":
                uncertainty_func = compute_entropy
            elif uncertainty.lower() == "least_confidence":
                uncertainty_func = compute_least_confidence
            elif uncertainty.lower() == "smallest_margin":
                uncertainty_func = compute_smallest_margin
            elif uncertainty.lower() == "largest_margin":
                uncertainty_func = compute_largest_margin
            elif uncertainty.lower() == "random":
                pass
            else:
                raise ValueError(f"Unknown uncertainty method: {uncertainty}")

            # Select samples based on the selected method
            query_ratio = budget
            num_query_samples = int(query_ratio * len(unlabeled_indices))

            if uncertainty.lower() == "random":
                # Non-class-aware random selection
                random_indices = np.random.choice(len(unlabeled_indices), size=num_query_samples, replace=False)
                random_sample_indices = np.array(unlabeled_indices)[random_indices]
                oracle_annotation_labels = unlabeled_ground_truth[random_indices]
                all_labels[random_sample_indices] = oracle_annotation_labels
            else:
                # Uncertainty-based selection
                uncertainty_score = uncertainty_func(logits)
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
                    all_labels[uncertain_sample_dataset_indices] = oracle_annotation_labels.flatten()
                else:
                    uncertainty_order = np.argsort(-uncertainty_score)
                    uncertain_indices = uncertainty_order[:num_query_samples]
                    uncertain_sample_indices = np.array(unlabeled_indices)[uncertain_indices]
                    oracle_annotation_labels = unlabeled_ground_truth[uncertain_indices]
                    all_labels[uncertain_sample_indices] = oracle_annotation_labels

            # Save the updated labels
            np.save(f"{dataset_name}_{uncertainty}_balance-{class_aware}_budget{budget}_labels.npy", all_labels)

            # Labeling accuracy after the active learning round
            updated_unlabeled_labels = all_labels[unlabeled_indices]
            corrects = np.sum(updated_unlabeled_labels == unlabeled_ground_truth)
            new_labeling_acc = corrects / len(unlabeled_ground_truth)
            print(f"Labeling Accuracy after first AL round: {new_labeling_acc * 100:.2f}%")

    if hasattr(train_dataset, 'targets'):
        train_dataset.targets = all_labels
    elif hasattr(train_dataset, 'labels'):
        train_dataset.labels = all_labels

    # Handle embedding of test dataset if embed_input is True
    if embed_input:
        # Filenames for test embeddings and labels
        test_embeddings_fname = f"{dataset_name}_embeddings_test.pt"
        test_labels_fname = f"{dataset_name}_test_labels.pt"
        
        # Check if test embeddings already exist
        if os.path.exists(os.path.join(save_path, test_embeddings_fname)):
            test_embeddings = torch.load(os.path.join(save_path, test_embeddings_fname))
            test_labels = torch.load(os.path.join(save_path, test_labels_fname))
        else:
            # Create a subset of the test_dataset to encode all test samples
            test_subset = torch.utils.data.Subset(test_dataset_for_embeddings, list(range(len(test_dataset_for_embeddings))))
            test_embeddings, test_labels = get_embeddings(
                test_subset, model, device, 
                fname=test_embeddings_fname, 
                lname=test_labels_fname, 
                batch_size=batch_size, 
                save_path=save_path
            )
        test_labels = torch.from_numpy(test_labels).long()

    if initial_only:
        initial_data = torch.load(f"{dataset_name}_initial_data.pt")
        train_labels = torch.load(f"{dataset_name}_initial_labels.pt")
        initial_data = torch.stack(initial_data)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        train_dataset = CustomTensorDataset(initial_data, train_labels)
    elif initial_with_random:
        initial_with_random_data = torch.load(f"{dataset_name}_initial_with_random_data.pt")
        train_labels = torch.load(f"{dataset_name}_initial_with_random_labels.pt")
        initial_with_random_data = torch.stack(initial_with_random_data)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        train_dataset = CustomTensorDataset(initial_with_random_data, train_labels)

    # Return evaluation dataset if required
    if return_eval_ds:
        if embed_input:
            test_dataset_embeddings = CustomTensorDataset(test_embeddings, test_labels)
            eval_loader = DataLoader(test_dataset_embeddings, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)
            num_samples = len(test_dataset_embeddings)
        else:
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
        else:
            split_fn = get_split_fn(split)
            # Split data into client-specific subsets
            train_indices = split_fn(idxs=extract_labels(train_dataset), num_shards=num_clients,
                                    num_samples=len(train_dataset), num_classes=num_classes, seed=seed)[int(id)]
        data_ratio = len(train_indices) / len(train_dataset)

        if embed_input:
            if initial_only:
                # Load initial data
                train_embeddings = torch.load(f"{dataset_name}_initial_embeddings.pt")
            elif initial_with_random:
                train_embeddings = torch.load(f"{dataset_name}_initial_with_random_embeddings.pt")
            else:
                train_embeddings = torch.load(os.path.join(save_path, train_embeddings_fname)).float()
    
            all_labels = torch.tensor(extract_labels(train_dataset), dtype=torch.long)
            subset_embeddings = train_embeddings[train_indices]
            subset_labels = all_labels[train_indices]
            
            # Create EmbeddingDataset and Dataloader for the client's data
            train_dataset_embeddings = CustomTensorDataset(subset_embeddings, subset_labels)
            train_loader = DataLoader(train_dataset_embeddings, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            num_samples = len(train_indices)
        else:
            # Create a subset of the train dataset for the client
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            num_samples = len(train_indices)

        return train_loader, num_classes, num_samples, data_ratio
