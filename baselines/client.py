import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import NDArrays, Scalar

from utils import DatasetSplit, get_al_strategy

class FedAvgClient(fl.client.NumPyClient):
    def __init__(self, cid, dataset_train, dataset_query, dataset_test, args):
        self.cid = cid
        self.args = args
        self.device = args.device
        
        # Datasets
        self.dataset_train = dataset_train
        self.dataset_query = dataset_query
        self.dataset_test = dataset_test
        
        # Data indices - ensure cid is a string
        self.unlabeled_indices = set(args.client_unlabeled_dict[cid])
        self.labeled_indices = set(args.client_labeled_dict[cid])
        
        # Create model
        self.model = args.get_model_fn().to(self.device)
        self.local_model = copy.deepcopy(self.model)  # Local-only model for active learning
        
        # Keep best model
        self.best_model = None
        self.best_acc = 0.0
        
        # Active learning strategy
        self.al_strategy = get_al_strategy(args.al_method)
    
    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train model on local dataset."""
        # Update model parameters
        self.set_parameters(parameters)
        
        # Check if there are labeled samples
        if len(self.labeled_indices) == 0:
            print(f"Warning: Client {self.cid} has no labeled data. Skipping training.")
            return self.get_parameters(config={}), 0, {}
        
        # Train the model
        self.train(self.labeled_indices)
        
        # Train local-only model if it's needed for active learning
        if self.args.query_model_mode in ["local_only", "both"]:
            # Make a deep copy to avoid affecting the global model
            self.local_model = copy.deepcopy(self.model)
            # Train local-only model
            self.train_local_only(self.labeled_indices)
        
        # Return updated model parameters and statistics
        return self.get_parameters(config={}), len(self.labeled_indices), {}

    def evaluate(self, parameters, config):
        """Evaluate model on local test dataset."""
        # Update model parameters
        self.set_parameters(parameters)
        
        # Evaluate the model
        acc, loss = self.test(self.dataset_test)
        
        # Store best model
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_model = copy.deepcopy(self.model.state_dict())
        
        # Return statistics
        return float(loss), len(self.dataset_test), {"accuracy": float(acc)}
    
    def train(self, indices):
        """Train model on local dataset."""
        if not indices:
            return  # Skip training if no labeled data
            
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        
        train_dataset = DatasetSplit(self.dataset_train, indices)
        trainloader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        
        for epoch in range(self.args.local_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (data, target, _) in enumerate(trainloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output, _ = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            epoch_loss = total_loss / len(trainloader)
            epoch_acc = 100. * correct / total
            print(f"Client {self.cid} - Epoch {epoch+1}/{self.args.local_epochs} - Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
    
    def train_local_only(self, indices):
        """Train local-only model on local dataset."""
        if not indices:
            return  # Skip training if no labeled data
            
        self.local_model.train()
        optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        
        train_dataset = DatasetSplit(self.dataset_train, indices)
        trainloader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        
        for epoch in range(self.args.local_epochs):
            total_loss = 0.0
            for batch_idx, (data, target, _) in enumerate(trainloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output, _ = self.local_model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
    
    def test(self, dataset):
        """Test model on dataset."""
        self.model.eval()
        test_loader = DataLoader(dataset, batch_size=self.args.test_batch_size, shuffle=False)
        
        loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        
        return accuracy, loss
    
    def query_samples(self):
        """Query samples from unlabeled data."""
        if len(self.unlabeled_indices) == 0:
            return []
        
        # Calculate the actual query budget for this client
        # Each client should get an equal share of the total query budget
        per_client_query_budget = int(self.args.query_budget_per_round / self.args.num_clients)
        print(f"Client {self.cid} - Per client query budget: {per_client_query_budget}")
        
        # Choose the appropriate models based on query_model_mode
        net_global = self.model if self.args.query_model_mode in ["global", "both"] else None
        net_local = self.local_model if self.args.query_model_mode in ["local_only", "both"] else None
        
        # Query samples
        try:
            queried_indices = self.al_strategy.query(
                user_idx=self.cid,  # Pass cid as is (already a string)
                label_idxs=self.labeled_indices,
                unlabel_idxs=self.unlabeled_indices,
                dataset=self.dataset_query,
                net_global=net_global,
                net_local=net_local,
                args=self.args,
                query_budget=per_client_query_budget  # Pass the client-specific budget
            )
            
            # Update labeled and unlabeled sets
            self.labeled_indices.update(queried_indices)
            self.unlabeled_indices.difference_update(queried_indices)
            
            return queried_indices
            
        except Exception as e:
            print(f"Error in query_samples for client {self.cid}: {e}")
            # Fallback to random sampling in case of error
            if len(self.unlabeled_indices) <= per_client_query_budget:
                queried_indices = list(self.unlabeled_indices)
            else:
                queried_indices = list(np.random.choice(
                    list(self.unlabeled_indices), 
                    per_client_query_budget, 
                    replace=False
                ))
            
            # Update labeled and unlabeled sets
            self.labeled_indices.update(queried_indices)
            self.unlabeled_indices.difference_update(queried_indices)
            
            return queried_indices
