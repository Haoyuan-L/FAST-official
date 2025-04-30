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
        
        # Training history
        self.train_history = {"loss": [], "accuracy": []}
        
        # Active learning strategy
        self.al_strategy = get_al_strategy(args.al_method)
        
        print(f"Client {self.cid} initialized with {len(self.labeled_indices)} labeled and {len(self.unlabeled_indices)} unlabeled samples")
    
    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        # When we update the global model, also initialize the local model from it
        self.local_model = copy.deepcopy(self.model)
    
    def fit(self, parameters, config):
        """Train model on local dataset."""
        # Update model parameters
        self.set_parameters(parameters)
        
        # Check if there are labeled samples
        if len(self.labeled_indices) == 0:
            print(f"Warning: Client {self.cid} has no labeled data. Skipping training.")
            return self.get_parameters(config={}), 0, {}
        
        # Train the model
        train_stats = self.train(self.labeled_indices)
        
        # Train local-only model if it's needed for active learning
        if self.args.query_model_mode in ["local_only", "both"]:
            # Already copied the model in set_parameters
            # Train local-only model with more aggressive local updates
            local_stats = self.train_local_only(self.labeled_indices)
        
        # Return updated model parameters and statistics
        return self.get_parameters(config={}), len(self.labeled_indices), train_stats

    def evaluate(self, parameters, config):
        """Evaluate model on local test dataset."""
        # Update model parameters
        self.set_parameters(parameters)
        
        # Evaluate the model
        acc, loss = self.test(self.dataset_test)
        
        # Return statistics
        return float(loss), len(self.dataset_test), {"accuracy": float(acc)}
    
    def train(self, indices):
        """Train model on local dataset."""
        if not indices:
            return {"loss": 0.0, "accuracy": 0.0}
            
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        
        train_dataset = DatasetSplit(self.dataset_train, indices)
        trainloader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        
        epoch_losses = []
        epoch_accs = []
        
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
            epoch_losses.append(epoch_loss)
            epoch_accs.append(epoch_acc / 100.0)  # Store as decimal
            
            print(f"Client {self.cid} - Epoch {epoch+1}/{self.args.local_epochs} - Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        
        # Store training history
        self.train_history["loss"].extend(epoch_losses)
        self.train_history["accuracy"].extend(epoch_accs)
        
        return {"loss": epoch_losses[-1], "accuracy": epoch_accs[-1]}
    
    def train_local_only(self, indices):
        """Train local-only model on local dataset."""
        if not indices:
            return {"loss": 0.0, "accuracy": 0.0}
            
        self.local_model.train()
        optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        
        train_dataset = DatasetSplit(self.dataset_train, indices)
        trainloader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        
        epoch_losses = []
        epoch_accs = []
        
        # Train for more epochs locally to increase divergence for LoGo
        local_epochs = self.args.local_epochs * 2  # Double the local epochs
        
        for epoch in range(local_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_idx, (data, target, _) in enumerate(trainloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output, _ = self.local_model(data)
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
            epoch_losses.append(epoch_loss)
            epoch_accs.append(epoch_acc / 100.0)  # Store as decimal
            
            if epoch % 5 == 0:  # Print only every 5 epochs to reduce output
                print(f"Client {self.cid} - Local Model - Epoch {epoch+1}/{local_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        
        return {"loss": epoch_losses[-1], "accuracy": epoch_accs[-1]}
    
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
