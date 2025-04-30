import os
import argparse
import numpy as np
import torch
import flwr as fl
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import random
import time

from utils import set_seed, load_config, create_dir_if_not_exists, dir_partition, shard_partition
from data import get_dataset
from models import get_model
from client import FedAvgClient
from server import FedAvgServer

def parse_args():
    parser = argparse.ArgumentParser(description='Federated Active Learning with LoGo')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    return parser.parse_args()

class GlobalModelKeeper:
    """Class to keep track of the global model between AL rounds"""
    def __init__(self, model_fn):
        self.model = model_fn()
        self.parameters = None
    
    def update_parameters(self, parameters):
        """Update model with new parameters"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        self.parameters = parameters
    
    def get_parameters(self):
        """Get current model parameters"""
        if self.parameters is None:
            self.parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        return self.parameters

class CustomFedAvgStrategy(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy that provides access to aggregated parameters"""
    def __init__(self, global_model_keeper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model_keeper = global_model_keeper
        
    def aggregate_fit(self, *args, **kwargs):
        """Aggregate parameters and update the global model keeper"""
        result = super().aggregate_fit(*args, **kwargs)
        if result is not None:
            parameters_aggregated, _ = result
            # Extract parameters from FedAvg result
            numpy_params = fl.common.parameters_to_ndarrays(parameters_aggregated)
            # Update the global model keeper
            self.global_model_keeper.update_parameters(numpy_params)
        return result

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration from yaml file
    config = load_config(args.config)
    
    # Convert config to a namespace object for easy access
    args_dict = argparse.Namespace(**config)
    
    # Set device
    args_dict.device = torch.device(args_dict.device if torch.cuda.is_available() else "cpu")
    
    # Set random seed
    set_seed(args_dict.seed)
    
    # Create result directory
    create_dir_if_not_exists(args_dict.save_dir)
    
    # Load datasets
    dataset_train, dataset_query, dataset_test = get_dataset(args_dict)
    
    # Partition dataset among clients
    if args_dict.partition == 'dir':
        # Use Dirichlet distribution for non-IID data
        client_data_dict = dir_partition(dataset_train, args_dict.num_clients, args_dict.alpha, equal_samples=True)
    elif args_dict.partition == 'shard':
        # Use shard partitioning for non-IID data
        client_data_dict = shard_partition(dataset_train, args_dict.num_clients, args_dict.num_classes_per_user)
    else:
        raise ValueError(f"Unsupported partition method: {args_dict.partition}")
    
    # Verify equal distribution
    for cid in range(args_dict.num_clients):
        print(f"Client {cid} data size: {len(client_data_dict[cid])}")
    
    # Initialize unlabeled and labeled data indices for each client
    client_unlabeled_dict = {}
    client_labeled_dict = {}
    
    # Generate all possible client IDs beforehand
    all_client_ids = [str(i) for i in range(args_dict.num_clients)]
    
    for cid in all_client_ids:
        # Get the corresponding integer client ID for the data dictionary
        int_cid = int(cid)
        if int_cid in client_data_dict:
            indices = client_data_dict[int_cid]
            
            # Initially, all data is unlabeled
            client_unlabeled_dict[cid] = set(indices)
            client_labeled_dict[cid] = set()
            
            # Randomly select initial labeled data
            if args_dict.initial_budget > 0:
                initial_labeled = random.sample(indices, min(args_dict.initial_budget_size, len(indices)))
                client_labeled_dict[cid].update(initial_labeled)
                client_unlabeled_dict[cid].difference_update(initial_labeled)
        else:
            # Create empty sets if client ID doesn't exist in data dictionary
            client_unlabeled_dict[cid] = set()
            client_labeled_dict[cid] = set()
            print(f"Warning: Client {cid} not found in client_data_dict.")
    
    # Store data dictionaries in args
    args_dict.client_unlabeled_dict = client_unlabeled_dict
    args_dict.client_labeled_dict = client_labeled_dict
    
    # Add model function to args
    args_dict.get_model_fn = lambda: get_model(args_dict)
    
    # Initialize server
    server = FedAvgServer(args_dict)
    
    # Create global model keeper to preserve model state between rounds
    global_model_keeper = GlobalModelKeeper(args_dict.get_model_fn)
    
    # Define client function
    def client_fn(cid):
        # Ensure cid is a string
        str_cid = str(cid)
        
        # If client ID doesn't exist, create it
        if str_cid not in args_dict.client_unlabeled_dict:
            print(f"Warning: Client {str_cid} not found in client_unlabeled_dict. Creating new entry.")
            args_dict.client_unlabeled_dict[str_cid] = set()
            args_dict.client_labeled_dict[str_cid] = set()
        
        return FedAvgClient(
            cid=str_cid,
            dataset_train=dataset_train,
            dataset_query=dataset_query,
            dataset_test=dataset_test,
            args=args_dict
        )
    
    # Run active learning process
    current_ratio = args_dict.initial_budget
    active_learning_round = 0
    
    while current_ratio <= args_dict.end_ratio:
        server.logger.info(f"[Active Learning Round] - Current labeled ratio: {current_ratio:.3f}")
        print(f"\n[Active Learning Round] - Current labeled ratio: {current_ratio:.3f}")
        
        # Update active learning round
        active_learning_round += 1
        
        # Start timer for this round
        start_time = time.time()
        
        # Configure strategy with current global model
        fl_strategy = CustomFedAvgStrategy(
            global_model_keeper=global_model_keeper,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=args_dict.num_clients,
            min_evaluate_clients=args_dict.num_clients,
            min_available_clients=args_dict.num_clients,
            on_fit_config_fn=server.fit_config,
            on_evaluate_config_fn=server.evaluate_config,
            evaluate_metrics_aggregation_fn=server.weighted_average,
            initial_parameters=fl.common.ndarrays_to_parameters(global_model_keeper.get_parameters()),
        )
        
        # Run federated learning for multiple rounds
        server.logger.info(f"Starting federated training for {args_dict.num_rounds} rounds")
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=args_dict.num_clients,
            config=fl.server.ServerConfig(num_rounds=args_dict.num_rounds, round_timeout=1500),
            strategy=fl_strategy,
            client_resources={"num_cpus": 2, "num_gpus": 0.2},
        )
        
        # Extract training metrics
        try:
            if history.metrics_distributed and 'accuracy' in history.metrics_distributed:
                acc_values = history.metrics_distributed['accuracy']
                if acc_values and len(acc_values) > 0:
                    training_accuracy = acc_values[-1][1]
                    server.logger.info(f"Training accuracy from distributed metrics: {training_accuracy:.4f}")
        except Exception as e:
            server.logger.error(f"Error extracting distributed accuracy: {e}")
        
        # Evaluate the global model
        global_model = args_dict.get_model_fn().to(args_dict.device)
        global_params = global_model_keeper.get_parameters()
        
        # Set parameters for the global model
        params_dict = zip(global_model.state_dict().keys(), global_params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        global_model.load_state_dict(state_dict, strict=True)
        
        # Evaluate the global model
        global_model.eval()
        test_loader = torch.utils.data.DataLoader(
            dataset_test, 
            batch_size=args_dict.test_batch_size, 
            shuffle=False
        )
        
        correct = 0
        total_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(args_dict.device), target.to(args_dict.device)
                output, _ = global_model(data)
                loss = torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
                total_loss += loss
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss = total_loss / len(test_loader.dataset)
        test_accuracy = correct / len(test_loader.dataset)
        
        server.logger.info(f"Global model evaluation: Accuracy={test_accuracy:.4f}, Loss={test_loss:.4f}")
        
        # Calculate the time taken for this round
        end_time = time.time()
        round_time = end_time - start_time
        server.logger.info(f"Round time: {round_time:.2f} seconds")
        
        # Save metrics
        server.save_metrics(active_learning_round, test_accuracy, test_loss, round_time)
        
        # Query new samples for each client
        client_data = {}
        server.logger.info("Starting active learning query phase")
        
        for cid in all_client_ids:
            client = client_fn(cid)
            # Set the global model parameters for active learning
            client.set_parameters(global_model_keeper.get_parameters())
            
            initial_labeled_size = len(client.labeled_indices)
            queried_indices = client.query_samples()
            
            # Update server's data dictionaries
            server.client_labeled_dict[cid] = client.labeled_indices
            server.client_unlabeled_dict[cid] = client.unlabeled_indices
            
            # Store client data for metrics
            client_data[cid] = {
                "labeled_size": len(client.labeled_indices),
                "queried_size": len(queried_indices)
            }
            
            log_msg = f"Client {cid}: Labeled={len(client.labeled_indices)}, " + \
                      f"Unlabeled={len(client.unlabeled_indices)}, " + \
                      f"Queried={len(queried_indices)}"
            server.logger.info(log_msg)
            print(log_msg)
        
        # Update args with the updated data dictionaries
        args_dict.client_labeled_dict = server.client_labeled_dict
        args_dict.client_unlabeled_dict = server.client_unlabeled_dict
        
        # Update current ratio
        current_ratio += args_dict.query_budget
    
    server.logger.info("\nFederated Active Learning completed!")
    server.logger.info(f"Final test accuracy: {test_accuracy:.4f}")
    server.logger.info(f"Results saved to {server.result_dir}")
    
    print("\nFederated Active Learning completed!")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Results saved to {server.result_dir}")

if __name__ == "__main__":
    main()
