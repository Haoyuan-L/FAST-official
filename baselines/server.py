import flwr as fl
from flwr.common import Metrics
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from collections import OrderedDict
import torch
import time
import os
import logging
import json

from utils import set_seed, create_dir_if_not_exists

# Set up logging
def setup_logger(args):
    """Setup logger for the experiment."""
    log_dir = f"{args.save_dir}/{args.dataset}/{args.al_method}_alpha{args.alpha}"
    create_dir_if_not_exists(log_dir)
    log_file = f"{log_dir}/experiment.log"
    
    # Create a logger
    logger = logging.getLogger("FedAL")
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Compute weighted average of metrics."""
    # Calculate the weighted average
    weighted_metrics = {}
    
    if not metrics:
        return {}
    
    # Sum up weighted metrics and weights for each key
    for num_examples, metric_dict in metrics:
        for key, value in metric_dict.items():
            if key not in weighted_metrics:
                weighted_metrics[key] = [0, 0]  # [weighted_sum, total_weight]
            weighted_metrics[key][0] += value * num_examples
            weighted_metrics[key][1] += num_examples
    
    # Compute the weighted average for each key
    return {key: values[0] / values[1] if values[1] > 0 else 0 for key, values in weighted_metrics.items()}

class FedAvgServer:
    def __init__(self, args):
        self.args = args
        
        # Setting random seed
        set_seed(self.args.seed)
        
        # Creating directories for results
        self.result_dir = f"{self.args.save_dir}/{self.args.dataset}/{self.args.al_method}_alpha{self.args.alpha}"
        create_dir_if_not_exists(self.result_dir)
        
        # Setup logger
        self.logger = setup_logger(args)
        self.logger.info(f"Experiment started: {args.dataset}, {args.al_method}, alpha={args.alpha}")
        self.logger.info(f"Configuration: {vars(args)}")
        
        # Store metrics
        self.metrics = {
            "rounds": [],
            "accuracy": [],
            "loss": [],
            "time_per_round": [],
            "client_data": {}
        }
        
        # Store labeled and unlabeled data indices for each client
        self.client_labeled_dict = {}
        self.client_unlabeled_dict = {}
    
    def fit_config(self, server_round: int):
        """Return training configuration."""
        config = {
            "lr": self.args.lr,
            "momentum": self.args.momentum,
            "weight_decay": self.args.weight_decay,
            "local_epochs": self.args.local_epochs,
            "server_round": server_round,
        }
        return config
    
    def evaluate_config(self, server_round: int):
        """Return evaluation configuration."""
        config = {
            "server_round": server_round,
        }
        return config
    
    def get_fl_strategy(self):
        """Return the FL strategy."""
        if self.args.fl_strategy == 'fedavg':
            return fl.server.strategy.FedAvg(
                fraction_fit=1.0,  # Sample 100% of clients for training
                fraction_evaluate=1.0,  # Sample 100% of clients for evaluation
                min_fit_clients=self.args.num_clients,  # Number of clients needed for training
                min_evaluate_clients=self.args.num_clients,  # Number of clients needed for evaluation
                min_available_clients=self.args.num_clients,  # Minimum number of available clients
                on_fit_config_fn=self.fit_config,  # Function to configure training
                on_evaluate_config_fn=self.evaluate_config,  # Function to configure evaluation
                evaluate_metrics_aggregation_fn=weighted_average,  # Function to aggregate metrics
                initial_parameters=None,  # Initial parameters
            )
        elif self.args.fl_strategy == 'fedprox':
            # FedProx is similar to FedAvg but adds a proximal term
            # Since Flower doesn't have a built-in FedProx, we use FedAvg
            # and add the proximal term in the client's optimization
            return fl.server.strategy.FedAvg(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=self.args.num_clients,
                min_evaluate_clients=self.args.num_clients,
                min_available_clients=self.args.num_clients,
                on_fit_config_fn=self.fit_config,
                on_evaluate_config_fn=self.evaluate_config,
                evaluate_metrics_aggregation_fn=weighted_average,
                initial_parameters=None,
            )
        else:
            raise ValueError(f"Unsupported FL strategy: {self.args.fl_strategy}")
    
    def save_metrics(self, round_num, accuracy, loss, round_time=None, client_data=None):
        """Save metrics for the current round."""
        self.metrics["rounds"].append(round_num)
        self.metrics["accuracy"].append(accuracy)
        self.metrics["loss"].append(loss)
        
        if round_time is not None:
            self.metrics["time_per_round"].append(round_time)
        
        if client_data:
            for client_id, data in client_data.items():
                if client_id not in self.metrics["client_data"]:
                    self.metrics["client_data"][client_id] = {"labeled_size": [], "queried_size": []}
                self.metrics["client_data"][client_id]["labeled_size"].append(data["labeled_size"])
                self.metrics["client_data"][client_id]["queried_size"].append(data["queried_size"])
        
        # Save to file
        torch.save(self.metrics, f"{self.result_dir}/metrics.pt")
        
        # Also save as JSON for easier inspection
        metrics_json = {
            "rounds": self.metrics["rounds"],
            "accuracy": [float(acc) for acc in self.metrics["accuracy"]],
            "loss": [float(loss) for loss in self.metrics["loss"]],
            "time_per_round": self.metrics["time_per_round"],
            "client_data": self.metrics["client_data"]
        }
        
        with open(f"{self.result_dir}/metrics.json", 'w') as f:
            json.dump(metrics_json, f, indent=4)
        
        # Log metrics
        self.logger.info(f"Round {round_num}: Accuracy = {accuracy:.4f}, Loss = {loss:.4f}")
        if round_time is not None:
            self.logger.info(f"Round {round_num} Time: {round_time:.2f} seconds")
        
        # Print metrics if it's a log interval
        if round_num % self.args.log_interval == 0:
            print(f"Round {round_num}: Accuracy = {accuracy:.4f}, Loss = {loss:.4f}")
            if round_time is not None:
                print(f"Round {round_num} Time: {round_time:.2f} seconds")
