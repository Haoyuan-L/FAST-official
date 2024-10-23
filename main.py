import os
import ray
import sys
import time
import yaml
import argparse
import flwr as fl
import random
import numpy as np
import torch
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
from utils import grab_gpu
os.environ['CUDA_VISIBLE_DEVICES'] = grab_gpu()
warnings.simplefilter("ignore")

from utils import *
from args import args_parser
import network
from data import get_data
from client import Client
from server import Server

def run_experiment(num_rounds=100, num_clients=10, participation=1.0, data_split='iid', max_parallel_executions=5, active_oracle=True,
                   timeout=1500, init_model=None, dataset="cifar10", skewness_alpha=None, class_aware=False, uncertainty="norm", 
                   model="resnet18", encoder="SigLIP", budget=0.1, fl_method="fedavg"):
    
    embed_input = False
    if model == "resnet18":
        network_fn = network.get_resnet18_network
    elif model == "cnn4":
        network_fn = network.get_cnn4_network
    elif model == "linear":
        network_fn = network.get_linear_network
        embed_input = True
    else:
        raise NotImplementedError(f"Model '{model}' is not supported.") 
    
    def create_client(cid):
        time.sleep(int(cid) * 0.75)
        return Client(cid=int(cid), dataset=dataset, model_loader=network_fn, embed_input=embed_input, fl_method=fl_method,
                      data_loader=lambda: get_data(dataset_name=dataset, id=cid, num_clients=num_clients, embed_input=embed_input, encoder=encoder, active_oracle=active_oracle,
                                                   split=data_split, alpha=skewness_alpha, class_aware=class_aware, uncertainty=uncertainty, budget=budget))

    def create_server(init_model=None):
        return Server(num_rounds=num_rounds, num_clients=num_clients, embed_input=embed_input, fl_method=fl_method,
                      participation=participation, model_loader=network_fn, dataset=dataset, 
                      data_loader=lambda: get_data(dataset_name=dataset, embed_input=embed_input, encoder=encoder, active_oracle=active_oracle, split=data_split, 
                                                   alpha=skewness_alpha, return_eval_ds=True, budget=budget), init_model=init_model)
    ray.shutdown()
    ray.init()
    total_resources = ray.cluster_resources()
    print("Total resources:", total_resources)
    # allocate the client resources
    total_cpus = total_resources.get('CPU', 1.0)
    total_gpus = total_resources.get('GPU', 0.0)
    max_concurrent_clients = int(min(max_parallel_executions, num_clients))
    print(f"Max concurrent clients: {max_concurrent_clients}")

    # Calculate CPU per client
    cpu_per_client = total_cpus / max_concurrent_clients
    min_cpu_per_client = 0.1
    cpu_per_client = max(cpu_per_client, min_cpu_per_client)
    if cpu_per_client > 1.0:
        cpu_per_client = int(cpu_per_client)
    else:
        cpu_per_client = round(cpu_per_client, 2)

    # Calculate GPU per client
    if total_gpus > 0:
        gpu_per_client = total_gpus / max_concurrent_clients
        min_gpu_per_client = 0.1
        gpu_per_client = max(gpu_per_client, min_gpu_per_client)
        if gpu_per_client > 1.0:
            gpu_per_client = int(gpu_per_client)
        else:
            gpu_per_client = round(gpu_per_client, 2)  # Round to two decimal places
    else:
        gpu_per_client = 0.0  # No GPUs available

    client_resources = {
    "num_cpus": cpu_per_client,
    "num_gpus": gpu_per_client
    }

    server = create_server()
    history = fl.simulation.start_simulation(client_fn=create_client, server=server, num_clients=num_clients,
                                             ray_init_args={"ignore_reinit_error": True, "num_cpus": max_concurrent_clients},
                                             config=fl.server.ServerConfig(num_rounds=num_rounds, round_timeout=timeout), client_resources=client_resources)
    return history


def run_with_different_configs(yaml_config_file):

    # Load configurations from YAML file
    with open(yaml_config_file, 'r') as file:
        configs = yaml.safe_load(file)

    # Loop through each experiment configuration
    for config in configs['experiments']:
        # Set the seed for reproducibility
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        # Run the experiment with current configuration
        history = run_experiment(num_rounds=config["num_rounds"], num_clients=config["num_clients"], model=config["model"],
                                 data_split=config["data_split"], participation=config["participation"],
                                 max_parallel_executions=config["max_parallel_executions"], dataset=config["dataset"],
                                 skewness_alpha=config["skewness_alpha"], class_aware=config["class_aware"], uncertainty=config["uncertainty"],
                                 active_oracle=config["active_oracle"], encoder=config["encoder"], budget=config["budget"], fl_method=config["fl_method"])

        # Log the results of the experiment
        fname = f'{config["model"]}_{config["dataset"]}_{config["data_split"]}_{config["uncertainty"]}_class_aware-{str(config["class_aware"])}_Oracle-{str(config["active_oracle"])}_{config["encoder"]}_budget-{config["budget"]}_{config["fl_method"]}.log'
        log_results(history, config, fname)


def log_results(history, config, fname):
    with open(fname, "a") as file:
        file.write(f"Config: {config}\n")
        file.write(f"History: {history}\n")
        file.write("---------------------------\n")


if __name__ == "__main__":
    args = args_parser()
    yaml_config_file = "config.yaml"
    run_with_different_configs(yaml_config_file)
