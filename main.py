import os
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
import data
from client import Client
from server import Server

def run_experiment(num_rounds=120, num_clients=20, participation=1.0, data_split='iid', max_parallel_executions=5,
                   timeout=1500, init_model=None, dataset="cifar10"):

    def create_client(cid):
        time.sleep(int(cid) * 0.75)
        return Client(int(cid), num_clients=num_clients, model_loader=network.get_network,
                      data_loader=lambda: get_data(dataset_name=dataset, id=cid, num_clients=num_clients, split_fn=get_split_fn(data_split)))

    def create_server(init_model=None):
        return Server(num_rounds=num_rounds, num_clients=num_clients, participation=participation,
                      model_loader=network.get_network, data_loader=lambda: get_data(dataset_name=dataset, split_fn=get_split_fn(data_split)), 
                      init_model=init_model)

    server = create_server()
    history = fl.simulation.start_simulation(client_fn=create_client, server=server, num_clients=num_clients,
                                             ray_init_args={"ignore_reinit_error": True, "num_cpus": int(min(max_parallel_executions, num_clients))},
                                             config=fl.server.ServerConfig(num_rounds=num_rounds, round_timeout=timeout))
    return history


def run_with_different_configs(yaml_config_file):
    random.seed(42)
    np.random.seed(42)

    # Load configurations from YAML file
    with open(yaml_config_file, 'r') as file:
        configs = yaml.safe_load(file)

    # Loop through each experiment configuration
    for config in configs['experiments']:
        # Run the experiment with current configuration
        history = run_experiment(num_rounds=config["num_rounds"], num_clients=config["num_clients"],
                                 data_split=config["data_split"], participation=config["participation"],
                                 max_parallel_executions=config["max_parallel_executions"])

        # Log the results of the experiment
        log_results(history, config)


def log_results(history, config):
    with open("exp_log.txt", "a") as file:
        file.write(f"Config: {config}\n")
        file.write(f"History: {history}\n")
        file.write("---------------------------\n")


if __name__ == "__main__":
    args = args_parser()
    yaml_config_file = "config.yaml"
    run_with_different_configs(yaml_config_file)