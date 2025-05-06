import time
import torch
import yaml
import ray
import json
import os
import logging
from client import Client
from server import Server
import network
from utils import *
import flwr as fl

def run_experiment(num_rounds=100, num_clients=10, participation=1.0, data_split='iid', max_parallel_executions=5, active_oracle=True,
                   timeout=1500, init_model=None, dataset="cifar10", skewness_alpha=None, class_aware=False, uncertainty="norm", local_epochs=5,
                   model="resnet18", encoder="SigLIP", budget=0.1, fl_method="fedavg", initial_only=False, initial_with_random=False, seed=42):
    
    # Start timing the experiment
    experiment_start_time = time.time()
    
    embed_input = False
    if model == "resnet18":
        network_fn = network.get_resnet18_network
    elif model == "resnet8":
        network_fn = network.get_resnet8_network
    elif model == "cnn4":
        network_fn = network.get_cnn4_network
    elif model == "linear":
        network_fn = network.get_linear_network
        embed_input = True
    else:
        raise NotImplementedError(f"Model '{model}' is not supported.") 
    
    def create_client(cid):
        time.sleep(int(cid) * 0.75)
        return Client(cid=int(cid), dataset=dataset, model_loader=network_fn, embed_input=embed_input, fl_method=fl_method, seed=seed,
                      num_clients=num_clients, encoder=encoder, active_oracle=active_oracle, data_split=data_split, skewness_alpha=skewness_alpha, 
                      class_aware=class_aware, uncertainty=uncertainty, budget=budget, initial_only=initial_only, initial_with_random=initial_with_random)

    def create_server(init_model=None):
        return Server(num_rounds=num_rounds, num_clients=num_clients, embed_input=embed_input, fl_method=fl_method, participation=participation, seed=seed,
                      model_loader=network_fn, dataset=dataset, encoder=encoder, active_oracle=active_oracle, data_split=data_split, skewness_alpha=skewness_alpha, 
                      uncertainty=uncertainty, class_aware=class_aware, return_eval_ds=True, budget=budget, initial_only=initial_only, initial_with_random=initial_with_random, 
                      local_epochs=local_epochs, init_model=init_model)
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
    
    # Run the simulation
    history = fl.simulation.start_simulation(
        client_fn=create_client, 
        server=server, 
        num_clients=num_clients,
        ray_init_args={"ignore_reinit_error": True, "num_cpus": max_concurrent_clients},
        config=fl.server.ServerConfig(num_rounds=num_rounds, round_timeout=timeout), 
        client_resources=client_resources
    )
    
    # Calculate total experiment time
    total_experiment_time = time.time() - experiment_start_time
    print(f"Total experiment wall-clock time: {total_experiment_time:.2f} seconds")
    
    # Collect and organize communication and computation metrics
    metrics = {
        "experiment_info": {
            "model": model,
            "dataset": dataset,
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "fl_method": fl_method,
            "data_split": data_split
        },
        "communication_overhead": {
            "server_to_client_bytes": server.measurements["comm_overhead"]["server_to_client_bytes"],
            "client_to_server_bytes": server.measurements["comm_overhead"]["client_to_server_bytes"],
            "total_bytes": server.measurements["comm_overhead"]["total_bytes"],
            "server_to_client_mb": server.measurements["comm_overhead"]["server_to_client_bytes"] / (1024 * 1024),
            "client_to_server_mb": server.measurements["comm_overhead"]["client_to_server_bytes"] / (1024 * 1024),
            "total_mb": server.measurements["comm_overhead"]["total_bytes"] / (1024 * 1024)
        },
        "computation_time": {
            "server_time": server.measurements["computation_time"]["server_time"],
            "round_times": server.measurements["computation_time"]["rounds_time"],
            "total_experiment_time": total_experiment_time
        }
    }
    
    # Add metrics to history object to return
    history.metrics = metrics
    
    return history


def run_with_different_configs(yaml_config_file):
    # Create a directory for storing metrics
    metrics_dir = os.path.join(os.getcwd(), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Load configurations from YAML file
    with open(yaml_config_file, 'r') as file:
        configs = yaml.safe_load(file)

    # Loop through each experiment configuration
    for config in configs['experiments']:
        # Set the seed for reproducibility
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])
        
        print(f"Running experiment with configuration: {config}")
        
        # Run the experiment with current configuration
        history = run_experiment(num_rounds=config["num_rounds"], num_clients=config["num_clients"], model=config["model"],
                                 data_split=config["data_split"], participation=config["participation"], local_epochs=config["local_epochs"],
                                 max_parallel_executions=config["max_parallel_executions"], dataset=config["dataset"],
                                 skewness_alpha=config["skewness_alpha"], class_aware=config["class_aware"], uncertainty=config["uncertainty"],
                                 active_oracle=config["active_oracle"], encoder=config["encoder"], budget=config["budget"], fl_method=config["fl_method"],
                                 initial_only=config["initial_only"], initial_with_random=config["initial_with_random"], seed=config["seed"])

        # Generate filename for results
        if config["initial_only"]:
            fname = f'{config["model"]}_{config["dataset"]}_{config["data_split"]}_initialOnly_Oracle-{str(config["active_oracle"])}_{config["encoder"]}_budget-{config["budget"]}_{config["fl_method"]}_local_epochs{config["local_epochs"]}_clients-{config["num_clients"]}.log'
        elif config["initial_with_random"]:
            fname = f'{config["model"]}_{config["dataset"]}_{config["data_split"]}_initialRandom_Oracle-{str(config["active_oracle"])}_{config["encoder"]}_budget-{config["budget"]}_{config["fl_method"]}_local_epochs{config["local_epochs"]}_clients-{config["num_clients"]}.log'
        else:
            fname = f'{config["model"]}_{config["dataset"]}_{config["data_split"]}_{config["uncertainty"]}_Oracle-{str(config["active_oracle"])}_{config["encoder"]}_budget-{config["budget"]}_{config["fl_method"]}_local_epochs{config["local_epochs"]}_clients-{config["num_clients"]}.log'
        
        # Save metrics to a separate JSON file
        metrics_fname = os.path.join(metrics_dir, f'metrics_{os.path.splitext(fname)[0]}.json')
        with open(metrics_fname, 'w') as f:
            json.dump(history.metrics, f, indent=4)
        
        # Log the results of the experiment
        log_results(history, config, fname)


def log_results(history, config, fname):
    with open(fname, "a") as file:
        file.write(f"Config: {config}\n")
        file.write(f"History: {history}\n")
        
        # Add communication and computation metrics
        if hasattr(history, 'metrics'):
            file.write(f"Communication Overhead:\n")
            file.write(f"  - Server to Clients: {history.metrics['communication_overhead']['server_to_client_mb']:.2f} MB\n")
            file.write(f"  - Clients to Server: {history.metrics['communication_overhead']['client_to_server_mb']:.2f} MB\n")
            file.write(f"  - Total: {history.metrics['communication_overhead']['total_mb']:.2f} MB\n")
            
            file.write(f"Computation Time:\n")
            file.write(f"  - Server Time: {history.metrics['computation_time']['server_time']:.2f} seconds\n")
            file.write(f"  - Total Experiment Time: {history.metrics['computation_time']['total_experiment_time']:.2f} seconds\n")
            
            # Calculate average round time
            if history.metrics['computation_time']['round_times']:
                avg_round_time = sum(history.metrics['computation_time']['round_times']) / len(history.metrics['computation_time']['round_times'])
                file.write(f"  - Average Round Time: {avg_round_time:.2f} seconds\n")
            
            # Try to read and log embedding times
            embedding_log_file = f"{config['dataset']}_{config['encoder']}_embedding_times.log"
            if os.path.exists(embedding_log_file):
                file.write(f"Embedding Generation Times (from {embedding_log_file}):\n")
                with open(embedding_log_file, 'r') as emb_file:
                    for line in emb_file:
                        if "embedding" in line.lower() and "time" in line.lower():
                            file.write(f"  - {line.strip()}\n")
        
        file.write("---------------------------\n")

if __name__ == "__main__":
    yaml_config_file = "config.yaml"
    run_with_different_configs(yaml_config_file)
