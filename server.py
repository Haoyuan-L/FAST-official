import logging
import flwr as fl
import torch
from data import *
import collections
import torchmetrics
from strategy import *
from utils import get_learning_rate
import time  # Import for timing measurements
import sys   # Import for calculating size

class Server(fl.server.Server):

    def __init__(self, dataset, model_loader, encoder, active_oracle, data_split, skewness_alpha, class_aware, uncertainty, budget,
                 return_eval_ds, initial_only, initial_with_random, num_rounds, num_clients=10, embed_input=False, participation=1.0, 
                 init_model=None, log_level=logging.INFO, initial_lr=1e-3, decay_factor=0.1, num_decays=3, fl_method="fedavg", seed=42, local_epochs=5):
        
        # Initialize measurements dictionary for communication and computation metrics
        self.measurements = {
            "comm_overhead": {
                "server_to_client_bytes": 0,
                "client_to_server_bytes": 0,
                "total_bytes": 0
            },
            "computation_time": {
                "server_time": 0,
                "rounds_time": []
            }
        }
        
        self.fl_method = fl_method
        self.num_rounds = num_rounds
        self.data, self.num_classes, self.num_samples = get_data(dataset_name=dataset, num_clients=num_clients, embed_input=embed_input, 
                                                           encoder=encoder, active_oracle=active_oracle,split=data_split, seed=seed,
                                                             alpha=skewness_alpha, class_aware=class_aware, uncertainty=uncertainty, budget=budget,
                                                             return_eval_ds=return_eval_ds, initial_only=initial_only, initial_with_random=initial_with_random)
        
        self.embed_input = embed_input
        if self.embed_input:
            try:
                first_batch = next(iter(self.data))
                first_embedding = first_batch[0]
                if isinstance(first_embedding, torch.Tensor):
                    emb_dim = first_embedding.shape[-1]
                    self.input_shape = (emb_dim,)
                else:
                    raise ValueError("Expected embedding to be a torch.Tensor")
            except StopIteration:
                raise ValueError("DataLoader is empty. Cannot determine embedding dimension.")
        else:
            self.input_shape = self.get_dataset_config(dataset)
        self.model_loader = model_loader
        self.init_model = init_model
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.num_decays = num_decays
        self.clients_config = {"epochs":local_epochs, "lr":initial_lr}
        self.num_clients = num_clients
        self.participation = participation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._client_manager = fl.server.client_manager.SimpleClientManager()
        self.max_workers = None
        self.set_strategy(self)
        self.embed_input = embed_input
        logging.getLogger("flower").setLevel(log_level)

    def set_max_workers(self, *args, **kwargs):
        return super(Server, self).set_max_workers(*args, **kwargs)

    def set_strategy(self, *_):
        if self.fl_method.lower() == "fedavg":
            self.strategy = fl.server.strategy.FedAvg(
                min_available_clients=self.num_clients, fraction_fit=self.participation,
                min_fit_clients=int(self.participation*self.num_clients), fraction_evaluate=0.0,
                min_evaluate_clients=0, evaluate_fn=self.get_evaluation_fn(),
                on_fit_config_fn=self.get_client_config_fn(), initial_parameters=self.get_initial_parameters(),)
        elif self.fl_method.lower() == "fedprox":
            self.strategy = fl.server.strategy.FedProx(
            min_available_clients=self.num_clients, fraction_fit=self.participation,
            min_fit_clients=int(self.participation*self.num_clients), fraction_evaluate=0.0,
            min_evaluate_clients=0, evaluate_fn=self.get_evaluation_fn(),
            on_fit_config_fn=self.get_client_config_fn(), initial_parameters=self.get_initial_parameters(),
            proximal_mu=0.1,)
        elif self.fl_method.lower() == "fednova":
            self.strategy = CustomFedNova(
                min_available_clients=self.num_clients, fraction_fit=self.participation,
                min_fit_clients=int(self.participation*self.num_clients), fraction_evaluate=0.0,
                min_evaluate_clients=0, evaluate_fn=self.get_evaluation_fn(),
                on_fit_config_fn=self.get_client_config_fn(), initial_parameters=self.get_initial_parameters(),)

    def client_manager(self, *args, **kwargs):
        return super(Server, self).client_manager(*args, **kwargs)

    def get_parameters(self, config={}):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def get_dataset_config(self, dataset):
        if dataset.lower() in ["cifar10", "svhn", "cifar100"]:
            input_shape=(3, 32, 32)
        elif dataset.lower() in ["pathmnist", "dermamnist"]:
            input_shape=(3, 28, 28)
        elif dataset.lower() == "tiny-imagenet":
            input_shape = (3, 64, 64)
        else:
            raise NotImplementedError(f"Dataset '{dataset}' is not supported.")
        return input_shape

    def set_parameters(self, parameters, config):
        if not hasattr(self, 'model'):
            self.model = self.model_loader(input_shape=self.input_shape, num_classes=self.num_classes).to(self.device)
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_initial_parameters(self, *_):
        """ Get initial random model weights """
        if self.init_model is not None:
            self.init_weights = torch.load(self.init_model, map_location=self.device).state_dict()
        else:
            self.init_weights = [val.cpu().numpy() for _, val in self.model_loader(input_shape=self.input_shape, num_classes=self.num_classes).state_dict().items()]
        
        # Measure size of initial parameters sent to clients
        init_params = fl.common.ndarrays_to_parameters(self.init_weights)
        init_params_size = self._calculate_parameters_size(init_params)
        self.measurements["comm_overhead"]["server_to_client_bytes"] += init_params_size
        self.measurements["comm_overhead"]["total_bytes"] += init_params_size
        logging.info(f"Initial parameters size: {init_params_size / (1024 * 1024):.2f} MB")
        
        return init_params

    def get_evaluation_fn(self):
        def evaluation_fn(rnd, parameters, config):
            self.set_parameters(parameters, config)
            metrics = __class__.evaluate(model=self.model, ds=self.data, num_classes=self.num_classes)
            return metrics[0], {"accuracy":metrics[1]}
        return evaluation_fn

    def get_client_config_fn(self):
        """Define fit config function with dynamic learning rate based on round."""
        def config_fn(rnd):
            # Calculate the learning rate based on the current round
            current_lr = get_learning_rate(
                initial_lr=self.initial_lr,
                current_round=rnd,
                total_rounds=self.num_rounds,
                decay_factor=self.decay_factor,
                num_decays=self.num_decays
            )
            # Update the clients' configuration
            client_config = {
                "epochs": 5,
                "lr": current_lr,
                "round": rnd
            }
            logging.info(f"Round {rnd}: Setting client learning rate to {current_lr}")
            return client_config
        return config_fn

    # Override the fit method to measure time and data size
    def fit(self, num_rounds, timeout):
        """Run federated learning for a number of rounds."""
        total_start_time = time.time()
        
        # Use the original fit method
        history = super().fit(num_rounds, timeout)
        
        total_time = time.time() - total_start_time
        self.measurements["computation_time"]["server_time"] = total_time
        
        # Log final measurements
        logging.info(f"Total communication overhead: {self.measurements['comm_overhead']['total_bytes'] / (1024 * 1024):.2f} MB")
        logging.info(f"  - Server to clients: {self.measurements['comm_overhead']['server_to_client_bytes'] / (1024 * 1024):.2f} MB")
        logging.info(f"  - Clients to server: {self.measurements['comm_overhead']['client_to_server_bytes'] / (1024 * 1024):.2f} MB")
        logging.info(f"Total computation time: {total_time:.2f} seconds")
        
        return history
      
    # Method to measure parameters size
    def _calculate_parameters_size(self, parameters):
        """Calculate the size of parameters in bytes."""
        if not parameters.tensors:
            return 0
        
        # Get total size of all tensors
        total_bytes = 0
        for tensor in parameters.tensors:
            # Use sys.getsizeof to get memory footprint or compute based on ndarray size
            ndarray_size = sys.getsizeof(tensor)
            total_bytes += ndarray_size
            
        return total_bytes
    
    # Now use a hook-based approach to capture the data transmission instead of overriding fit_round
    def start_round(self, server_round: int) -> None:
        """Initialize a round on the server."""
        # Start timing the round
        round_start_time = time.time()
        
        # Call the parent class method
        super().start_round(server_round)
        
        # Store the start time for later use
        self._round_start_time = round_start_time
        
        # Log round start
        logging.info(f"Round {server_round} started")
        
    def end_round(self, server_round: int) -> None:
        """End a round on the server."""
        # Call the parent class method
        super().end_round(server_round)
        
        # Record round time
        if hasattr(self, '_round_start_time'):
            round_time = time.time() - self._round_start_time
            self.measurements["computation_time"]["rounds_time"].append(round_time)
            logging.info(f"Round {server_round} completed in {round_time:.2f} seconds")
        
        # Measure communication overhead
        # Assume all clients were selected in this round
        num_selected = int(self.participation * self.num_clients)
        
        # Assume each client has approximately the same model size
        if hasattr(self, 'model'):
            # Calculate model size once
            model_params = self.get_parameters()
            model_size = sum(param.nbytes for param in model_params)
            
            # Server to client communication (broadcast model)
            server_to_client = model_size * num_selected
            self.measurements["comm_overhead"]["server_to_client_bytes"] += server_to_client
            
            # Client to server communication (send updated models)
            client_to_server = model_size * num_selected
            self.measurements["comm_overhead"]["client_to_server_bytes"] += client_to_server
            
            # Total communication
            self.measurements["comm_overhead"]["total_bytes"] += server_to_client + client_to_server
            
            logging.info(f"Round {server_round} communication: {(server_to_client + client_to_server) / (1024 * 1024):.2f} MB")

    @staticmethod
    def evaluate(ds, model, num_classes, metrics=None, loss=torch.nn.CrossEntropyLoss(), verbose=False):
        device = next(model.parameters()).device
        if metrics is None:
            metrics = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
        model.eval()
        _loss = 0.0
        with torch.no_grad():
            for _, (x, y) in enumerate(ds):
                x, y = x.to(device), y.to(device).long().squeeze()
                preds = model(x)
                _loss += loss(preds, y).item()
                metrics(preds.max(1)[-1], y)
        _loss /= len(ds)
        acc = metrics.compute()
        if verbose:
            print(f"Loss: {_loss:.4f} - Accuracy: {100. * acc:.2f}%")
        return (_loss, acc)
