import flwr as fl
import torch
import collections
from typing import Callable, Dict, List, Optional, Tuple
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
import numpy as np

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        model_loader: Callable,
        data_loader: Callable,
        num_classes: int,
        device: str = 'cuda',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.best_accuracy = 0.0  # Initialize best accuracy
        self.best_parameters = None  # Store the best parameters
        self.device = device
        self.model_loader = model_loader
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Load evaluation data
        self.eval_data, _, _ = self.data_loader()

        # Initialize the model
        self.model = self.model_loader(num_classes=self.num_classes).to(self.device)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Aggregate the client updates
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Set the model parameters to the aggregated parameters
            self.set_model_params(aggregated_parameters)

            # Evaluate the new global model
            loss, accuracy = self.evaluate_model()
            print(f"Round {server_round} - Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

            # Check if the new accuracy is better than the best so far
            if accuracy > self.best_accuracy:
                print(f"New best model found at round {server_round} with accuracy {accuracy * 100:.2f}%")
                self.best_accuracy = accuracy
                self.best_parameters = aggregated_parameters
            else:
                print(f"Accuracy did not improve at round {server_round}. Keeping the previous best model.")
                # Revert to the best parameters
                aggregated_parameters = self.best_parameters
                # Also set the model parameters to the best parameters
                if self.best_parameters is not None:
                    self.set_model_params(self.best_parameters)

        return aggregated_parameters, metrics_aggregated

    def set_model_params(self, parameters: Parameters):
        # Convert parameters to model's state_dict format and load them
        params_dict = zip(
            self.model.state_dict().keys(),
            parameters_to_tensors(parameters),
        )
        state_dict = collections.OrderedDict({k: v.to(self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def evaluate_model(self) -> Tuple[float, float]:
        # Evaluate the model on the server's evaluation data
        self.model.eval()
        loss_total = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.eval_data:
                x, y = x.to(self.device), y.to(self.device).long()
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                loss_total += loss.item() * x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        average_loss = loss_total / total
        accuracy = correct / total

        return average_loss, accuracy

def parameters_to_tensors(parameters: Parameters):
    return [torch.tensor(np_param, dtype=torch.float32) for np_param in parameters_to_ndarrays(parameters)]