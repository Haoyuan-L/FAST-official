import flwr as fl
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, communication_tracker=None, **kwargs):
        super().__init__(**kwargs)
        self.communication_tracker = communication_tracker
        self.current_round = 0
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, Dict]]:
        self.current_round = server_round
        # Start timing for this round
        self.round_start_time = time.time()
        self.round_bytes_sent = 0
        self.round_bytes_received = 0
        
        # Get client configuration and clients
        config = self.on_fit_config_fn(server_round)
        clients = client_manager.sample(
            num_clients=self.min_fit_clients, 
            min_num_clients=self.min_fit_clients
        )
        
        # Track parameters sent to clients
        if self.communication_tracker:
            parameters_list = fl.common.parameters_to_ndarrays(parameters)
            bytes_sent_per_client = self.communication_tracker.track_parameters_sent(parameters_list, server_round)
            self.round_bytes_sent += bytes_sent_per_client * len(clients)
        
        return [(client, config) for client in clients]
    
    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Track parameters received from clients
        if self.communication_tracker:
            for _, fit_res in results:
                parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)
                bytes_received = self.communication_tracker.track_parameters_received(parameters, server_round)
                self.round_bytes_received += bytes_received
        
        # Calculate time taken for the round
        round_time = time.time() - self.round_start_time
        
        # Log communication stats for this round
        if self.communication_tracker:
            self.communication_tracker.log_round_communication(
                server_round, 
                self.round_bytes_sent, 
                self.round_bytes_received, 
                round_time
            )
        
        # Call the parent's aggregate_fit method
        return super().aggregate_fit(server_round, results, failures)

class CustomFedNova(fl.server.strategy.FedAvg):

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], fl.common.Scalar]:

        if not results:
            return None, {}

        # Compute tau_effective from summation of local client tau: Eqn-6: Section 4.1
        local_tau = [res.metrics["tau"] for _, res in results]
        tau_eff = np.sum(local_tau)

        aggregate_parameters = []

        for _client, res in results:
            params = parameters_to_ndarrays(res.parameters)
            # compute the scale by which to weight each client's gradient
            # res.metrics["local_norm"] contains total number of local update steps
            # for each client
            # res.metrics["ratio"] contains the ratio of client dataset size
            # Below corresponds to Eqn-6: Section 4.1
            scale = tau_eff / float(res.metrics["local_norm"])
            scale *= float(res.metrics["ratio"])

            aggregate_parameters.append((params, scale))

        # Aggregate all client parameters with a weighted average using the scale
        agg_cum_gradient = aggregate(aggregate_parameters)

        # Aggregate custom metrics if aggregation function was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return ndarrays_to_parameters(agg_cum_gradient), metrics_aggregated
