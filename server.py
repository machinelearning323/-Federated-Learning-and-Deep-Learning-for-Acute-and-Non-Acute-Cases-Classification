import flwr as fl
import sys
import numpy as np
print(sys.argv)
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            weights_ndarrays= fl.common.parameters_to_ndarrays(aggregated_weights[0])
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"C:\\Users\\Abdussalam\\Desktop\\final_package\\final_package\\Weight_FL_server\\round-{rnd}-weights.npz", *weights_ndarrays)
        return aggregated_weights

# Create strategy and run server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = 'localhost:'+str(8080) , 
        config=fl.server.ServerConfig(num_rounds=100) ,
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy
)