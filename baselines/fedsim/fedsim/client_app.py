"""fedsim: A Flower Baseline."""

import torch
import pickle
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from fedsim.dataset import load_data_femnist
from fedsim.model import FEMNIST_MLR_Net, get_weights, set_weights, test, train, get_gradients

from flwr.common.logger import log
from logging import INFO


class FlowerClient(NumPyClient):
    """A class defining the client."""

    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        """Traim model using this client's data."""
        gradients = get_gradients(self.net, self.trainloader, self.device)
        serialized_gradients = pickle.dumps(gradients)

        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss, "serialized_gradients": serialized_gradients},
        )

    def evaluate(self, parameters, config):
        """Evaluate model using this client's data."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy, "loss": loss}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Load model and data
    net = FEMNIST_MLR_Net()
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    trainloader, valloader = load_data_femnist(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
