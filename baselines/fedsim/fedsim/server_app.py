"""fedsim: A Flower Baseline."""

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fedsim.strategy import FedSim
import json
import os

from fedsim.model import FEMNIST_MLR_Net, get_weights

RESULTS_PATH = "./results/"
RESULTS_FILE = "results.json"

def config_json_file(context: Context) -> None:
    """Initialize the json file and write the run configurations."""
    # Initialize the execution results directory.

    data = {
        "run_config": dict(context.run_config.items()),
        "round_res": [],
    }

    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    with open(os.path.join(RESULTS_PATH, RESULTS_FILE), "w+", encoding="UTF-8") as fout:
        json.dump(data, fout, indent=4)

def write_res(new_res: dict[str, float]) -> None:
    """Load the json file, append result and re-write json collection."""
    with open(os.path.join(RESULTS_PATH, RESULTS_FILE), encoding="UTF-8") as fin:
        data = json.load(fin)
    data["round_res"].append(new_res)

    # Write the updated data back to the JSON file
    with open(os.path.join(RESULTS_PATH, RESULTS_FILE), "w", encoding="UTF-8") as fout:
        json.dump(data, fout, indent=4)

# Define metric aggregation function
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Do weighted average of accuracy metric."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    val_losses = [num_examples * float(m["loss"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    average_acc = sum(accuracies) / sum(examples)
    average_val_loss = sum(val_losses) / sum(examples)
    # Aggregate and return custom metric (weighted average)
    write_res({"accuracy": average_acc, "loss": average_val_loss})

    return {"accuracy": average_acc , "test_loss": average_val_loss}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Read from config
    config_json_file(context)

    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(FEMNIST_MLR_Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedSim(
        fraction_fit=float(fraction_fit),
        fraction_evaluate=1.0,
        min_available_clients=20,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=int(num_rounds))

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
