"""fedsim: A Flower Baseline."""

from collections import OrderedDict
import torch
from torch import nn


class FEMNIST_MLR_Net(nn.Module):
    """Mutlinomial Logistic Regression."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28*28, 10)

    def forward(self, x):
        """Do forward."""
        x = self.linear(torch.flatten(x, 1))
        return x


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.003)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"]
            labels = batch["character"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["character"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    """Extract model parameters as numpy arrays from state_dict."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Apply parameters to an existing model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_gradients(net, trainloader, device):
    """Compute gradients of the loss w.r.t. model parameters on the whole training set."""
    net.to(device)
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    # Zero gradients
    for param in net.parameters():
        if param.grad is not None:
            param.grad.zero_()
    # Accumulate gradients over the whole training set
    for batch in trainloader:
        images = batch["image"].to(device)
        labels = batch["character"].to(device)
        net.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
    # Extract gradients as numpy arrays
    gradients = torch.cat([ p.grad.view(-1) for p in net.parameters() if p.grad is not None])
    return gradients
