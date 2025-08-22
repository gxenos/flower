"""fedsim: A Flower Baseline."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DistributionPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from flwr_datasets.preprocessor import Preprocessor
import numpy as np


FDS = None  # Cache FederatedDataset

class FEMNISTFilter(Preprocessor):
    """A Preprocessor class that filter the FEMNIST data.
    It filters data with label 36 to 45 (lower case letters 'a'-'j')
    and reindexes the labels.
    """

    def __call__(self, dataset_dict):
        """Call function."""
        def keep_a_to_j(example):
            return example["character"] in range(36, 36 + 10)

        filtered = dataset_dict.filter(keep_a_to_j)

        # Reindex labels 36..45 -> 0..9
        def reindex(example):
            example["character"] = example["character"] - 36
            return example

        return filtered.map(reindex)

def load_data_femnist(partition_id: int, num_partitions: int):
    """Load partition FEMNIST data."""
    # Only initialize `FederatedDataset` once
    global FDS  # pylint: disable=global-statement
    if FDS is None:
        num_partitions = 200
        num_unique_labels_per_partition = 3
        num_unique_labels = 10
        preassigned_num_samples_per_label = 5


        rng = np.random.default_rng(42)
        distribution_array = rng.lognormal(
            4.0,
            1.0,
            (num_partitions * num_unique_labels_per_partition),
        )
        distribution_array = distribution_array.reshape(
            (num_unique_labels, -1)
        )
        partitioner = DistributionPartitioner(
            distribution_array=distribution_array,
            num_partitions=num_partitions,
            num_unique_labels_per_partition=num_unique_labels_per_partition,
            partition_by="character",
            preassigned_num_samples_per_label=preassigned_num_samples_per_label,
        )
        FDS = FederatedDataset(
            dataset="flwrlabs/femnist",
            preprocessor=FEMNISTFilter(),
            partitioners={"train": partitioner},
        )
    partition = FDS.load_partition(partition_id)


    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    MNIST_TRANSFORMS = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [MNIST_TRANSFORMS(img) for img in batch["image"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=10, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=10)
    return trainloader, testloader
