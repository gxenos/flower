"""fedsim: A Flower Baseline."""

import pickle
from flwr.server.strategy import FedAvg
from flwr.common.logger import log
from logging import INFO

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from collections import defaultdict

from functools import partial, reduce
from typing import Union

from flwr.common import FitRes, NDArray, NDArrays, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy


class FedSim(FedAvg):
    """FedSim strategy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_clusters = 9

    def aggregate_fit(self, rnd, results, failures):
        """Aggregate method for FedSim."""
        X = []

        for _, returns in enumerate(results):
            if 'serialized_gradients' in returns[1].metrics:
                gradients = pickle.loads(returns[1].metrics['serialized_gradients'])
                X.append(gradients)

        X = np.array(X)

        pca = PCA(n_components=0.95, svd_solver= 'full')
        X_reduced = pca.fit_transform(X)

        km = KMeans(n_clusters=self.num_clusters)
        km.fit(X_reduced)

        clusters = defaultdict(list)
        for label, value in zip(km.labels_, results, strict=True):
            clusters[label].append(value)

        cluster_results = []

        for cluster_id, items in clusters.items():
            aggregated_ndarrays = aggregate_inplace(items)
            cluster_results.append(aggregated_ndarrays)


        aggregated_ndarrays = aggregate_unweighted(cluster_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated


def aggregate_inplace(results: list[tuple[ClientProxy, FitRes]]) -> NDArrays:
    """Compute in-place weighted average."""
    # Count total examples
    num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)

    # Compute scaling factors for each result
    scaling_factors = np.asarray(
        [fit_res.num_examples / num_examples_total for _, fit_res in results]
    )

    def _try_inplace(
        x: NDArray, y: Union[NDArray, np.float64], np_binary_op: np.ufunc
    ) -> NDArray:
        return (  # type: ignore[no-any-return]
            np_binary_op(x, y, out=x)
            if np.can_cast(y, x.dtype, casting="same_kind")
            else np_binary_op(x, np.array(y, x.dtype), out=x)
        )


    params = [
        _try_inplace(x, scaling_factors[0], np_binary_op=np.multiply)
        for x in parameters_to_ndarrays(results[0][1].parameters)
    ]

    for i, (_, fit_res) in enumerate(results[1:], start=1):
        res = (
            _try_inplace(x, scaling_factors[i], np_binary_op=np.multiply)
            for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [
            reduce(partial(_try_inplace, np_binary_op=np.add), layer_updates)
            for layer_updates in zip(params, res)
        ]

    return params


def aggregate_unweighted(weights_list: list[NDArrays]) -> NDArrays:
    """Compute simple average of model weights (no sample weighting)."""
    num_models = len(weights_list)

    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_models
        for layer_updates in zip(*weights_list)
    ]
    return weights_prime

