from __future__ import annotations

import networkx as nx
import numpy as np


def activation_sparsity(acts: np.ndarray, thresh: float = 0.0) -> dict:
    mask = acts > thresh
    frac = mask.mean()
    x = acts.clip(min=0)
    pr = (x.sum() ** 2) / (np.square(x).sum() + 1e-12)
    return {'frac': float(frac), 'pr': float(pr)}


def specialization_index(task_probs: np.ndarray) -> float:
    eps = 1e-12
    probs = task_probs.clip(eps, 1.0)
    entropy = -(probs * np.log(probs)).sum(axis=1)
    num_tasks = task_probs.shape[1]
    return float(1.0 - (entropy / np.log(num_tasks)).mean())


def coactivation_modularity(acts: np.ndarray, k_top: int = 200) -> float:
    binary_acts = (acts > 0).astype(np.float32)
    marginal = binary_acts.mean(0)
    pairwise = (binary_acts.T @ binary_acts) / max(1, binary_acts.shape[0])
    with np.errstate(divide='ignore'):
        pmi = np.log((pairwise + 1e-9) / (marginal[:, None] * marginal[None, :] + 1e-9))
    np.fill_diagonal(pmi, 0.0)
    indices = np.dstack(np.unravel_index(np.argsort(-pmi.ravel()), pmi.shape))[0][:k_top]
    graph = nx.Graph()
    for i, j in indices:
        w = float(pmi[i, j])
        if w > 0:
            graph.add_edge(int(i), int(j), weight=w)
    if graph.number_of_edges() == 0:
        return 0.0
    communities = nx.algorithms.community.greedy_modularity_communities(graph, weight='weight')
    return float(nx.algorithms.community.quality.modularity(graph, communities, weight='weight'))
