from __future__ import annotations

import numpy as np
import networkx as nx

def activation_sparsity(acts: np.ndarray, thresh: float = 0.0) -> dict:
    mask = acts > thresh
    frac = mask.mean()
    x = acts.clip(min=0)
    pr = (x.sum() ** 2) / (np.square(x).sum() + 1e-12)
    return {"frac": float(frac), "pr": float(pr)}

def specialization_index(task_probs: np.ndarray) -> float:
    eps = 1e-12
    p = task_probs.clip(eps, 1.0)
    H = -(p * np.log(p)).sum(axis=1)
    K = task_probs.shape[1]
    return float(1.0 - (H / np.log(K)).mean())

def coactivation_modularity(acts: np.ndarray, k_top: int = 200) -> float:
    A = (acts > 0).astype(np.float32)
    p_i = A.mean(0)
    Pij = (A.T @ A) / max(1, A.shape[0])
    with np.errstate(divide="ignore"):
        PMI = np.log((Pij + 1e-9) / (p_i[:, None] * p_i[None, :] + 1e-9))
    np.fill_diagonal(PMI, 0.0)
    idx = np.dstack(np.unravel_index(np.argsort(-PMI.ravel()), PMI.shape))[0][:k_top]
    G = nx.Graph()
    for i, j in idx:
        w = float(PMI[i, j])
        if w > 0:
            G.add_edge(int(i), int(j), weight=w)
    if G.number_of_edges() == 0:
        return 0.0
    communities = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
    return float(nx.algorithms.community.quality.modularity(G, communities, weight="weight"))
