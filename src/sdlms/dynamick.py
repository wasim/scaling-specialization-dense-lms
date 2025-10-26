from __future__ import annotations

import torch

def topk_gate(x: torch.Tensor, k: float | int):
    if isinstance(k, float):
        k_count = max(1, int(x.size(-1) * k))
    else:
        k_count = int(k)

    _, idx = torch.topk(x.abs(), k_count, dim=-1)
    mask = torch.zeros_like(x).scatter_(-1, idx, 1.0)
    return mask
