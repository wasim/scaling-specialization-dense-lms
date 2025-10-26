from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type == 'cuda':
        return torch.float16
    if device.type == 'mps':
        return torch.float16
    return torch.float32


def list_layers(model_name: str) -> list[str]:
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    return [name for name, _ in model.named_modules() if 'mlp' in name or 'feed_forward' in name]


def collect_ffn_acts(
    model_name: str,
    texts: list[str],
    layer_names: list[str],
    max_tokens: int = 2048,
) -> dict[str, np.ndarray]:
    device = infer_device()
    dtype = resolve_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()

    hooks: list[torch.utils.hooks.RemovableHandle] = []
    buffers: dict[str, list[Tensor]] = {}

    def make_hook(name: str) -> torch.utils.hooks.RemovableHandle:
        def hook(module: torch.nn.Module, _inp: tuple[Tensor, ...], out: Tensor) -> None:
            tensor = out.detach().to('cpu')
            hidden = tensor.shape[-1]
            flattened = tensor.reshape(-1, hidden)
            buffers.setdefault(name, []).append(flattened)

        return module.register_forward_hook(hook)

    for layer_name in layer_names:
        module = dict(model.named_modules())[layer_name]
        hooks.append(make_hook(layer_name))

    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text, return_tensors='pt', truncation=True, max_length=max_tokens
            ).to(device)
            model(**encoded)

    for hook_handle in hooks:
        hook_handle.remove()

    array_buffers: dict[str, np.ndarray] = {}
    for key, tensor_list in buffers.items():
        array_buffers[key] = torch.cat(tensor_list, dim=0).float().numpy()

    return array_buffers
