from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def list_layers(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    return [name for name, _ in model.named_modules() if "mlp" in name or "feed_forward" in name]

def collect_ffn_acts(
    model_name: str, texts: list[str], layer_names: list[str], max_tokens: int = 2048
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    hooks: list[torch.utils.hooks.RemovableHandle] = []
    buffers: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(module, _inp, out):
            buffers.setdefault(name, []).append(out.detach().to("cpu"))

        return hook

    for layer_name in layer_names:
        module = dict(model.named_modules())[layer_name]
        hooks.append(module.register_forward_hook(make_hook(layer_name)))

    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_tokens
            ).to(model.device)
            model(**encoded)

    for hook_handle in hooks:
        hook_handle.remove()

    for key, tensor_list in buffers.items():
        buffers[key] = torch.cat(tensor_list, dim=0).float().numpy()

    return buffers
