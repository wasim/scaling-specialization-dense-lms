from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from sdlms.probe_tasks import ProbeTask, load_probe_tasks, select_probe_tasks


@dataclass
class RunningActivationStats:
    """Lightweight accumulator for activation sparsity metrics."""

    active: float = 0.0
    total: float = 0.0
    positive_sum: float = 0.0
    positive_sum_sq: float = 0.0

    def update(self, activations: Tensor, threshold: float) -> None:
        tensor = activations.detach()
        mask = tensor > threshold
        self.active += mask.sum().item()
        self.total += float(mask.numel())
        clipped = torch.clamp(tensor, min=0).to(dtype=torch.float32)
        self.positive_sum += clipped.sum().item()
        self.positive_sum_sq += (clipped * clipped).sum().item()

    def finalize(self) -> dict[str, float]:
        if self.total == 0:
            return {'frac': 0.0, 'pr': 0.0}
        frac = self.active / self.total
        pr = (
            0.0
            if self.positive_sum_sq == 0.0
            else (self.positive_sum**2) / (self.positive_sum_sq + 1e-12)
        )
        return {'frac': float(frac), 'pr': float(pr)}


def slugify(model_name: str) -> str:
    return model_name.replace('/', '-')


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def iter_dataset_texts(dataset_name: str, split: str, limit: int) -> Iterator[str]:
    dataset = load_dataset(dataset_name, split=split)
    for idx, sample in enumerate(dataset):
        if 0 <= limit <= idx:
            break
        yield sample.get('text', str(sample))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Estimate activation sparsity metrics for FFNs in a causal LM',
    )
    parser.add_argument(
        '--model',
        default='EleutherAI/pythia-70m-deduped',
        help='Model identifier from the HF Hub',
    )
    parser.add_argument(
        '--prompt',
        help='Optional prompt text; if provided, bypass the dataset loader',
    )
    parser.add_argument(
        '--dataset',
        default='wikitext',
        help='Dataset name for context tokens',
    )
    parser.add_argument(
        '--split',
        default='validation',
        help='Dataset split to sample',
    )
    parser.add_argument(
        '--num-docs',
        type=int,
        default=4,
        help='Number of documents to sample (use -1 for entire split)',
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=2048,
        help='Truncate sequences to this many tokens',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='Activation threshold for sparsity (values > threshold count as active)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('artifacts') / 'sparsity',
        help='Directory to write CSV/metadata',
    )
    parser.add_argument(
        '--layers',
        nargs='*',
        default=None,
        help='Specific layer names to probe (defaults to all FFN layers)',
    )
    parser.add_argument(
        '--dtype',
        choices=['auto', 'float16', 'bfloat16', 'float32'],
        default='auto',
        help='Computation dtype',
    )
    parser.add_argument(
        '--probe-manifest',
        type=Path,
        help=(
            'JSONL manifest describing probe tasks '
            '(overrides prompt/dataset arguments when provided)'
        ),
    )
    parser.add_argument(
        '--task-id',
        dest='task_ids',
        action='append',
        help='Task IDs from the manifest to execute (repeatable). Defaults to all tasks.',
    )
    return parser.parse_args()


def resolve_dtype(arg: str, device: torch.device) -> torch.dtype:
    if arg == 'float16':
        return torch.float16
    if arg == 'bfloat16':
        return torch.bfloat16
    if arg == 'float32':
        return torch.float32
    if device.type == 'cuda':
        return torch.float16
    if device.type == 'mps':
        return torch.float16
    return torch.float32


def build_output_paths(
    base_dir: Path, model_name: str, task: ProbeTask | None = None
) -> tuple[Path, Path]:
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    slug = slugify(model_name)
    if task is not None:
        slug += f'_{slugify(task.task_id)}'
    base_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_dir / f'{timestamp}_{slug}_sparsity.csv'
    meta_path = base_dir / f'{timestamp}_{slug}_meta.jsonl'
    return csv_path, meta_path


def tokenize_batches(
    tokenizer: AutoTokenizer,
    texts: Iterable[str],
    max_tokens: int,
    device: torch.device,
) -> Iterable[dict[str, torch.Tensor]]:
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_tokens,
        )
        yield {key: value.to(device) for key, value in encoded.items()}


def collect_layers(model: AutoModelForCausalLM, requested: list[str] | None) -> list[str]:
    if requested is not None:
        return requested
    return [
        name
        for name, _ in model.named_modules()
        if name.endswith('mlp') or 'feed_forward' in name.lower()
    ]


def register_hooks(
    model: AutoModelForCausalLM,
    layer_names: list[str],
    stats: dict[str, RunningActivationStats],
    threshold: float,
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    modules = dict(model.named_modules())

    for layer_name in layer_names:
        module = modules[layer_name]

        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[Tensor, ...],
            outputs: Tensor,
            *,
            name: str = layer_name,
        ) -> None:
            stats[name].update(outputs, threshold)

        handles.append(module.register_forward_hook(hook))

    return handles


def resolve_text_stream(args: argparse.Namespace, probe_task: ProbeTask | None) -> Iterator[str]:
    if probe_task is not None:
        yield from probe_task.iter_texts()
        return
    if args.prompt is not None:
        count = 1 if args.num_docs == -1 else max(1, args.num_docs)
        for _ in range(count):
            yield args.prompt
        return
    yield from iter_dataset_texts(args.dataset, args.split, args.num_docs)


def run(args: argparse.Namespace, *, probe_task: ProbeTask | None = None) -> tuple[Path, Path]:
    device = infer_device()
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    dtype = resolve_dtype(args.dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()

    layer_names = collect_layers(model, args.layers)

    if not layer_names:
        raise ValueError('No layers selected for sparsity measurement')

    stats = {layer: RunningActivationStats() for layer in layer_names}
    handles = register_hooks(model, layer_names, stats, args.threshold)

    total_tokens = 0
    effective_max_tokens = (
        probe_task.max_tokens
        if probe_task and probe_task.max_tokens is not None
        else args.max_tokens
    )
    text_stream = resolve_text_stream(args, probe_task)
    batches = tokenize_batches(tokenizer, text_stream, effective_max_tokens, device)
    texts_processed = 0

    with torch.no_grad():
        for batch in batches:
            texts_processed += 1
            total_tokens += batch['input_ids'].numel()
            model(**batch)

    for handle in handles:
        handle.remove()

    csv_path, meta_path = build_output_paths(args.output_dir, args.model, probe_task)

    fieldnames = ['layer', 'frac', 'pr']
    if probe_task is not None:
        fieldnames.insert(0, 'task_id')

    with csv_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for layer_name, running in stats.items():
            result = running.finalize()
            row = {'layer': layer_name, **result}
            if probe_task is not None:
                row['task_id'] = probe_task.task_id
            writer.writerow(row)

    meta = {
        'model': args.model,
        'threshold': args.threshold,
        'dtype': str(dtype),
        'device': device.type,
        'tokens_processed': total_tokens,
        'layers': layer_names,
        'max_tokens': effective_max_tokens,
        'texts_processed': texts_processed,
    }
    if probe_task is None:
        meta.update(
            {
                'dataset': args.dataset,
                'split': args.split,
                'num_docs': args.num_docs,
                'max_tokens_arg': args.max_tokens,
            }
        )
    else:
        meta['task'] = probe_task.metadata()

    meta_path.write_text(json.dumps(meta) + '\n', encoding='utf-8')

    return csv_path, meta_path


def main() -> None:
    args = parse_args()
    if args.probe_manifest:
        manifest = load_probe_tasks(args.probe_manifest)
        tasks = select_probe_tasks(manifest, args.task_ids)
        if not tasks:
            raise SystemExit('No tasks selected from probe manifest.')
        for task in tasks:
            csv_path, meta_path = run(args, probe_task=task)
            print(f'[sparsity] task {task.task_id} wrote metrics to {csv_path}')
            print(f'[sparsity] task {task.task_id} wrote metadata to {meta_path}')
    else:
        csv_path, meta_path = run(args)
        print(f'[sparsity] wrote metrics to {csv_path}')
        print(f'[sparsity] wrote metadata to {meta_path}')


if __name__ == '__main__':
    main()
