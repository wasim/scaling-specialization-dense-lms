from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sdlms.activations import collect_ffn_acts
from sdlms.probe_tasks import load_probe_tasks, select_probe_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Capture FFN activations for probe tasks.')
    parser.add_argument('--model', required=True, help='Model identifier from the Hugging Face Hub')
    parser.add_argument(
        '--probe-manifest',
        type=Path,
        default=Path('data/probe_tasks.jsonl'),
        help='JSONL manifest describing probe tasks (defaults to data/probe_tasks.jsonl)',
    )
    parser.add_argument(
        '--task-id',
        dest='task_ids',
        action='append',
        help=(
            'Specific task_id entries to capture (can be passed multiple times). '
            'Defaults to all tasks.'
        ),
    )
    parser.add_argument(
        '--layers',
        nargs='*',
        default=None,
        help='Layer names to hook. Defaults to model.layers.10.mlp if omitted.',
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=2048,
        help='Fallback max token length when the task does not specify one.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('artifacts') / 'captures',
        help='Directory to store activation dumps and metadata.',
    )
    parser.add_argument(
        '--no-save',
        dest='save_activations',
        action='store_false',
        help='Do not persist activation arrays to disk (metadata is still written).',
    )
    parser.set_defaults(save_activations=True)
    return parser.parse_args()


def sanitize_layer(layer_name: str) -> str:
    return layer_name.replace('.', '_').replace('/', '-').replace(':', '-')


def main() -> None:
    args = parse_args()
    manifest = load_probe_tasks(args.probe_manifest)
    tasks = select_probe_tasks(manifest, args.task_ids)
    if not tasks:
        raise SystemExit('No tasks resolved from manifest; nothing to capture.')

    layer_names = args.layers or ['model.layers.10.mlp']

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_dir = args.output_dir / f'{timestamp}_{args.model.replace("/", "-")}'
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / 'meta.jsonl'

    for task in tasks:
        texts = list(task.iter_texts())
        if not texts:
            print(f'[capture] task {task.task_id} produced no texts; skipping.')
            continue

        effective_tokens = task.max_tokens or args.max_tokens
        buffers = collect_ffn_acts(args.model, texts, layer_names, max_tokens=effective_tokens)

        task_dir = run_dir / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        activation_paths: dict[str, str | None] = {}
        for layer_name, array in buffers.items():
            activation_paths[layer_name] = None
            if args.save_activations:
                filename = sanitize_layer(layer_name) + '.npy'
                out_path = task_dir / filename
                np.save(out_path, array)
                activation_paths[layer_name] = str(out_path.resolve())
            print(f'[capture] {task.task_id} | {layer_name}: {array.shape}')

        record = {
            'model': args.model,
            'task': task.metadata(),
            'layer_names': list(buffers.keys()),
            'max_tokens': effective_tokens,
            'texts_processed': len(texts),
            'activation_paths': activation_paths,
            'saved': args.save_activations,
        }

        with meta_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')

    print(f'[capture] wrote run metadata to {meta_path}')


if __name__ == '__main__':
    main()
