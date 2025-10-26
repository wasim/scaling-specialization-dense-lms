from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

from sdlms.cli import sparsity
from sdlms.probe_tasks import load_probe_tasks


def test_sparsity_cli_writes_outputs(tmp_path: Path) -> None:
    args = Namespace(
        model='hf-internal-testing/tiny-random-gpt2',
        prompt='Testing sparsity',
        dataset='wikitext',
        split='validation',
        num_docs=1,
        max_tokens=64,
        threshold=0.0,
        output_dir=tmp_path,
        layers=None,
        dtype='float32',
        probe_manifest=None,
        task_ids=None,
    )

    csv_path, meta_path = sparsity.run(args)

    assert csv_path.exists(), 'CSV output file missing'
    assert meta_path.exists(), 'Meta output file missing'

    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows, 'CSV output empty'
    for row in rows:
        assert {'layer', 'frac', 'pr'} <= row.keys()


def test_sparsity_cli_probe_manifest(tmp_path: Path) -> None:
    manifest = load_probe_tasks(Path('data/probe_tasks.jsonl'))
    task = manifest['ioi_minimal']

    args = Namespace(
        model='hf-internal-testing/tiny-random-gpt2',
        prompt=None,
        dataset='wikitext',
        split='validation',
        num_docs=1,
        max_tokens=64,
        threshold=0.0,
        output_dir=tmp_path,
        layers=None,
        dtype='float32',
        probe_manifest=None,
        task_ids=None,
    )

    csv_path, meta_path = sparsity.run(args, probe_task=task)

    assert csv_path.exists()
    assert meta_path.exists()

    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows, 'CSV output empty'
    assert all(row.get('task_id') == task.task_id for row in rows)

    with meta_path.open() as handle:
        meta_line = handle.readline()
    meta = json.loads(meta_line)
    assert meta.get('task', {}).get('task_id') == task.task_id
