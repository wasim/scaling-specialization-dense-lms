from __future__ import annotations

import csv
from argparse import Namespace
from pathlib import Path

from sdlms.cli import sparsity


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
