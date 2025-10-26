from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sdlms.cli import sae_train


def _make_activation_file(tmp_path: Path, rows: int = 64, dim: int = 16) -> Path:
    data = np.random.default_rng(seed=0).standard_normal(size=(rows, dim)).astype('float32')
    path = tmp_path / 'acts.npy'
    np.save(path, data)
    return path


def test_sae_train_writes_artifacts(tmp_path: Path) -> None:
    act_path = _make_activation_file(tmp_path)
    out_dir = tmp_path / 'out'

    argv = [
        '--activations',
        str(act_path),
        '--out',
        str(out_dir),
        '--epochs',
        '1',
        '--batch-size',
        '16',
        '--latent-size',
        '8',
    ]

    sae_train.main(argv)

    weights = out_dir / 'sae.pt'
    metrics = out_dir / 'metrics.csv'
    meta = out_dir / 'meta.json'

    assert weights.exists()
    assert metrics.exists()
    assert meta.exists()

    meta_obj = json.loads(meta.read_text())
    assert meta_obj['latent_dim'] == 8
    assert meta_obj['input_dim'] == 16
