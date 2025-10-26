from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    activations: Sequence[Path]
    out_dir: Path
    latent_size: int | None
    expansion: float
    l1: float
    lr: float
    epochs: int
    batch_size: int
    val_fraction: float
    seed: int
    device: torch.device
    dtype: torch.dtype
    grad_clip: float | None


class ActivationDataset(Dataset[torch.Tensor]):
    """Memory-mapped view over one or more .npy activation shards."""

    def __init__(self, files: Sequence[Path], dtype: torch.dtype) -> None:
        self._arrays: list[np.memmap] = [np.load(str(path), mmap_mode='r') for path in files]
        if not self._arrays:
            msg = 'At least one activation file is required'
            raise ValueError(msg)
        self._lengths = [arr.shape[0] for arr in self._arrays]
        self._cumulative = np.cumsum([0] + self._lengths)
        self._dtype = dtype

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_idx = int(np.searchsorted(self._cumulative, index, side='right') - 1)
        offset = index - int(self._cumulative[shard_idx])
        row = np.asarray(self._arrays[shard_idx][offset]).copy()
        return torch.as_tensor(row, dtype=self._dtype)

    @property
    def feature_dim(self) -> int:
        return int(self._arrays[0].shape[1])


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(d_in, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_in, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.relu(self.encoder(x))
        recon = self.decoder(features)
        return recon, features


def infer_latent_size(d_in: int, latent_size: int | None, expansion: float) -> int:
    if latent_size is not None:
        return latent_size
    expanded = int(d_in * expansion)
    return max(d_in, expanded)


def split_indices(
    num_items: int, val_fraction: float, generator: torch.Generator
) -> tuple[np.ndarray, np.ndarray]:
    indices = torch.randperm(num_items, generator=generator).numpy()
    val_count = int(num_items * val_fraction)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    return train_idx, val_idx


def build_dataloaders(
    dataset: ActivationDataset,
    config: TrainConfig,
    generator: torch.Generator,
) -> tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
    train_idx, val_idx = split_indices(len(dataset), config.val_fraction, generator)
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx, generator=generator)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx, generator=generator)

    def collate(batch: Iterable[torch.Tensor]) -> torch.Tensor:
        return torch.stack(tuple(batch))

    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        collate_fn=collate,
    )
    return train_loader, val_loader


def evaluate(
    model: SparseAutoencoder,
    loader: DataLoader[torch.Tensor],
    criterion: nn.Module,
    config: TrainConfig,
    split: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_l1 = 0.0
    total_active = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(config.device)
            recon, features = model(batch)
            total_loss += criterion(recon, batch).item() * batch.size(0)
            total_l1 += features.abs().mean(dim=1).sum().item()
            total_active += (features > 0).float().mean(dim=1).sum().item()
            total_count += batch.size(0)
    denom = max(1, total_count)
    return {
        f'loss_{split}': total_loss / denom,
        f'l1_{split}': total_l1 / denom,
        f'active_frac_{split}': total_active / denom,
    }


def train(config: TrainConfig) -> dict[str, torch.Tensor | list[dict[str, float]]]:
    generator = torch.Generator().manual_seed(config.seed)
    dataset = ActivationDataset(config.activations, config.dtype)
    d_in = dataset.feature_dim
    d_hidden = infer_latent_size(d_in, config.latent_size, config.expansion)

    train_loader, val_loader = build_dataloaders(dataset, config, generator)

    model = SparseAutoencoder(d_in, d_hidden).to(device=config.device, dtype=config.dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history: list[dict[str, float]] = []
    criterion = nn.MSELoss()

    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            recon, features = model(batch)
            loss = criterion(recon, batch)
            l1_term = features.abs().mean()
            total_loss = loss + config.l1 * l1_term
            total_loss.backward()
            if config.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        metrics = evaluate(model, train_loader, criterion, config, split='train')
        metrics.update(evaluate(model, val_loader, criterion, config, split='val'))
        history.append({'epoch': float(epoch + 1), **{k: float(v) for k, v in metrics.items()}})

    return {
        'state_dict': model.state_dict(),
        'history': history,
        'd_in': torch.tensor(d_in),
        'd_hidden': torch.tensor(d_hidden),
    }


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def parse_args(argv: Sequence[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(
        description='Train a sparse autoencoder on cached activations.',
    )
    parser.add_argument(
        '--activations',
        type=Path,
        nargs='+',
        required=True,
        help='One or more .npy files containing activation matrices (num_samples x dim).',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=Path('artifacts') / 'sae',
        help='Directory for outputs (weights, metrics, metadata).',
    )
    parser.add_argument(
        '--latent-size',
        type=int,
        default=None,
        help='Explicit latent size (overrides expansion factor).',
    )
    parser.add_argument(
        '--expansion',
        type=float,
        default=8.0,
        help='Expansion factor relative to input dimension when latent size is unspecified.',
    )
    parser.add_argument(
        '--l1',
        type=float,
        default=5e-4,
        help='L1 coefficient on hidden activations.',
    )
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=256, help='Mini-batch size.')
    parser.add_argument(
        '--val-fraction',
        type=float,
        default=0.1,
        help='Fraction of samples reserved for validation.',
    )
    parser.add_argument('--seed', type=int, default=0, help='Random seed for shuffling and init.')
    parser.add_argument(
        '--dtype',
        choices=['float32', 'float16', 'bfloat16'],
        default='float32',
        help='Computation dtype.',
    )
    parser.add_argument(
        '--grad-clip',
        type=float,
        default=None,
        help='Clip gradient norm to this value (disabled if omitted).',
    )

    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }

    return TrainConfig(
        activations=args.activations,
        out_dir=args.out,
        latent_size=args.latent_size,
        expansion=args.expansion,
        l1=args.l1,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=infer_device(),
        dtype=dtype_map[args.dtype],
        grad_clip=args.grad_clip,
    )


def persist_artifacts(
    config: TrainConfig, outputs: dict[str, torch.Tensor | list[dict[str, float]]]
) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = config.out_dir / 'sae.pt'
    torch.save(outputs['state_dict'], weights_path)

    history = outputs['history']
    history_path = config.out_dir / 'metrics.csv'
    if history:
        import csv

        with history_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            for row in history:
                writer.writerow(row)

    meta = {
        'activations': [str(p) for p in config.activations],
        'input_dim': int(outputs['d_in'].item()),
        'latent_dim': int(outputs['d_hidden'].item()),
        'l1': config.l1,
        'lr': config.lr,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'val_fraction': config.val_fraction,
        'seed': config.seed,
        'device': config.device.type,
        'dtype': str(config.dtype),
        'grad_clip': config.grad_clip,
    }
    (config.out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    outputs = train(config)
    persist_artifacts(config, outputs)
    print(f'[sae_train] saved weights to {config.out_dir / "sae.pt"}')


if __name__ == '__main__':
    main()
