from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train sparse autoencoders on cached activations.',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=Path('artifacts') / 'sae',
        help='Output directory',
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError('sae_train CLI is a placeholder; integrate SAELens training here.')


if __name__ == '__main__':
    main()
