from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate dynamic-k execution strategies.',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=Path('artifacts') / 'dynamic_k',
        help='Output directory',
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError('dynamic_k CLI is a placeholder; implement evaluation pipeline.')


if __name__ == '__main__':
    main()
