from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compute specialization and modularity metrics.',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=Path('artifacts') / 'si_modularity',
        help='Output directory',
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        'si_modularity CLI is a placeholder; implement metrics pipeline next.'
    )


if __name__ == '__main__':
    main()
