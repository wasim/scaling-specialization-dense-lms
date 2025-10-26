from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot sparsity metrics from a CSV file.')
    parser.add_argument(
        'csv',
        type=Path,
        help='Path to sparsity CSV produced by sdlms.cli.sparsity',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Optional output image path (defaults to <csv>_plot.png)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    if 'task_id' in df.columns:
        title = f"Sparsity metrics — {df['task_id'].iloc[0]}"
    else:
        title = 'Sparsity metrics'

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    df.plot(x='layer', y='frac', kind='bar', ax=axes[0], legend=False, color='#4c72b0')
    axes[0].set_ylabel('Activation fraction')
    axes[0].set_xlabel('Layer')

    df.plot(x='layer', y='pr', kind='bar', ax=axes[1], legend=False, color='#55a868')
    axes[1].set_ylabel('Participation ratio')
    axes[1].set_xlabel('Layer')

    fig.suptitle(title)
    fig.tight_layout()

    output_path = args.output or args.csv.with_suffix('.png')
    fig.savefig(output_path, dpi=200)
    print(f'[plot] wrote {output_path}')


if __name__ == '__main__':
    main()
