from pathlib import Path
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(os.environ.get('BAHBENCH_RESULTS', 'results'))
FOLDER = ROOT / 'pooling_TORGO'
OUT = ROOT / 'plots'
OUT.mkdir(parents=True, exist_ok=True)
POOLS = ['cls', 'mean', 'max']


def load_rows():
    rows = []
    for path in sorted(FOLDER.glob('paper_transformer_frozen*facebook_hubert-base-ls960.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            args = data.get('args', {})
            metrics = data.get('metrics', {})
            pool = args.get('pool', 'cls')
            if pool in POOLS and all(k in metrics for k in ('WA', 'UA', 'WF1')):
                rows.append({
                    'pool': pool,
                    'WA': float(metrics['WA']),
                    'UA': float(metrics['UA']),
                    'WF1': float(metrics['WF1']),
                    'path': str(path),
                })
        except Exception as exc:
            print(f'[skip] {path}: {exc}')
    rank = {name: i for i, name in enumerate(POOLS)}
    return sorted(rows, key=lambda row: rank[row['pool']])


def write_csv(rows):
    out = ROOT / 'pooling_comparison_TORGO.csv'
    with out.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['pool', 'WA', 'UA', 'WF1'])
        writer.writeheader()
        writer.writerows({k: row[k] for k in ('pool', 'WA', 'UA', 'WF1')} for row in rows)
    print(f'[out] {out}')


def draw(rows):
    pools = [row['pool'] for row in rows]
    x = np.arange(len(pools))
    width = 0.24
    metrics = [('WA', '#3B6FB6'), ('UA', '#D45A4A'), ('WF1', '#2E8B57')]
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=180)
    for offset, (name, color) in zip([-width, 0.0, width], metrics):
        values = [row[name] for row in rows]
        bars = ax.bar(x + offset, values, width=width, label=name, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8.5)
    best = max(rows, key=lambda row: row['WF1'])
    ax.set_title('TORGO pooling comparison (HuBERT-base, frozen Transformer)')
    ax.set_xlabel('Pooling over encoder sequence')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(pools)
    ax.set_ylim(0, 1.08)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(loc='lower right')
    ax.text(0.015, 0.98, f"best WF1: {best['pool']} = {best['WF1']:.4f}",
            transform=ax.transAxes, ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#999999'))
    fig.tight_layout()
    out = OUT / 'pooling_TORGO_comparison.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'[out] {out}')


def summarize(rows):
    by_pool = {row['pool']: row for row in rows}
    if 'mean' in by_pool and 'max' in by_pool:
        delta = by_pool['mean']['WF1'] - by_pool['max']['WF1']
        if abs(delta) < 0.01:
            conclusion = 'mean 与 max 的 WF1 差异小于 1 个百分点，本数据上未观察到稳定优势。'
        else:
            winner = 'mean' if delta > 0 else 'max'
            conclusion = f'{winner} pooling 的 WF1 更高，差值为 {abs(delta):.4f}。'
    else:
        conclusion = 'mean/max 数据不完整，暂不判定。'
    print('[summary] ' + '; '.join(f"{row['pool']}: WF1={row['WF1']:.4f}, UA={row['UA']:.4f}" for row in rows))
    print('[conclusion] ' + conclusion)


def main():
    rows = load_rows()
    if not rows:
        raise SystemExit(f'no pooling results found under {FOLDER}')
    write_csv(rows)
    draw(rows)
    summarize(rows)


if __name__ == '__main__':
    main()