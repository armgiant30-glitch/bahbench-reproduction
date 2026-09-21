from pathlib import Path
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(os.environ.get('BAHBENCH_RESULTS', 'results'))
OUT = ROOT / 'plots'
OUT.mkdir(parents=True, exist_ok=True)
SIZES = [50, 100, 200, 500]
RANKS = [2, 4, 8, 16]
METRICS = ['WA', 'UA', 'WF1']
COLORS = {'WA': '#3B6FB6', 'UA': '#D45A4A', 'WF1': '#2E8B57'}


def load_results():
    rows = []
    for n in SIZES:
        folder = ROOT / f'data_size_rank_TORGO_n{n}'
        for path in sorted(folder.glob('paper_transformer_lora_r*_facebook_hubert-base-ls960.json')):
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                metrics = data.get('metrics', {})
                rank = data.get('args', {}).get('r')
                if rank in RANKS and all(k in metrics for k in METRICS):
                    rows.append({
                        'n_per_class': n,
                        'rank': int(rank),
                        'WA': float(metrics['WA']),
                        'UA': float(metrics['UA']),
                        'WF1': float(metrics['WF1']),
                        'path': str(path),
                    })
            except Exception as exc:
                print(f'[skip] {path}: {exc}')
    return sorted(rows, key=lambda row: (row['n_per_class'], row['rank']))


def write_csv(rows):
    out = ROOT / 'data_size_rank_TORGO.csv'
    fields = ['n_per_class', 'rank', 'WA', 'UA', 'WF1']
    with out.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    print(f'[out] {out}')


def best_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['n_per_class'], []).append(row)
    best = []
    for n in SIZES:
        candidates = grouped.get(n, [])
        if candidates:
            winner = max(candidates, key=lambda row: (row['WF1'], row['UA'], row['WA']))
            best.append(winner)
    return best


def write_best_csv(rows):
    out = ROOT / 'data_size_rank_TORGO_best.csv'
    fields = ['n_per_class', 'rank', 'WA', 'UA', 'WF1']
    with out.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    print(f'[out] {out}')


def draw_heatmap(rows):
    matrix = np.full((len(SIZES), len(RANKS)), np.nan)
    index_n = {n: i for i, n in enumerate(SIZES)}
    index_r = {r: i for i, r in enumerate(RANKS)}
    for row in rows:
        matrix[index_n[row['n_per_class']], index_r[row['rank']]] = row['WF1']
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=180)
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap='YlGnBu', aspect='auto')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isnan(value):
                color = 'white' if value < 0.55 else 'black'
                ax.text(j, i, f'{value:.3f}', ha='center', va='center', color=color, fontsize=10)
    ax.set_xticks(np.arange(len(RANKS)))
    ax.set_xticklabels([f'r={r}' for r in RANKS])
    ax.set_yticks(np.arange(len(SIZES)))
    ax.set_yticklabels([f'{n}/class' for n in SIZES])
    ax.set_xlabel('LoRA rank')
    ax.set_ylabel('Training samples per class')
    ax.set_title('TORGO: sample size x LoRA rank (WF1)')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label='WF1')
    fig.tight_layout()
    out = OUT / 'data_size_rank_TORGO_heatmap.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'[out] {out}')


def draw_curves(rows):
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), dpi=180, sharex=True)
    cmap = plt.get_cmap('viridis')
    for idx, n in enumerate(SIZES):
        subset = sorted([row for row in rows if row['n_per_class'] == n], key=lambda row: row['rank'])
        if not subset:
            continue
        color = cmap(idx / max(len(SIZES) - 1, 1))
        for ax, metric in zip(axes, METRICS):
            ax.plot([row['rank'] for row in subset], [row[metric] for row in subset],
                    marker='o', linewidth=2, color=color, label=f'{n}/class')
    for ax, metric in zip(axes, METRICS):
        ax.set_title(metric)
        ax.set_xlabel('LoRA rank')
        ax.set_xticks(RANKS)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('Score')
    axes[-1].legend(loc='best', fontsize=8)
    fig.suptitle('TORGO: metric curves under sample-size x rank sweep', y=1.02)
    fig.tight_layout()
    out = OUT / 'data_size_rank_TORGO_curves.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'[out] {out}')


def draw_best_rank(best):
    fig, ax = plt.subplots(figsize=(7.0, 4.3), dpi=180)
    ns = [row['n_per_class'] for row in best]
    ranks = [row['rank'] for row in best]
    scores = [row['WF1'] for row in best]
    ax.plot(ns, ranks, marker='o', linewidth=2.4, color='#1F4E79')
    for n, rank, score in zip(ns, ranks, scores):
        ax.annotate(f'r={rank}\nWF1={score:.3f}', (n, rank), xytext=(0, 11),
                    textcoords='offset points', ha='center', fontsize=9)
    ax.set_xscale('log')
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(n) for n in SIZES])
    ax.set_xlabel('Training samples per class')
    ax.set_ylabel('Best LoRA rank by WF1')
    ax.set_yticks(RANKS)
    ax.set_title('TORGO: best LoRA rank versus sample size')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = OUT / 'data_size_rank_TORGO_best_rank.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'[out] {out}')


def conclude(best):
    if not best:
        return 'no complete result'
    first = best[0]
    last = best[-1]
    if last['rank'] > first['rank']:
        return f"best rank moved upward from r={first['rank']} to r={last['rank']}; the turning point moves right with more data."
    if last['rank'] == first['rank'] and all(row['rank'] == first['rank'] for row in best):
        return f"best rank stayed at r={first['rank']}; no turning-point shift was observed."
    return 'best rank is non-monotonic across sample sizes; the turning point is not stable.'


def main():
    rows = load_results()
    if not rows:
        raise SystemExit('no data-size rank results found')
    write_csv(rows)
    best = best_rows(rows)
    write_best_csv(best)
    draw_heatmap(rows)
    draw_curves(rows)
    draw_best_rank(best)
    print('[best] ' + '; '.join(f"{row['n_per_class']}/class -> r={row['rank']} (WF1={row['WF1']:.4f})" for row in best))
    print('[conclusion] ' + conclude(best))


if __name__ == '__main__':
    main()