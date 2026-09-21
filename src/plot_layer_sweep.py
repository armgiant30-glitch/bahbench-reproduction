from pathlib import Path
import csv
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(os.environ.get('BAHBENCH_RESULTS', 'results'))
OUT = ROOT / 'plots'
OUT.mkdir(parents=True, exist_ok=True)
TASKS = {
    'ICBHI': ROOT / 'layer_sweep_ICBHI',
    'TORGO': ROOT / 'layer_sweep_TORGO',
    'SEP-28k': ROOT / 'layer_sweep_SEP-28k',
}
LAYERS = [1, 4, 7, 10, 12]
METRICS = ['WA', 'UA', 'WF1']
COLORS = {'WA': '#3B6FB6', 'UA': '#D45A4A', 'WF1': '#2E8B57'}
PATTERN = re.compile(r'_layer(-?\d+)_')


def load_task(folder):
    rows = []
    for path in sorted(folder.glob('paper_transformer_frozen_layer*_facebook_hubert-base-ls960.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            metrics = data.get('metrics', {})
            match = PATTERN.search(path.name)
            layer = int(match.group(1)) if match else data.get('args', {}).get('layer')
            if layer in LAYERS and all(k in metrics for k in METRICS):
                rows.append({
                    'layer': int(layer),
                    'WA': float(metrics['WA']),
                    'UA': float(metrics['UA']),
                    'WF1': float(metrics['WF1']),
                    'path': str(path),
                })
        except Exception as exc:
            print(f'[skip] {path}: {exc}')
    return sorted(rows, key=lambda row: row['layer'])


def write_csv(rows_by_task):
    combined = []
    for task, rows in rows_by_task.items():
        out = ROOT / f'layer_sweep_{task.replace("-", "").lower()}.csv'
        fields = ['layer', 'WA', 'UA', 'WF1']
        with out.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in rows)
        print(f'[out] {out}')
        for row in rows:
            combined.append({'task': task, **{key: row[key] for key in fields}})
    if combined:
        out = ROOT / 'layer_sweep_summary.csv'
        fields = ['task', 'layer', 'WA', 'UA', 'WF1']
        with out.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(combined)
        print(f'[out] {out}')


def draw_task(task, rows):
    layers = [row['layer'] for row in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=180)
    for metric in METRICS:
        ax.plot(layers, [row[metric] for row in rows], marker='o', linewidth=2.2,
                color=COLORS[metric], label=metric)
    ax.axvspan(0.5, 6.5, color='#EAF2F8', alpha=0.65, label='low/mid 1-6')
    ax.axvspan(6.5, 12.5, color='#FDEDEC', alpha=0.6, label='high 7-12')
    best = max(rows, key=lambda row: (row['WF1'], row['UA']))
    ax.axvline(best['layer'], color='#555555', linestyle=':', linewidth=1.2)
    ax.annotate(f"best layer={best['layer']}\nWF1={best['WF1']:.4f}",
                xy=(best['layer'], best['WF1']), xytext=(8, -34), textcoords='offset points',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#999999'))
    ax.set_title(f'{task}: HuBERT layer selection (paper Transformer)')
    ax.set_xlabel('Hidden-state layer')
    ax.set_ylabel('Score')
    ax.set_xticks(LAYERS)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    out = OUT / f'layer_sweep_{task.replace("-", "").lower()}.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'[out] {out}')


def conclusion(rows_by_task):
    parts = []
    for task, rows in rows_by_task.items():
        if not rows:
            continue
        best = max(rows, key=lambda row: (row['WF1'], row['UA']))
        last = next((row for row in rows if row['layer'] == 12), None)
        delta = None if last is None else best['WF1'] - last['WF1']
        parts.append(f"{task}: best layer={best['layer']}, WF1={best['WF1']:.4f}"
                     + (f", delta_vs_layer12={delta:+.4f}" if delta is not None else ''))
    if not parts:
        return 'no complete layer results'
    return '; '.join(parts)


def main():
    rows_by_task = {task: load_task(folder) for task, folder in TASKS.items()}
    if not any(rows_by_task.values()):
        raise SystemExit('no layer sweep results found')
    write_csv(rows_by_task)
    for task, rows in rows_by_task.items():
        if rows:
            draw_task(task, rows)
    print('[summary] ' + conclusion(rows_by_task))


if __name__ == '__main__':
    main()