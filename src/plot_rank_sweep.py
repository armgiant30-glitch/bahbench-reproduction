from pathlib import Path
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get('BAHBENCH_RESULTS', 'results'))
OUT = ROOT / 'plots'
OUT.mkdir(parents=True, exist_ok=True)


def task_dir(name):
    candidates = [
        ROOT / f'paper_transformer_lora_{name}',
        ROOT / f'paper_transformer_lora_{name.lower()}',
        ROOT / f'paper_transformer_lora_{name.lower().replace("-", "")}',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

TASKS = {
    'ICBHI': task_dir('ICBHI'),
    'TORGO': task_dir('TORGO'),
    'SEP-28k': task_dir('SEP-28k'),
}
COLORS = {'WA': '#1f77b4', 'UA': '#d62728', 'WF1': '#2ca02c'}

def load_rows(folder):
    rows = []
    for p in sorted(folder.glob('paper_transformer_lora_r*_facebook_hubert-base-ls960.json')):
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
            m = d.get('metrics', {})
            r = d.get('args', {}).get('r')
            if r is not None and all(k in m for k in ('WA', 'UA', 'WF1')):
                rows.append((int(r), m['WA'], m['UA'], m['WF1']))
        except Exception as e:
            print(f'[skip] {p}: {e}')
    return sorted(rows)

summary = {}
for name, folder in TASKS.items():
    rows = load_rows(folder)
    if not rows:
        print(f'[warn] no data for {name}: {folder}')
        continue
    summary[name] = rows
    rs = [x[0] for x in rows]
    wa = [x[1] for x in rows]
    ua = [x[2] for x in rows]
    wf1 = [x[3] for x in rows]
    best = max(rows, key=lambda x: x[3])
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=180)
    ax.plot(rs, wa, marker='o', linewidth=2, color=COLORS['WA'], label='WA')
    ax.plot(rs, ua, marker='s', linewidth=2, color=COLORS['UA'], label='UA')
    ax.plot(rs, wf1, marker='^', linewidth=2.6, color=COLORS['WF1'], label='WF1')
    ax.axvline(best[0], color='#888888', linestyle=':', linewidth=1.3)
    ax.scatter([best[0]], [best[3]], s=90, facecolors='none', edgecolors='black', linewidths=1.5)
    ax.annotate(f'best r={best[0]}\nWF1={best[3]:.4f}', xy=(best[0], best[3]),
                xytext=(8, -32), textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#999999'))
    ax.set_title(f'{name}: HuBERT-base LoRA rank sweep (paper Transformer)')
    ax.set_xlabel('LoRA rank r')
    ax.set_ylabel('Score')
    ax.set_xticks(rs)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc='best')
    fig.tight_layout()
    out = OUT / f'lora_rank_{name.lower().replace("-", "")}.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('[out]', out)

if summary:
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=180)
    for name, rows in summary.items():
        rs = [x[0] for x in rows]
        wf1 = [x[3] for x in rows]
        line, = ax.plot(rs, wf1, marker='o', linewidth=2.2, label=name)
        best = max(rows, key=lambda x: x[3])
        ax.scatter([best[0]], [best[3]], s=75, facecolors='none', edgecolors=line.get_color(), linewidths=1.5)
        ax.annotate(f'r={best[0]}', xy=(best[0], best[3]), xytext=(5, 6),
                    textcoords='offset points', fontsize=9, color=line.get_color())
    ax.set_title('HuBERT-base LoRA rank sweep: WF1 comparison')
    ax.set_xlabel('LoRA rank r')
    ax.set_ylabel('WF1')
    ax.set_xticks(sorted({x[0] for rows in summary.values() for x in rows}))
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc='best')
    fig.tight_layout()
    out = OUT / 'lora_rank_wf1_summary.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('[out]', out)