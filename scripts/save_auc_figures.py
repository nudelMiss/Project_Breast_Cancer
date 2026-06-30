"""
Generate and save all CORUM AUC figures to results/auc/.
Reproduces the interactive dashboard charts as static matplotlib PNGs.
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

BASE   = Path(__file__).parent.parent
AUC_IN = BASE / 'results' / 'auc_results'
OUT    = BASE / 'results' / 'auc'
OUT.mkdir(parents=True, exist_ok=True)

TEAL   = '#1D9E75'
CORAL  = '#D85A30'
PURPLE = '#7F77DD'
AMBER  = '#BA7517'
GRAY   = '#888780'
BLUE   = '#378ADD'

CELLS = {
    'CID3586':4593,'CID3838':1343,'CID3921':1473,'CID3941':266,
    'CID3948':1438,'CID3963':2668,'CID4040':1510,'CID4066':2165,
    'CID4067':686, 'CID4290A':534,'CID4398':3718,'CID44041':742,
    'CID4463':215, 'CID4471':604, 'CID4495':3421,'CID44971':4347,
    'CID44991':827,'CID4513':1454,'CID4515':402, 'CID45171':1344,
    'CID4530N':279,
}

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
})

rng = np.random.default_rng(42)

def load(tag):
    p = AUC_IN / f'star_var75_{tag}' / 'summary_unknown_metric_walks_unknown_walks.csv'
    df = pd.read_csv(p)
    df['patient'] = df['embedding_path'].str.extract(r'/(CID[^/]+)_Tcells/')
    return df

def box_stats(aucs):
    s = np.sort(np.asarray(aucs))
    return {'q1':(s[4]+s[5])/2, 'q3':(s[15]+s[16])/2,
            'median':s[len(s)//2], 'mean':s.mean(), 'min':s[0], 'max':s[-1]}

def dot_color(auc):
    return TEAL if auc > 0.5 else CORAL

def add_hline(ax, y=0.5, label='random (0.5)'):
    ax.axhline(y, color=AMBER, linestyle='--', linewidth=1.5, label=label)

df5   = load('w5');  df5['cells']  = df5['patient'].map(CELLS)
df50  = load('w50')
df100 = load('w100')

jx_global = rng.uniform(-0.25, 0.25, 21)

# ── 1. strip + IQR box (w5) ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 7))
aucs5 = df5['mean_auc'].values
bs = box_stats(aucs5)
rect = mpatches.FancyBboxPatch((-0.35, bs['q1']), 0.70, bs['q3']-bs['q1'],
    boxstyle='square,pad=0', linewidth=1, edgecolor=TEAL, facecolor=TEAL, alpha=0.12)
ax.add_patch(rect)
ax.hlines(bs['median'], -0.35, 0.35, colors=PURPLE, linewidths=2.5, label=f"median {bs['median']:.4f}")
ax.hlines(bs['mean'],   -0.35, 0.35, colors=BLUE,   linewidths=1.5, linestyles='--', label=f"mean {bs['mean']:.4f}")
for yv in [bs['min'], bs['max']]:
    ax.vlines(0, min(yv,bs['q1']), max(yv,bs['q3']), colors=TEAL, linewidths=0.8, linestyles=':')
    ax.hlines(yv, -0.12, 0.12, colors=TEAL, linewidths=1.2)
ax.scatter(jx_global, aucs5, c=[dot_color(a) for a in aucs5], s=60, zorder=5,
           edgecolors='white', linewidths=0.5)
add_hline(ax)
ax.set_xlim(-0.6, 0.6); ax.set_xticks([])
ax.set_ylabel('mean AUC')
ax.set_title('AUC distribution — 21 T-cell patients\n(var75, star walk, 5 walks per gene)')
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT / 'auc_strip_w5.png'); plt.close(fig)
print('Saved: auc_strip_w5.png')

# ── 2. per-patient ranking (w5) ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ranked = df5.sort_values('mean_auc').reset_index(drop=True)
ax.barh(ranked['patient'], ranked['mean_auc'] - 0.47, left=0.47,
        color=[dot_color(a) for a in ranked['mean_auc']],
        edgecolor='white', linewidth=0.4, height=0.7)
add_hline(ax)
ax.set_xlim(0.47, 0.550); ax.set_xlabel('mean AUC')
ax.set_title('Per-patient AUC ranking (var75, star walk, w5)')
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT / 'auc_patient_ranking_w5.png'); plt.close(fig)
print('Saved: auc_patient_ranking_w5.png')

# ── 3. cell count vs AUC (w5) ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df5['cells'], df5['mean_auc'],
           c=[dot_color(a) for a in df5['mean_auc']], s=65,
           edgecolors='white', linewidths=0.5, zorder=5)
for _, row in df5.iterrows():
    if row['patient'] in {'CID44041','CID4530N','CID3586','CID44971','CID4463'}:
        ax.annotate(row['patient'], (row['cells'], row['mean_auc']),
                    textcoords='offset points', xytext=(6, 2), fontsize=8)
r = np.corrcoef(df5['cells'], df5['mean_auc'])[0, 1]
add_hline(ax)
ax.set_xlabel('number of T-cells'); ax.set_ylabel('mean AUC')
ax.set_title(f'Cell count vs mean AUC — var75, w5   (Pearson r = {r:.3f})')
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT / 'auc_cells_vs_auc_w5.png'); plt.close(fig)
print('Saved: auc_cells_vs_auc_w5.png')

# ── 4. gene coverage vs AUC (w5) ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df5['n_shared_genes'], df5['mean_auc'],
           c=[dot_color(a) for a in df5['mean_auc']], s=65,
           edgecolors='white', linewidths=0.5, zorder=5)
for _, row in df5.iterrows():
    if row['patient'] in {'CID44041','CID4463','CID3941','CID44971'}:
        ax.annotate(row['patient'], (row['n_shared_genes'], row['mean_auc']),
                    textcoords='offset points', xytext=(5, 2), fontsize=8)
r2 = np.corrcoef(df5['n_shared_genes'], df5['mean_auc'])[0, 1]
add_hline(ax)
ax.set_xlabel('CORUM-shared genes in embedding'); ax.set_ylabel('mean AUC')
ax.set_title(f'Gene coverage vs mean AUC — var75, w5   (Pearson r = {r2:.3f})')
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT / 'auc_coverage_vs_auc_w5.png'); plt.close(fig)
print('Saved: auc_coverage_vs_auc_w5.png')

# ── 5. per-patient walk trajectory ───────────────────────────────────────────
by_walk = {
    5:   dict(zip(df5['patient'],   df5['mean_auc'])),
    50:  dict(zip(df50['patient'],  df50['mean_auc'])),
    100: dict(zip(df100['patient'], df100['mean_auc'])),
}
patients = sorted(df5['patient'].tolist())
walk_x = [5, 50, 100]

fig, ax = plt.subplots(figsize=(8, 6))
for pid in patients:
    ys = [by_walk[w].get(pid, np.nan) for w in walk_x]
    if pid == 'CID44041':
        ax.plot(walk_x, ys, color=CORAL, linewidth=2.5, linestyle='--', zorder=5,
                label='CID44041 (w5 outlier — collapses)')
        ax.scatter(walk_x, ys, color=CORAL, s=55, zorder=6)
    elif pid == 'CID45171':
        ax.plot(walk_x, ys, color=TEAL, linewidth=2.5, zorder=5,
                label='CID45171 (most consistent)')
        ax.scatter(walk_x, ys, color=TEAL, s=55, zorder=6)
    else:
        ax.plot(walk_x, ys, color=GRAY, linewidth=0.9, alpha=0.45, zorder=2)

means = [np.mean([by_walk[w][p] for p in patients]) for w in walk_x]
ax.plot(walk_x, means, color=PURPLE, linewidth=3, zorder=6, label='cohort mean')
ax.scatter(walk_x, means, color=PURPLE, s=70, zorder=7)

add_hline(ax)
ax.set_xticks(walk_x); ax.set_xticklabels(['5 walks', '50 walks', '100 walks'])
ax.set_ylabel('mean AUC'); ax.set_ylim(0.465, 0.550)
ax.set_title('Per-patient AUC trajectory across walk counts — var75, k5')
ax.legend(fontsize=9, loc='upper right')
fig.tight_layout(); fig.savefig(OUT / 'auc_walk_trajectory.png'); plt.close(fig)
print('Saved: auc_walk_trajectory.png')

# ── 6. three-way comparison strip ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
configs = [(0, '5 walks', df5), (1, '50 walks', df50), (2, '100 walks', df100)]
jx2 = rng.uniform(-0.22, 0.22, 21)
for xc, label, df in configs:
    bs = box_stats(df['mean_auc'].values)
    bw = 0.35
    ax.add_patch(mpatches.FancyBboxPatch(
        (xc-bw/2, bs['q1']), bw, bs['q3']-bs['q1'],
        boxstyle='square,pad=0', linewidth=1,
        edgecolor=GRAY, facecolor=GRAY, alpha=0.15))
    ax.hlines(bs['median'], xc-bw/2, xc+bw/2, colors=PURPLE, linewidths=2.5)
    ax.scatter(xc + jx2, df['mean_auc'].values,
               c=[dot_color(a) for a in df['mean_auc']], s=50,
               edgecolors='white', linewidths=0.4, zorder=5)
add_hline(ax)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['5 walks', '50 walks', '100 walks'])
ax.set_ylabel('mean AUC'); ax.set_ylim(0.465, 0.550)
ax.set_title('AUC distribution by walk count — var75, k5\n(box = IQR, bar = median)')
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT / 'auc_walk_comparison_strip.png'); plt.close(fig)
print('Saved: auc_walk_comparison_strip.png')

# ── 7-9. copy and rename the seaborn boxplots from aggregation script ─────────
for tag in ('w5', 'w50', 'w100'):
    src = AUC_IN / f'star_var75_{tag}' / 'summary_unknown_metric_walks_unknown_walks.png'
    dst = OUT / f'auc_aggregation_{tag}.png'
    if src.exists():
        shutil.copy2(src, dst)
        print(f'Copied:  auc_aggregation_{tag}.png')
    else:
        print(f'WARN: aggregation PNG missing for {tag}')

print(f'\nAll done. Files in {OUT}:')
for f in sorted(OUT.glob('*.png')):
    sz = f.stat().st_size // 1024
    print(f'  {f.name:<45} {sz} KB')
