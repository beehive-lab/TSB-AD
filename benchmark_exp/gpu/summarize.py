# Summarises a validate_gpu.py result file per detector.
#
#   python summarize.py validate_gpu.csv [--markdown]
import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('csv')
ap.add_argument('--markdown', action='store_true')
args = ap.parse_args()
d = pd.read_csv(args.csv)
rows = []
for det, g in d.groupby('detector', sort=False):
    dv = (g.vus_pr_gpu - g.vus_pr_cpu).abs()
    dv6 = (g.vus_pr_gpu_r6 - g.vus_pr_cpu_r6).abs()
    jit = (g.vus_pr_cpu_jitter - g.vus_pr_cpu).abs()
    row = {
        'detector': det,
        'files': len(g),
        'score corr (min)': g['corr'].min(),
        'score corr (median)': g['corr'].median(),
        'mean VUS-PR CPU': g.vus_pr_cpu.mean(),
        'mean VUS-PR GPU': g.vus_pr_gpu.mean(),
        '|dVUS-PR| mean': dv.mean(),
        '|dVUS-PR| > 0.01': int((dv > 0.01).sum()),
        '|dVUS-PR| > 0.01, rounded 1e-6': int((dv6 > 0.01).sum()),
        'CPU jitter 1e-12 > 0.01': int((jit > 0.01).sum()),
        'CPU s': g.t_cpu.sum(),
        'GPU s': g.t_gpu.sum(),
        'speedup': g.t_cpu.sum() / g.t_gpu.sum(),
        'speedup n>=100k': g[g.n >= 100000].t_cpu.sum() / max(g[g.n >= 100000].t_gpu.sum(), 1e-9),
    }
    if 'vus_pr_cpu_seed2' in g:
        row['|dVUS-PR| CPU seed 2024 vs 2025 mean'] = (g.vus_pr_cpu_seed2 - g.vus_pr_cpu).abs().mean()
    rows.append(row)
s = pd.DataFrame(rows).set_index('detector').T
if 'vus_err' in d:
    c = d.dropna(subset=['vus_err'])
    vus = pd.Series({'VUS checked (file x detector)': len(c), 'max |VUS GPU - VUS CPU| same scores': c.vus_err.max(),
                     'CPU VUS s': c.t_vus_cpu.sum(), 'GPU VUS s': c.t_vus_gpu.sum(),
                     'VUS speedup': c.t_vus_cpu.sum() / c.t_vus_gpu.sum()})
else:
    vus = None
if args.markdown:
    print(s.to_markdown(floatfmt='.4g'))
    if vus is not None:
        print()
        print(vus.to_frame('VUS metric').to_markdown(floatfmt='.4g'))
else:
    pd.set_option('display.width', 200)
    print(s)
    if vus is not None:
        print(vus)
