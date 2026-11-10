# Compares the GPU detectors and VUS metrics with their CPU counterparts on TSB-AD files.
#
#   python validate_gpu.py --dataset_dir ../../Datasets/TSB-AD-U/ --file_list ../../Datasets/File_List/TSB-AD-U-Eva.csv \
#       --detectors Sub_PCA MatrixProfile --out results.csv
#
# For every file and detector it records the run time of both versions, the Pearson correlation and the
# largest difference of the min-max normalised scores, and VUS-PR / VUS-ROC of each version.
#
# VUS jumps when tied scores are split: a difference of 1e-15 between two otherwise identical score vectors
# can move VUS-PR by 0.01 or more on series with repeated patterns. Two columns separate this from real
# disagreement: VUS-PR of both versions after rounding the normalised scores to 6 decimals (*_r6), and
# VUS-PR of the CPU scores times (1 + 1e-12 noise) (vus_pr_cpu_jitter), the metric's own sensitivity.
import argparse, contextlib, io, os, random, time, warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict
from TSB_AD.evaluation.basic_metrics import generate_curve
from TSB_AD.gpu.device import get_device, synchronize
from TSB_AD.gpu.vus import vus
from TSB_AD.model_wrapper import run_Unsupervise_AD
from TSB_AD.utils.slidingWindows import find_length_rank

warnings.filterwarnings('ignore')


def seed_all(seed=2024):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def timed(fn, seed=2024):
    seed_all(seed)
    t = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = fn()
    synchronize()
    return out, time.perf_counter() - t


def norm(s):
    return MinMaxScaler().fit_transform(np.asarray(s, dtype=float).reshape(-1, 1)).ravel()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', default='../../Datasets/TSB-AD-U/')
    ap.add_argument('--file_list', default=None)
    ap.add_argument('--files', nargs='*', default=None)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--detectors', nargs='+', default=None,
                    help='default: Sub_PCA Sub_KNN KMeansAD_U MatrixProfile, or PCA KNN KMeansAD with --multi')
    ap.add_argument('--multi', action='store_true', help="use TSB-AD's multivariate hyperparameters")
    ap.add_argument('--skip_cpu_vus', action='store_true', help='score VUS with the GPU version only')
    ap.add_argument('--cpu_vus_every', type=int, default=1, help='run the CPU VUS reference on every n-th file only')
    ap.add_argument('--no_seed2', action='store_true', help='skip the second CPU k-means seed')
    ap.add_argument('--resume', action='store_true', help='keep the rows already in --out and skip their files and detectors')
    ap.add_argument('--out', default='validate_gpu.csv')
    ap.add_argument('--ref_out', default=None,
                    help='also write the CPU scores (REF/DET/STEM.f64) and times (REF/results.csv) for reuse')
    args = ap.parse_args()

    files = args.files or pd.read_csv(args.file_list)['file_name'].tolist()
    hp_dict = Optimal_Multi_algo_HP_dict if args.multi else Optimal_Uni_algo_HP_dict
    if args.detectors is None:
        args.detectors = ['PCA', 'KNN', 'KMeansAD'] if args.multi else ['Sub_PCA', 'Sub_KNN', 'KMeansAD_U', 'MatrixProfile']
    if args.limit:
        files = files[:args.limit]
    print('device:', get_device())
    rows = []
    if args.resume and os.path.exists(args.out):
        rows = pd.read_csv(args.out).to_dict('records')
    done = {(r['file'], r['detector']) for r in rows}
    for fi, f in enumerate(files):
        df = pd.read_csv(os.path.join(args.dataset_dir, f)).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        sw = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        for name in args.detectors:
            if (f, name) in done:
                continue
            hp = hp_dict[name]
            cpu, t_cpu = timed(lambda: run_Unsupervise_AD(name, data, **hp))
            gpu, t_gpu = timed(lambda: run_Unsupervise_AD(name + '_GPU', data, **hp))
            if isinstance(cpu, str) or isinstance(gpu, str):
                print(f, name, 'failed:', cpu if isinstance(cpu, str) else gpu)
                continue
            a, b = norm(cpu), norm(gpu)
            corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')
            (vr_g, vp_g), t_vus_g = timed(lambda: vus(label, b, sw))
            (vr_gc, vp_gc) = vus(label, a, sw)
            jit = norm(a * (1 + 1e-12 * np.random.default_rng(0).standard_normal(len(a))))
            vp_jit = vus(label, jit, sw)[1]
            vp_g6 = vus(label, np.round(b, 6), sw)[1]
            vp_c6 = vus(label, np.round(a, 6), sw)[1]
            row = dict(file=f, n=len(label), sw=sw, detector=name, t_cpu=t_cpu, t_gpu=t_gpu,
                       corr=corr, max_abs_diff=float(np.abs(a - b).max()),
                       vus_pr_gpu=vp_g, vus_roc_gpu=vr_g, vus_pr_cpu=vp_gc, vus_roc_cpu=vr_gc, t_vus_gpu=t_vus_g,
                       vus_pr_cpu_jitter=vp_jit, vus_pr_cpu_r6=vp_c6, vus_pr_gpu_r6=vp_g6)
            if name.startswith('KMeansAD') and not args.no_seed2:
                # k-means depends on its random seeding: a second CPU seed gives the spread to expect
                cpu2, _ = timed(lambda: run_Unsupervise_AD(name, data, **hp), seed=2025)
                row.update(vus_pr_cpu_seed2=vus(label, norm(cpu2), sw)[1])
            if not args.skip_cpu_vus and fi % args.cpu_vus_every == 0:
                r, t_vus_c = timed(lambda: generate_curve(label, a, sw, 'opt', 250))
                row.update(vus_pr_cpu_ref=r[7], vus_roc_cpu_ref=r[6], t_vus_cpu=t_vus_c,
                           vus_err=max(abs(r[7] - vp_gc), abs(r[6] - vr_gc)))
            rows.append(row)
            if args.ref_out:
                os.makedirs(os.path.join(args.ref_out, name), exist_ok=True)
                np.asarray(cpu, dtype='<f8').tofile(os.path.join(args.ref_out, name, os.path.splitext(f)[0] + '.f64'))
                ref_csv = os.path.join(args.ref_out, 'results.csv')
                pd.DataFrame([dict(file=f, detector=name, n=len(label), eval_window=sw, cpu_ms=t_cpu * 1e3,
                                   vus_ms=row.get('t_vus_cpu', np.nan) * 1e3, vus_roc=row.get('vus_roc_cpu_ref', np.nan),
                                   vus_pr=row.get('vus_pr_cpu_ref', np.nan))]).to_csv(
                    ref_csv, mode='a', header=not os.path.exists(ref_csv), index=False)
            print(f"{f[:44]:44s} {name:14s} n={len(label):7d} cpu {t_cpu:7.2f}s gpu {t_gpu:6.2f}s corr {corr:.6f} "
                  f"VUS-PR cpu {vp_gc:.4f} gpu {vp_g:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
