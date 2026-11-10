# GPU versions of TSB-AD's expensive parts

`TSB_AD/gpu` holds PyTorch implementations (CUDA, Apple MPS or CPU) of the VUS metrics and of four
subsequence detectors. They reproduce TSB-AD's windowing, normalisation and scoring, so their output can
be compared value by value with the originals.

| GPU name | Original | Notes |
|---|---|---|
| `get_metrics(..., version='gpu')` | VUS-ROC / VUS-PR (`opt`, 250 thresholds) | same thresholds and tie handling |
| `Sub_PCA_GPU`, `PCA_GPU` | `Sub_PCA`, `PCA` | covariance in float64 (CPU float64 on MPS) |
| `Sub_KNN_GPU`, `KNN_GPU` | `Sub_KNN`, `KNN` | exact brute-force neighbours, float64 refinement |
| `KMeansAD_U_GPU`, `KMeansAD_GPU` | `KMeansAD_U`, `KMeansAD` | k-means++ and Lloyd as in scikit-learn; seeding differs |
| `MatrixProfile_GPU` | `MatrixProfile` (stumpy) | GEMM of z-normalised windows, float64 refinement |

The GPU detectors are registered in `run_Unsupervise_AD` and take the same hyperparameters:

```python
from TSB_AD.model_wrapper import run_Unsupervise_AD
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
from TSB_AD.evaluation.metrics import get_metrics

score = run_Unsupervise_AD('Sub_PCA_GPU', data, **Optimal_Uni_algo_HP_dict['Sub_PCA_GPU'])
result = get_metrics(score, label, slidingWindow=sw, version='gpu')
```

`TSB_AD_DEVICE=cuda|mps|cpu` picks the device (default: CUDA, then MPS, then CPU).
`Run_Detector_U.py` and `Run_Detector_M.py` take `--gpu_metrics` to score with the GPU VUS.

## Validation

```bash
python validate_gpu.py --dataset_dir ../../Datasets/TSB-AD-U/ --file_list ../../Datasets/File_List/TSB-AD-U-Eva.csv \
  --cpu_vus_every 5 --out validate_U.csv
python summarize.py validate_U.csv --markdown
```

For every file and detector, `validate_gpu.py` runs the CPU and the GPU detector, compares the scores and
both VUS-PR values, and every n-th file (`--cpu_vus_every`) also runs TSB-AD's own VUS on the CPU scores.
`--multi` uses the multivariate hyperparameters, `--resume` continues an interrupted run, `--ref_out` keeps
the CPU scores for reuse (for example by the TornadoVM version).

VUS jumps when tied scores are split, so float rounding alone can move VUS-PR by more than 0.01 on series
with repeated patterns. The report therefore also gives the difference after rounding the normalised
scores to 6 decimals, and how often a relative 1e-12 perturbation of the CPU scores alone moves VUS-PR by
more than 0.01.

## Results

NVIDIA A100 (80 GB PCIe) against the CPU originals on a 128-core Arm Neoverse-N1 host, with OpenBLAS and OpenMP
limited to 64 threads (with more threads scikit-learn's neighbour search crashes inside OpenBLAS). Times
are single calls in a warm process and include host-device transfers.

**TSB-AD-U**, 64 files of the evaluation list (NAB 23, WSD 20, Stock 8, MSL 7, MITDB 5, Daphnet 1;
1,000 to 650,000 points, 5 with at least 100,000):

| | Sub_PCA | Sub_KNN | KMeansAD_U | MatrixProfile |
|:--|--:|--:|--:|--:|
| score correlation, min / median | 1.0 / 1.0 | 1.0 / 1.0 | 0.39 / 0.988 | 0.9976 / 1.0 |
| mean VUS-PR, CPU / GPU | 0.3573 / 0.3572 | 0.5420 / 0.5418 | 0.2896 / 0.2962 | 0.2629 / 0.2627 |
| mean abs. VUS-PR difference | 2.2e-5 | 1.6e-4 | 0.017 | 5.2e-4 |
| files with difference > 0.01 | 0 | 0 | 22 | 1 |
| ... after rounding to 1e-6 | 0 | 0 | 22 | 0 |
| ... CPU vs CPU with 1e-12 noise | 2 | 4 | 0 | 3 |
| CPU time (s) | 242 | 966 | 2228 | 441 |
| GPU time (s) | 25.0 | 206 | 22.1 | 56.6 |
| speedup | 9.7x | 4.7x | 101x | 7.8x |
| speedup, files >= 100k points | 10.1x | 4.6x | 121x | 6.5x |

KMeansAD_U differs because its random seeding differs: two CPU runs with seeds 2024 and 2025 differ by
0.015 VUS-PR on average over the same files, about the CPU-GPU difference.

VUS metric, 52 file x detector pairs scored by both TSB-AD's VUS and the GPU VUS on identical scores:
largest difference 1.05e-8, 662 s on the CPU against 13.6 s on the GPU (49x).

**TSB-AD-M**, 10 files of the evaluation list, one per source (Genesis, MSL, SMAP, SMD, Exathlon,
Daphnet, TAO, SWaT, CATSv2, LTDB; 1,827 to 100,000 points, 2 to 66 channels), multivariate
hyperparameters:

| | PCA | KNN | KMeansAD |
|:--|--:|--:|--:|
| score correlation, min / median | 1.0 / 1.0 | 1.0 / 1.0 | 0.43 / 0.987 |
| mean VUS-PR, CPU / GPU | 0.4567 / 0.4567 | 0.3462 / 0.3463 | 0.5664 / 0.5502 |
| mean abs. VUS-PR difference | 1.7e-9 | 1.7e-4 | 0.017 |
| files with difference > 0.01 | 0 | 0 | 3 |
| CPU time (s) | 64.2 | 8.0 | 40.9 |
| GPU time (s) | 19.8 | 2.4 | 4.1 |
| speedup | 3.2x | 3.3x | 9.9x |

These series are short, so fixed costs weigh more. VUS metric on all 30 file x detector pairs: largest
difference 2.5e-9, 221 s on the CPU against 0.9 s on the GPU (242x).

## Limits

- On MPS, float64 parts (PCA covariance, refinement) run on the CPU.
- KMeansAD's seeding differs from scikit-learn's, so its scores agree statistically, not value by value.
- Sub-KNN is exact brute force; its cost grows with N^2 m.
