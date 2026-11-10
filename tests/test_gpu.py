"""Checks the GPU implementations against the CPU ones on synthetic series (runs on CPU if no GPU)."""
import numpy as np
import pytest

from TSB_AD.evaluation.basic_metrics import generate_curve
from TSB_AD.gpu.vus import vus
from TSB_AD.models.KNN import KNN
from TSB_AD.models.MatrixProfile import MatrixProfile as MatrixProfileCPU
from TSB_AD.models.PCA import PCA
from TSB_AD.gpu import detectors as gpu


def series(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 50) + 0.1 * rng.standard_normal(n)
    label = np.zeros(n, dtype=int)
    for s in (700, 1500, 2400):
        x[s:s + 30] += rng.normal(0, 1.5, 30)
        label[s:s + 30] = 1
    label[2995:] = 1  # a run touching the end
    return x.reshape(-1, 1), label


@pytest.mark.parametrize("window", [0, 1, 7, 50, 125])
def test_vus_matches_cpu(window):
    x, label = series()
    score = np.abs(x.ravel()) + np.random.default_rng(1).normal(0, 0.05, len(label))
    score = (score - score.min()) / (score.max() - score.min())
    ref = generate_curve(label, score, window, 'opt', 250)
    roc, pr = vus(label, score, window)
    assert abs(roc - ref[6]) < 1e-7
    assert abs(pr - ref[7]) < 1e-7


def test_vus_with_ties():
    x, label = series()
    score = np.round(np.abs(x.ravel()), 1)  # heavy ties
    ref = generate_curve(label, score, 40, 'opt', 250)
    roc, pr = vus(label, score, 40)
    assert abs(roc - ref[6]) < 1e-7 and abs(pr - ref[7]) < 1e-7


def test_sub_pca():
    x, _ = series()
    a = PCA(slidingWindow=50).fit(x).decision_scores_
    b = gpu.SubPCA(slidingWindow=50).fit(x).decision_scores_
    assert np.corrcoef(a, b)[0, 1] > 0.99999
    assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < 1e-4


def test_sub_knn():
    x, _ = series()
    a = KNN(slidingWindow=50, n_neighbors=10).fit(x).decision_scores_
    b = gpu.SubKNN(slidingWindow=50, n_neighbors=10).fit(x).decision_scores_
    assert np.max(np.abs(a - b)) < 1e-9


def test_matrix_profile():
    x, _ = series()
    x[100:200] = 3.0  # constant stretch
    a = MatrixProfileCPU(window=50).fit(x).decision_scores_
    b = gpu.MatrixProfile(window=50).fit(x).decision_scores_
    assert np.max(np.abs(a - b)) < 1e-6


def test_kmeans_ad_shape_and_quality():
    x, label = series()
    s = gpu.KMeansAD(k=5, window_size=50, random_state=0).fit_predict(x)
    assert s.shape == (len(label),)
    assert s[label == 1].mean() > s[label == 0].mean()
