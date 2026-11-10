"""GPU versions of the subsequence detectors Sub-PCA, Sub-KNN, KMeansAD and MatrixProfile.

Each class mirrors the CPU detector of the same name in ``TSB_AD.models`` (same windowing,
normalisation, padding and scoring) and exposes ``fit(X)`` and ``decision_scores_``. Pairwise work
runs on the device returned by ``TSB_AD.gpu.device.get_device()`` in float32; the small numerically
sensitive pieces (PCA covariance and eigendecomposition, subsequence means and norms) stay in float64.
"""

import math

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from .device import get_device, supports_float64

_ROWS = 2048
# candidates on each side of rank k whose distances are recomputed in float64
_REFINE = 4


def _windows(X, window):
    """Window.convert(): (n - window + 1, d * window) view of the sliding windows."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return sliding_window_view(X, window_shape=window, axis=0).reshape(X.shape[0] - window + 1, -1)


def _zscore_cols(X):
    """utility.zscore(axis=0, ddof=0)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num((X - mu) / sd)


def _topk_cols(g, k, block=1024):
    """Indices of the k largest entries of each row; two-level (per column block, then global),
    which is several times faster than one torch.topk over long rows on some devices."""
    rows, cols = g.shape
    if cols <= 4 * block or k > block:
        return torch.topk(g, k, dim=1).indices
    nb = (cols + block - 1) // block
    if nb * block != cols:
        g = torch.nn.functional.pad(g, (0, nb * block - cols), value=-float("inf"))
    v, i = torch.topk(g.view(rows, nb, block), k, dim=2)
    i = i + (torch.arange(nb, device=g.device) * block)[None, :, None]
    top = torch.topk(v.reshape(rows, -1), k, dim=1).indices
    return torch.gather(i.reshape(rows, -1), 1, top)


def _pad(scores, window, n):
    """Pads window scores to n points the way the CPU detectors do."""
    if len(scores) < n:
        front = math.ceil((window - 1) / 2)
        back = (window - 1) // 2
        scores = np.concatenate([np.full(front, scores[0]), scores, np.full(back, scores[-1])])
    return scores


class SubPCA:
    """GPU counterpart of ``models.PCA.PCA`` with its default arguments (all components, weighted)."""

    def __init__(self, slidingWindow=100, n_components=None, device=None):
        self.slidingWindow = slidingWindow
        self.n_components = n_components
        self.device = device or get_device()

    def fit(self, X, y=None):
        n = X.shape[0]
        dev = self.device
        Xw = _zscore_cols(_windows(X, self.slidingWindow))
        # standardizer(): StandardScaler with unit scale for constant columns
        mu = Xw.mean(axis=0)
        sd = Xw.std(axis=0)
        sd[sd < 10 * np.finfo(np.float64).eps] = 1.0
        Xw = (Xw - mu) / sd
        Xw = Xw[:, np.any(Xw != 0, axis=0)]
        N, m = Xw.shape

        # covariance_eigh solver of sklearn.decomposition.PCA. With all components (the default) the
        # smallest eigenvalues carry the largest weights and need float64; with fewer components the
        # leading ones are well conditioned and a float32 product on the device is accurate enough
        if not supports_float64(dev) and self.n_components is not None:
            Xt = torch.as_tensor(Xw, device=dev, dtype=torch.float32)
            mean = Xw.mean(axis=0)
            C = np.zeros((m, m))
            for s in range(0, N, 1 << 14):
                b = Xt[s:s + (1 << 14)]
                C += (b.T @ b).cpu().numpy().astype(np.float64)
            C = (C - N * np.outer(mean, mean)) / (N - 1)
            del Xt
        elif supports_float64(dev):
            Xt = torch.as_tensor(Xw, device=dev)
            mean = Xt.mean(dim=0)
            C = Xt.T @ Xt - N * torch.outer(mean, mean)
            C = (C / (N - 1)).cpu().numpy()
            mean = mean.cpu().numpy()
        else:
            mean = Xw.mean(axis=0)
            C = (Xw.T @ Xw - N * np.outer(mean, mean)) / (N - 1)
        if m > 1000:
            # divide and conquer: several times faster than numpy's eigh on large matrices
            import scipy.linalg
            evals, evecs = scipy.linalg.eigh(C, driver="evd")
        else:
            evals, evecs = np.linalg.eigh(C)
        evals = evals[::-1].copy()
        evecs = evecs[:, ::-1]
        evals[evals < 0] = 0
        Vt = evecs.T.copy()
        signs = np.sign(Vt[np.arange(Vt.shape[0]), np.argmax(np.abs(Vt), axis=1)])
        Vt *= signs[:, None]
        ratio = evals / evals.sum()
        if self.n_components is None:
            k = min(N, m)
        elif 0 < self.n_components < 1:
            # sklearn: fewest components whose explained variance exceeds the fraction
            k = int(np.searchsorted(np.cumsum(ratio), self.n_components, side="right")) + 1
        else:
            k = int(self.n_components)
        comps, w = Vt[:k], ratio[:k]

        # sum_c ||x - v_c|| / w_c for unit-norm v_c
        V = torch.as_tensor(comps, device=dev, dtype=torch.float32)
        inv_w = torch.as_tensor(1.0 / w, device=dev, dtype=torch.float32)
        out = np.empty(N)
        for s in range(0, N, _ROWS * 8):
            xb = torch.as_tensor(Xw[s:s + _ROWS * 8], device=dev, dtype=torch.float32)
            d2 = (xb * xb).sum(dim=1, keepdim=True) + (V * V).sum(dim=1)[None, :] - 2 * xb @ V.T
            out[s:s + len(xb)] = (torch.sqrt(torch.clamp(d2, min=0)) @ inv_w).cpu().numpy()
        self.components_ = comps
        self.explained_variance_ratio_ = w
        self.decision_scores_ = _pad(out, self.slidingWindow, n)
        return self


class SubKNN:
    """GPU counterpart of ``models.KNN.KNN``: distance to the k-th nearest window (method 'largest'),
    'mean' or 'median', by exact brute force."""

    def __init__(self, slidingWindow=100, n_neighbors=10, method="largest", device=None):
        self.slidingWindow = slidingWindow
        self.n_neighbors = n_neighbors
        self.method = method
        self.device = device or get_device()

    def fit(self, X, y=None):
        n = X.shape[0]
        dev = self.device
        Xw = _zscore_cols(_windows(X, self.slidingWindow))
        N = Xw.shape[0]
        k = self.n_neighbors
        extra = min(k + 8, N - 1)
        A = torch.as_tensor(Xw, device=dev, dtype=torch.float32)
        sq = (A * A).sum(dim=1)
        dist = np.empty((N, k))
        cand = np.empty((N, extra), dtype=np.int64)
        for s in range(0, N, _ROWS):
            e = min(s + _ROWS, N)
            ab = A[s:e]
            # -||a - b||^2 + ||a||^2: same order as the distance, one fused kernel
            g = torch.addmm(-sq[None, :], ab, A.T, alpha=2)
            g[torch.arange(e - s, device=dev), torch.arange(s, e, device=dev)] = -float("inf")
            idx = _topk_cols(g, extra)
            # float32 distances of the candidates, sorted
            exact = ((A[idx] - ab[:, None, :]) ** 2).sum(dim=2)
            d32, order = torch.sort(exact, dim=1)
            dist[s:e] = torch.sqrt(d32[:, :k]).cpu().numpy()
            cand[s:e] = torch.gather(idx, 1, order).cpu().numpy()
        # float64 distances of the candidates near rank k, so that ties and near ties come out as on the CPU
        lo = max(0, k - 1 - _REFINE)
        hi = min(extra, k + _REFINE)
        for s in range(0, N, 4096):
            r = np.arange(s, min(s + 4096, N))
            c = cand[r, lo:hi]
            d64 = np.sort(np.sqrt(((Xw[c] - Xw[r][:, None, :]) ** 2).sum(axis=2)), axis=1)
            dist[r, lo:] = d64[:, :k - lo]
        if self.method == "largest":
            sc = dist[:, -1]
        elif self.method == "mean":
            sc = dist.mean(axis=1)
        else:
            sc = np.median(dist, axis=1)
        self.decision_scores_ = _pad(sc, self.slidingWindow, n)
        return self


class KMeansAD:
    """GPU counterpart of ``models.KMeansAD.KMeansAD`` (stride 1): k-means++ seeding and Lloyd
    iterations as in sklearn's KMeans (n_init=1, max_iter=300, tol=1e-4), distance of each window
    to its centre, averaged back onto the points."""

    def __init__(self, k, window_size, stride=1, max_iter=300, tol=1e-4, random_state=None, device=None):
        if stride != 1:
            raise ValueError("the GPU KMeansAD supports stride 1 only")
        self.k = k
        self.window_size = window_size
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.device = device or get_device()

    def _preprocess(self, X):
        S = _windows(X, self.window_size)
        mu = S.mean(axis=1, keepdims=True)
        sd = S.std(axis=1, ddof=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num((S - mu) / sd)

    def _sqdist(self, A, sqA, C):
        return torch.clamp(sqA[:, None] + (C * C).sum(dim=1)[None, :] - 2 * A @ C.T, min=0)

    def _closest(self, A, sqA, C):
        best = torch.empty(A.shape[0], device=A.device, dtype=torch.long)
        bd = torch.empty(A.shape[0], device=A.device)
        for s in range(0, A.shape[0], _ROWS * 8):
            d = self._sqdist(A[s:s + _ROWS * 8], sqA[s:s + _ROWS * 8], C)
            bd[s:s + _ROWS * 8], best[s:s + _ROWS * 8] = torch.min(d, dim=1)
        return best, bd

    def _kmeanspp(self, A, sqA, rng):
        N = A.shape[0]
        trials = 2 + int(np.log(self.k))
        centers = [int(rng.integers(N))]
        closest = self._sqdist(A, sqA, A[centers[0]:centers[0] + 1])[:, 0]
        pot = closest.sum()
        for _ in range(1, self.k):
            r = torch.as_tensor(rng.random(trials), device=A.device, dtype=torch.float32) * pot
            cand = torch.searchsorted(torch.cumsum(closest, 0), r).clamp_(max=N - 1)
            d = torch.minimum(closest[None, :], self._sqdist(A, sqA, A[cand]).T)
            pots = d.sum(dim=1)
            b = int(torch.argmin(pots))
            centers.append(int(cand[b]))
            closest, pot = d[b], pots[b]
        return A[centers].clone()

    def fit_predict(self, X, y=None):
        dev = self.device
        Xw = self._preprocess(X)
        N = Xw.shape[0]
        rng = np.random.default_rng(self.random_state)
        A = torch.as_tensor(Xw, device=dev, dtype=torch.float32)
        sqA = (A * A).sum(dim=1)
        tol = self.tol * float(np.mean(np.var(Xw, axis=0)))
        C = self._kmeanspp(A, sqA, rng)
        labels = None
        for it in range(self.max_iter):
            new_labels, _ = self._closest(A, sqA, C)
            if labels is not None and torch.equal(new_labels, labels):
                break
            labels = new_labels
            onehot = torch.nn.functional.one_hot(labels, self.k).to(torch.float32)
            counts = onehot.sum(dim=0)
            # one-hot product instead of index_add_, whose atomics are slow on some devices
            sums = onehot.T @ A
            Cn = torch.where(counts[:, None] > 0, sums / counts.clamp(min=1)[:, None], C)
            shift = float(((Cn - C) ** 2).sum())
            C = Cn
            if shift <= tol:
                break
        labels, _ = self._closest(A, sqA, C)
        diffs = torch.sqrt(((A - C[labels]) ** 2).sum(dim=1)).cpu().numpy().astype(np.float64)
        self.cluster_centers_ = C.cpu().numpy()
        self.labels_ = labels.cpu().numpy()
        # reverse windowing: mean score of the windows covering each point
        ws = self.window_size
        n = N + ws - 1
        cs = np.concatenate([[0.0], np.cumsum(diffs)])
        t = np.arange(n)
        lo = np.maximum(t - ws + 1, 0)
        hi = np.minimum(t, N - 1)
        return (cs[hi + 1] - cs[lo]) / (hi - lo + 1)


class MatrixProfile:
    """GPU counterpart of ``models.MatrixProfile.MatrixProfile`` (stumpy.stump self-join): z-normalised
    Euclidean distance to the nearest non-trivial match, exclusion zone ceil(m / 4)."""

    def __init__(self, window, device=None):
        self.window = window
        self.device = device or get_device()

    def fit(self, X, y=None):
        dev = self.device
        x = np.asarray(X, dtype=np.float64).ravel()
        m = self.window
        S = sliding_window_view(x, m)
        N = S.shape[0]
        mu = S.mean(axis=1)
        const = S.max(axis=1) == S.min(axis=1)
        Z = np.empty((N, m), dtype=np.float32)
        for s in range(0, N, 1 << 16):
            blk = S[s:s + (1 << 16)] - mu[s:s + (1 << 16), None]
            nrm = np.sqrt((blk * blk).sum(axis=1, keepdims=True))
            nrm[nrm == 0] = 1
            Z[s:s + (1 << 16)] = blk / nrm
        Z[const] = 0
        Zt = torch.as_tensor(Z, device=dev)
        excl = int(math.ceil(m / 4))
        off = torch.arange(-excl, excl + 1, device=dev)
        # stumpy: a constant subsequence is at distance 0 from another constant one and sqrt(m) from
        # any other; zeroed rows give correlation 0 (distance sqrt(2m)), so only these cases need fixing
        cum = np.concatenate([[0], np.cumsum(const)])
        lo = np.clip(np.arange(N) - excl, 0, N)
        hi = np.clip(np.arange(N) + excl + 1, 0, N)
        const_far = (cum[-1] - (cum[hi] - cum[lo])) > 0
        best = np.empty(N)
        arg = np.empty(N, dtype=np.int64)
        for s in range(0, N, _ROWS):
            e = min(s + _ROWS, N)
            corr = Zt[s:e] @ Zt.T
            band = (torch.arange(s, e, device=dev)[:, None] + off[None, :]).clamp_(0, N - 1)
            corr.scatter_(1, band, -float("inf"))
            arg[s:e] = corr.argmax(dim=1).cpu().numpy()
        # float64 distance to the best match found in float32
        for s in range(0, N, 1 << 15):
            i = np.arange(s, min(s + (1 << 15), N))
            j = arg[i]
            ci = S[i] - mu[i, None]
            cj = S[j] - mu[j, None]
            num = (ci * cj).sum(axis=1)
            den = np.sqrt((ci * ci).sum(axis=1) * (cj * cj).sum(axis=1))
            with np.errstate(divide="ignore", invalid="ignore"):
                best[i] = np.where(den > 0, num / den, 0.0)
        prof = np.sqrt(np.clip(2 * m * (1 - best), 0, None))
        prof = np.where(const_far, np.minimum(prof, np.sqrt(m)), prof)
        prof = np.where(const & const_far, 0.0, prof)
        self.profile_ = prof
        res = np.full(len(x), prof.min())
        res[m // 2:-m // 2 + 1] = prof
        self.decision_scores_ = res
        return self
