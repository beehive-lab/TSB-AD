"""VUS-ROC and VUS-PR on a GPU.

Computes the same quantities as ``basic_metricor.RangeAUC_volume_opt`` (the ``opt`` version used by
``get_metrics``), without the per-window, per-threshold, per-segment Python loops.

For a buffer window ``w`` the extended label of a point ``t`` is 1 inside an anomaly and
``min(1, sum_k sqrt(1 - d_k(t) / w))`` within ``w // 2`` points of the anomalies around it.
For a threshold ``thr`` with prediction ``pred = score >= thr`` the loop in the reference reduces to

    TP       = #{anomaly points predicted}   + sum_{t normal} ext_w(t) * pred(t)
    N_labels = #{anomaly points}             + sum_{t normal} ext_w(t) * pred(t)
    existence = #{merged extended segments of window w that contain a predicted point}

so the whole (window x threshold) grid needs one matrix product ``ext @ pred`` over the points near
anomalies, which runs on the device, plus per-segment maxima of the score for the existence term.
"""

import numpy as np
import torch

from .device import get_device

_CHUNK = 1 << 15


def _ranges(label):
    """[(start, end)] of the anomalous runs of a 0/1 array, ends inclusive (as range_convers_new)."""
    d = np.diff(label)
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0]
    if len(ends):
        if not len(starts) or ends[0] < starts[0]:
            starts = np.concatenate([[0], starts])
    if len(starts):
        if not len(ends) or ends[-1] < starts[-1]:
            ends = np.concatenate([ends, [len(label) - 1]])
    return starts.astype(np.int64), ends.astype(np.int64)


def _merged(starts, ends, window, n):
    """Segments of new_sequence(): runs extended by window // 2, merged when they overlap."""
    h = window // 2
    split = ends[:-1] + h < starts[1:] - h
    first = np.concatenate([[True], split])
    last = np.concatenate([split, [True]])
    a = starts[first] - h
    b = ends[last] + h
    a[0] = max(a[0], 0)
    b[-1] = min(b[-1], n - 1)
    return a, b


def _range_max(score, a, b):
    """max(score[a_i : b_i + 1]) for each i, with a sparse table."""
    n = len(score)
    levels = [score]
    k = 1
    while (1 << k) <= n:
        prev = levels[-1]
        half = 1 << (k - 1)
        levels.append(np.maximum(prev[:-half], prev[half:]))
        k += 1
    length = b - a + 1
    lg = np.floor(np.log2(length)).astype(np.int64)
    out = np.empty(len(a))
    for lv in np.unique(lg):
        sel = lg == lv
        tab = levels[lv]
        out[sel] = np.maximum(tab[a[sel]], tab[b[sel] - (1 << lv) + 1])
    return out


def vus(label, score, sliding_window, thre=250, device=None, return_curves=False):
    """Returns (VUS_ROC, VUS_PR); with return_curves, also the per-window AUC-ROC and AUC-PR."""
    device = device or get_device()
    label = np.asarray(label).astype(int).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    n = len(score)
    W = int(sliding_window) + 1
    P = float(label.sum())

    starts, ends = _ranges(label)
    la, lb = _merged(starts, ends, int(sliding_window), n)

    # thresholds and the counts that do not depend on the window
    sorted_asc = np.sort(score)
    idx = np.linspace(0, n - 1, thre).astype(int)
    thr = sorted_asc[::-1][idx]
    n_pred = n - np.searchsorted(sorted_asc, thr, side="left")
    anom_sorted = np.sort(score[label == 1])
    a_pred = len(anom_sorted) - np.searchsorted(anom_sorted, thr, side="left")

    # points that can carry an extended label: the l segments of the largest window
    r_idx = np.concatenate([np.arange(x, y + 1) for x, y in zip(la, lb)])
    R = len(r_idx)
    pos = np.full(n, -1, dtype=np.int64)
    pos[r_idx] = np.arange(R)

    # contributions sqrt(1 - d / w) of every anomaly boundary, for d = 1 .. sliding_window // 2
    h = int(sliding_window) // 2
    S = np.zeros((W, thre))
    if h > 0:
        d = np.arange(1, h + 1)
        after = ends[:, None] + d[None, :]
        before = starts[:, None] - d[None, :]
        tgt = np.concatenate([after.ravel(), before.ravel()])
        dd = np.concatenate([np.broadcast_to(d, after.shape).ravel(), np.broadcast_to(d, before.shape).ravel()])
        ok = (tgt >= 0) & (tgt < n)
        tgt, dd = pos[tgt[ok]], dd[ok]

        w = torch.arange(W, device=device, dtype=torch.float32)[:, None]
        tgt_t = torch.as_tensor(tgt, device=device)
        dd_t = torch.as_tensor(dd, device=device, dtype=torch.float32)[None, :]
        ext = torch.zeros((W, R), device=device, dtype=torch.float32)
        for c in range(0, len(tgt), _CHUNK):
            dc = dd_t[:, c:c + _CHUNK]
            wt = torch.where(dc <= torch.floor(w / 2), torch.sqrt(torch.clamp(1 - dc / torch.clamp(w, min=1), min=0)), 0.0)
            ext.index_add_(1, tgt_t[c:c + _CHUNK], wt)
        ext.clamp_(max=1.0)
        normal = torch.as_tensor(label[r_idx] == 0, device=device)
        ext *= normal[None, :]

        # thresholds descend, so score >= thr[j] holds exactly for j >= first[r] (decided in float64)
        first = thre - np.searchsorted(thr[::-1], score[r_idx], side="right")
        first_t = torch.as_tensor(first, device=device)
        jj = torch.arange(thre, device=device)
        acc = torch.zeros((W, thre), device=device, dtype=torch.float32)
        for c in range(0, R, _CHUNK):
            pred = (jj[None, :] >= first_t[c:c + _CHUNK, None]).to(torch.float32)
            acc += ext[:, c:c + _CHUNK] @ pred
        S = acc.cpu().double().numpy()

    # existence: merged segments of each window holding a predicted point
    existence = np.zeros((W, thre))
    n_seg = np.zeros(W)
    for win in range(W):
        a, b = _merged(starts, ends, win, n)
        m = _range_max(score, a, b)
        existence[win] = (m[:, None] >= thr[None, :]).sum(axis=0)
        n_seg[win] = len(a)

    TP = a_pred[None, :] + S
    N_labels = P + S
    FP = n_pred[None, :] - TP
    P_new = (P + N_labels) / 2
    recall = np.minimum(TP / P_new, 1)
    TPR = recall * existence / n_seg[:, None]
    FPR = FP / (n - P_new)
    prec = TP / n_pred[None, :]

    tpr = np.concatenate([np.zeros((W, 1)), TPR, np.ones((W, 1))], axis=1)
    fpr = np.concatenate([np.zeros((W, 1)), FPR, np.ones((W, 1))], axis=1)
    auc = np.sum((fpr[:, 1:] - fpr[:, :-1]) * (tpr[:, 1:] + tpr[:, :-1]) / 2, axis=1)
    ap = np.sum((tpr[:, 1:-1] - tpr[:, :-2]) * prec, axis=1)
    if return_curves:
        return auc.mean(), ap.mean(), auc, ap
    return auc.mean(), ap.mean()
