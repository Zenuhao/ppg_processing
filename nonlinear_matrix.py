from embed_params import build_embedding
import numpy as np

def simplex_predict_error(x, m=6, tau=2, k=None):
    """
    Simplex Projection による局所予測誤差（平均絶対誤差）
    """
    X, y = build_embedding(x, m=m, tau=tau)
    if X is None:
        return np.nan

    N = len(y)
    if k is None:
        k = m + 1

    errs = []
    for i in range(N):
        d = np.linalg.norm(X[i] - X, axis=1)
        d[i] = np.inf  #自分自身は除外
        if np.all(~np.isfinite(d)):
            continue
        idx = np.argsort(d)[:k]
        d_neighbors = d[idx]
        d0 = np.min(d_neighbors) + 1e-12 #0割を防ぐために1e-12を足す
        w = np.exp(-d_neighbors / d0)
        w_sum = np.sum(w)
        if w_sum <= 0:
            continue
        w /= w_sum
        yhat = np.dot(w, y[idx])
        errs.append(np.abs(y[i] - yhat))

    if len(errs) == 0:
        return np.nan
    return float(np.mean(errs))

def compute_D2_grassberger(x, m=6, tau=2, r_bins=20):
    """
    短いセグメントでの相関次元D2推定
    Grassberger–Procaccia法の簡易実装
    """
    X, _ = build_embedding(x, m=m, tau=tau)
    if X is None:
        return np.nan

    dists = pdist(X)
    if len(dists) == 0:
        return np.nan

    r_min = np.percentile(dists, 5)
    r_max = np.percentile(dists, 95)
    if r_min <= 0 or r_max <= 0 or r_min >= r_max:
        return np.nan

    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), r_bins)
    C_vals = []
    for r in r_vals:
        C = np.mean(dists < r)
        C_vals.append(C)
    C_vals = np.array(C_vals)

    valid = (C_vals > 0) & (C_vals < 1)
    if np.sum(valid) < 5:
        return np.nan

    log_r = np.log(r_vals[valid])
    log_C = np.log(C_vals[valid])

    n = len(log_r)
    i1 = n // 4
    i2 = 3 * n // 4
    if i2 - i1 < 3:
        return np.nan

    coeff = np.polyfit(log_r[i1:i2], log_C[i1:i2], 1)
    D2 = coeff[0]
    return float(D2)

def compute_L1_rosenstein(x, fs, m=6, tau=2, max_t=2.0):
    """
    Rosenstein法による最大リアプノフ指数L1推定
    """
    X, _ = build_embedding(x, m=m, tau=tau)
    if X is None:
        return np.nan

    N = X.shape[0]
    if N < 2 * m:
        return np.nan

    theiler = m * tau  # Theiler window
    nn_idx = np.full(N, -1, dtype=int)

    for i in range(N):
        d = np.linalg.norm(X[i] - X, axis=1)
        left = max(0, i - theiler)
        right = min(N, i + theiler + 1)
        d[left:right] = np.inf
        j = np.argmin(d)
        if not np.isfinite(d[j]):
            continue
        nn_idx[i] = j

    max_k = int(max_t * fs)
    if max_k < 2:
        return np.nan

    l_sum = np.zeros(max_k, dtype=float)
    count = np.zeros(max_k, dtype=float)

    for i in range(N):
        j = nn_idx[i]
        if j < 0:
            continue
        kmax_i = min(max_k, N - i, N - j)
        if kmax_i <= 1:
            continue
        for k in range(kmax_i):
            dist = np.linalg.norm(X[i + k] - X[j + k])
            if dist <= 0:
                continue
            l_sum[k] += np.log(dist)
            count[k] += 1

    valid = count > 0
    if not np.any(valid):
        return np.nan

    t = np.arange(max_k)[valid] / fs
    y = l_sum[valid] / count[valid]

    # 立ち上がり部分（最初の1/3）で線形フィット
    n = len(t)
    if n < 4:
        return np.nan
    k_end = max(3, n // 3)

    coeff = np.polyfit(t[:k_end], y[:k_end], 1)
    L1 = coeff[0]
    return float(L1)

def pdist(x):
    return x