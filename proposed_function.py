from __future__ import annotations
import numpy as np
from dataclasses import dataclass

class KNNmodel:
    def __init__(self, X: np.ndarray, algorithm: str = "auto"):
        from sklearn.neighbors import NearestNeighbors
        
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim != 2 or self.X.shape[0] < 2:
            raise ValueError("KNN library must have shape (N, m) with N>=2.")
        self.nn = NearestNeighbors(algorithm=algorithm)
        self.nn.fit(self.X)

    def query(self, Q: np.ndarray, k: int, chunk: int = 50000):
        """
        Q: (Nq, m) or (m,)
        returns idx:(Nq,k), dist:(Nq,k)
        chunked to avoid huge memory spikes
        """
        Q = np.asarray(Q, dtype=float)
        if Q.ndim == 1:
            Q = Q.reshape(1,-1)
        
        k = int(k)
        k = min(k, self.X.shape[0])
        if k <= 0:
            raise ValueError("KNN library is empty.")
    
        Nq = Q.shape[0]
        idx_all = np.empty((Nq, k), dtype=int)
        dist_all = np.empty((Nq, k), dtype=float)

        for s in range(0, Nq, chunk):
            e = min(Nq, s + chunk)
            dist, idx = self.nn.kneighbors(Q[s:e], n_neighbors=k, return_distance=True)
            idx_all[s:e] = idx
            dist_all[s:e] = dist

        return idx_all, dist_all
    

def acc_magnitude(acc: np.ndarray | None) -> np.ndarray | None:
    if acc is None:
        return None
    acc = np.asarray(acc, dtype=float)
    if acc.ndim == 2:
        return np.linalg.norm(acc, axis=1)   # (T,)
    return acc.reshape(-1)

def as_1d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.reshape(-1)

def embed_delay(x: np.ndarray, m: int, tau: int):
    """
    Takens delay embedding:
      X_t = [x_t, x_{t-tau}, ..., x_{t-(m-1)tau}]^T

    Returns
    -------
    E : (N_embed, m) ndarray
        Embedding matrix, rows correspond to time indices t_idx.
    t_idx : (N_embed,) ndarray
        Original time indices for the first component x_t (E[:,0]).
    """
    x = as_1d(x)
    if m < 1 or tau < 1:
        raise ValueError("m>=1 and tau>=1 are required.")
    N = len(x)
    
    N_embed = N - (m - 1) * tau
    if N_embed <= 0:
        raise ValueError("Time series too short for given (m,tau).")

    E = np.empty((N_embed, m), dtype=float)
    # E[k,0] = x[t], E[k,1] = x[t-tau], ...
    for j in range(m):
        shift = j * tau
        E[:, j] = x[(m - 1) * tau - shift : (m - 1) * tau - shift + N_embed]

    t_idx = np.arange((m - 1) * tau, (m - 1) * tau + N_embed)
    return E, t_idx

def simplex_weights(dist: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    dist: (Nq,k) sorted ascending
    return w: (Nq,k)
    """
    dist = np.asarray(dist, dtype=float)
    d1 = np.maximum(dist[:, [0]], eps)
    w = np.exp(-dist / d1)
    w = w / (np.sum(w, axis=1, keepdims=True)+eps)
    return w

def simplex_forecast_1step(Y_query: np.ndarray, knn: KNNmodel, y_lib_next: np.ndarray, k: int) -> np.ndarray:
    Y_query = np.asarray(Y_query, float)
    y_lib_next = np.asarray(y_lib_next, float).reshape(-1)

    idx, dist = knn.query(Y_query, k=k)
    w = simplex_weights(dist)
    return np.sum(w * y_lib_next[idx], axis=1)

def simplex_project(Y: np.ndarray, knn: KNNmodel, k: int) -> np.ndarray:
    Y = np.asarray(Y, float)
    idx, dist = knn.query(Y, k=k)
    w = simplex_weights(dist)  # (N,k)
    # Z[i] = sum_j w[i,j] * X_lib[idx[i,j]]
    Z = np.einsum("nk,nkm->nm", w, knn.X[idx])
    return Z

def smap_local_linear_knn(
    X_lib: np.ndarray,
    X_lib_next: np.ndarray,
    q: np.ndarray,
    theta: float,
    knn: KNNmodel,
    k_smap: int = 100,
    ridge: float = 1e-6,
    eps: float = 1e-12,
    normalize_dist: bool = True,
):
    """
    Local weighted linear regression using ONLY k_smap nearest neighbors.
    """

    K = X_lib.shape[0]
    k_smap = min(int(k_smap), K)
    if k_smap <= 1:
        raise ValueError("Too few library points for S-map.")
    
    q = np.asarray(q, float).reshape(1, -1)
    idx, dist = knn.query(q, k=k_smap)
    idx = idx[0]
    d = dist[0]

    Xn = X_lib[idx]         # (K,m)
    Yn = X_lib_next[idx]    # (K,m)
    m = Xn.shape[1]

    if normalize_dist:
        dbar = float(np.mean(d) + eps)
        w = np.exp(-theta * (d / dbar))
    else:
        w = np.exp(-theta * d)

    W = w[:, None]  # (K,1)

    Phi = np.hstack([Xn, np.ones((k_smap, 1))])  # (K,m+1)

    # ridge（切片は正則化しない方が安定しやすい）
    reg = np.diag(np.r_[np.full(m, ridge), 0.0])

    G = Phi.T @ (Phi * W) + reg
    RHS = Phi.T @ (Yn * W)

    B = np.linalg.solve(G, RHS)  # (m+1,m)
    A = B[:m, :].T
    b = B[m, :]
    return A, b

def local_expansion_lambda(A: np.ndarray, dt: float, eps: float = 1e-12) -> float:
    """
    Lyapunov-like local expansion proxy:
      lambda = (1/dt) * log sigma_max(A)
    """
    s = np.linalg.svd(A, compute_uv=False)
    sigma_max = max(float(s[0]), eps)
    return (1.0 / dt) * np.log(sigma_max)

def reconstruct_from_embeddings(Xhat: np.ndarray, t_idx: np.ndarray, m: int, tau: int, T: int) -> np.ndarray:
    """
    Overlap-averaging reconstruction from embeddings.

    With embedding definition:
      Xhat[k, r] corresponds to x[t_idx[k] - r*tau]

    xhat_t = (1/C_t) * sum_{r=0}^{m-1} Xhat_{t+r*tau}[r]  (equivalent overlap)
    """
    Xhat = np.asarray(Xhat, dtype=float)
    t_idx = np.asarray(t_idx, dtype=int)

    x_sum = np.zeros(T, dtype=float)
    x_cnt = np.zeros(T, dtype=float)

    for k, t in enumerate(t_idx):
        for r in range(m):
            tt = t - r * tau
            if 0 <= tt < T:
                x_sum[tt] += Xhat[k, r]
                x_cnt[tt] += 1.0

    return x_sum / np.maximum(x_cnt, 1.0)

@dataclass
class DenoiseParams:
    m: int = 6
    tau: int = 2
    dt: float = 1.0          # seconds per sample
    theta: float = 2.0       # S-map localization
    ridge: float = 1e-5
    simplex_k: int | None = None
    n_iter: int = 5

    # clean set definition (ACC reference)
    acc_thr: float | None = None          # if None and acc given -> use quantile below
    acc_thr_q: float = 0.95               # quantile for auto acc_thr

    # fallback (if acc is None)
    simplex_err_q: float = 0.90

    # MA decision: lambda >= ma_lambda_thr
    ma_lambda_thr: float = 0.0
    ma_exclude_clean: bool = True         # if True: MA only where (~clean_mask) & (lambda>=thr)

    # denoise strength
    proj_strength: float = 1.0            # 0..1
    smooth_gamma: float = 0.0             # Tikhonov smoothing after each iter
   
def denoise_ppg_ma(y: np.ndarray, acc: np.ndarray | None, p: DenoiseParams, k_smap: int = 100):
    y = np.asarray(y, dtype=float).reshape(-1)
    T = len(y)
    x = y.copy()

    # (3) dt は呼び出し側で 1/fs を入れる前提（main.pyで既にOK）
    if not np.isfinite(p.dt) or p.dt <= 0:
        raise ValueError("p.dt must be positive (set dt=1/fs).")

    k_base = p.simplex_k if p.simplex_k is not None else (p.m + 1)
    acc_mag = acc_magnitude(acc)

    diag = {}

    for it in range(p.n_iter):
        X, t_idx = embed_delay(x, p.m, p.tau)
        Y, _ = embed_delay(y, p.m, p.tau)
        N = X.shape[0]

        # -----------------------------
        # (A) define clean_mask
        # -----------------------------
        if acc_mag is not None:
            # (1) threshold from acc_mag[t_idx] (embedding-aligned)
            acc_thr_eff = float(p.acc_thr) if (p.acc_thr is not None) else float(np.quantile(acc_mag[t_idx], p.acc_thr_q))
            clean_mask = acc_mag[t_idx] <= acc_thr_eff
        else:
            # fallback: simplex prediction error based clean selection
            valid_next = (t_idx + 1) < T
            X_lib_fc = X[valid_next]
            y_next = y[t_idx[valid_next] + 1]

            if X_lib_fc.shape[0] < 2:
                # too short -> treat all as clean
                clean_mask = np.ones(N, dtype=bool)
                acc_thr_eff = None
            else:
                knn_fc = KNNmodel(X_lib_fc)
                k_fc = min(int(k_base), X_lib_fc.shape[0])
                yhat_next = simplex_forecast_1step(Y[valid_next], knn_fc, y_next, k=k_fc)

                err = np.abs(y_next - yhat_next)
                thr = np.quantile(err, p.simplex_err_q)
                clean_mask = np.zeros(N, dtype=bool)
                clean_mask[np.where(valid_next)[0]] = (err <= thr)
                acc_thr_eff = None

        # clean library too small? -> fallback to all points
        if clean_mask.sum() < 2:
            clean_mask[:] = True

        # -----------------------------
        # (B) build S-map library from clean points (need next row)
        # -----------------------------
        lib_valid = clean_mask & ((np.arange(N) + 1) < N)
        if lib_valid.sum() < max(50, p.m + 5):
            lib_valid = (np.arange(N) + 1) < N

        X_lib = X[lib_valid]
        X_lib_next = X[np.where(lib_valid)[0] + 1]

        # if still too small -> no MA this iter (keep x)
        if X_lib.shape[0] < 2:
            lamb = np.full(N, -np.inf, dtype=float)
            ma_mask = np.zeros(N, dtype=bool)
            w = np.zeros(N, dtype=float)

            diag[f"iter_{it}"] = {
                "clean_ratio": float(clean_mask.mean()),
                "acc_thr_eff": None if acc_thr_eff is None else float(acc_thr_eff),
                "ma_ratio": 0.0,
                "lambda_thr": float(p.ma_lambda_thr),
                "lambda_mean": float(np.mean(lamb[np.isfinite(lamb)]) if np.any(np.isfinite(lamb)) else -np.inf),
                "lambda_p95": float(np.quantile(lamb[np.isfinite(lamb)], 0.95) if np.any(np.isfinite(lamb)) else -np.inf),
                "note": "X_lib too small -> skipped MA/projection",
            }
            continue

        knn_smap = KNNmodel(X_lib)

        # -----------------------------
        # (C) estimate lambda(t)
        # -----------------------------
        k_smap_eff = min(int(k_smap), X_lib.shape[0])   # (2) clamp here too
        if k_smap_eff < 2:
            k_smap_eff = 2

        lamb = np.empty(N, float)
        for qi in range(N):
            A, _b = smap_local_linear_knn(
                X_lib, X_lib_next, q=X[qi],
                theta=p.theta, knn=knn_smap, k_smap=k_smap_eff,
                ridge=p.ridge, normalize_dist=True
            )
            lamb[qi] = local_expansion_lambda(A, dt=p.dt)

        # -----------------------------
        # (D) MA mask: lambda >= threshold
        # -----------------------------
        ma_mask = (lamb >= float(p.ma_lambda_thr))
        if p.ma_exclude_clean:
            ma_mask = ma_mask & (~clean_mask)

        w = ma_mask.astype(float) * float(p.proj_strength)

        # -----------------------------
        # (E) simplex projection onto clean library
        # -----------------------------
        X_clean_lib = X[clean_mask] if clean_mask.any() else X
        if X_clean_lib.shape[0] < 2:
            # cannot project -> skip
            Xhat = Y.copy()
        else:
            knn_proj = KNNmodel(X_clean_lib)
            k_eff = min(int(k_base), X_clean_lib.shape[0])   # (2) clamp (important!)
            if k_eff < 2:
                k_eff = 2

            Xhat = Y.copy()
            ma_idx = np.where(w > 0)[0]
            if len(ma_idx) > 0:
                Z_ma = simplex_project(Y[ma_idx], knn_proj, k=k_eff)
                Xhat[ma_idx] = (1.0 - w[ma_idx, None]) * Y[ma_idx] + w[ma_idx, None] * Z_ma

        # -----------------------------
        # (F) reconstruct + smooth
        # -----------------------------
        x = reconstruct_from_embeddings(Xhat, t_idx, p.m, p.tau, T)
        if p.smooth_gamma > 0:
            x = smooth_tikhonov_1st(x, gamma=float(p.smooth_gamma))

        diag[f"iter_{it}"] = {
            "clean_ratio": float(clean_mask.mean()),
            "acc_thr_eff": None if acc_thr_eff is None else float(acc_thr_eff),
            "ma_ratio": float(ma_mask.mean()),
            "lambda_thr": float(p.ma_lambda_thr),
            "lambda_mean": float(np.mean(lamb)),
            "lambda_p95": float(np.quantile(lamb, 0.95)),
            "k_eff": int(min(int(k_base), X_clean_lib.shape[0])),
            "k_smap_eff": int(k_smap_eff),
        }

    info = {"diagnostics": diag, "last_lambda": lamb, "t_idx_embed": t_idx}
    return x, info

def smooth_tikhonov_1st(y: np.ndarray, gamma: float) -> np.ndarray:
    """
    Solve: min_x ||x-y||^2 + gamma * sum (x_{t+1}-x_t)^2
    => (I + gamma D^T D)x = y (tridiagonal)
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    if gamma <= 0:
        return y.copy()

    n = len(y)
    a = np.full(n - 1, -gamma, dtype=float)
    b = np.full(n, 1 + 2 * gamma, dtype=float)
    c = np.full(n - 1, -gamma, dtype=float)
    b[0] = 1 + gamma
    b[-1] = 1 + gamma

    # Thomas algorithm
    cp = np.zeros(n - 1, dtype=float)
    dp = np.zeros(n, dtype=float)

    cp[0] = c[0] / b[0]
    dp[0] = y[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (y[i] - a[i - 1] * dp[i - 1]) / denom

    dp[n - 1] = (y[n - 1] - a[n - 2] * dp[n - 2]) / (b[n - 1] - a[n - 2] * cp[n - 2])

    x = np.zeros(n, dtype=float)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


"""
def embedding(x, m, tau):
    
    時系列 x から遅延座標埋め込みベクトルを構成する。
    戻り値:
        X : shape (N_embed, m)
        idx : 各ベクトルの最後のインデックス
    
    x = np.asarray(x, dtype=float)
    N = len(x)
    if m < 1 or tau < 1:
        return None, None

    N_embed = N - (m - 1) * tau
    if N_embed <= 0:
        return None, None

    X = np.empty((N_embed, m))
    for j in range(m):
        X[:, j] = x[j * tau : j * tau + N_embed]
    idx = np.arange((m - 1) * tau, (m - 1) * tau + N_embed)
    return X, idx


def knn_indices(X_lib: np.ndarray, q: np.ndarray, k: int):
    
    X_lib = np.asarray(X_lib, dtype=float)
    q = np.asarray(q, dtype=float).reshape(1, -1)
    k = int(k)

    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(X_lib)
        dist, idx = nn.kneighbors(q, return_distance=True)
        return idx[0], dist[0]
    except Exception:
        d = np.linalg.norm(X_lib - q, axis=1)
        idx = np.argsort(d)[:k]
        return idx, d[idx]
"""


