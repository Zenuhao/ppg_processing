from __future__ import annotations
import numpy as np

def as_1d(x):
    x = np.asarray(x, dtype=float)
    return x.reshape(-1)

def embed_delay(x: np.ndarray, m: int, tau: int):
    x = as_1d(x)
    if m < 1 or tau < 1:
        raise ValueError("m>=1 and tau>=1 are required.")
    N = len(x)
    N_embed = N - (m - 1) * tau
    if N_embed <= 0:
        raise ValueError("Time series too short for given (m,tau).")
    
    E = np.empty((N_embed, m), dtype=float)

    for j in range(m):
        shift = j * tau
        E[:, j] = x[(m-1) * tau - shift : : (m - 1) * tau - shift + N_embed]

    t_idx = np.arange((m - 1) * tau, (m - 1) * tau + N_embed)
    return E, t_idx

def build_embedding(x, m, tau):
    """
    時系列 x から遅延座標埋め込みベクトルを構成する。
    戻り値:
        X : shape (N_embed, m)
        idx : 各ベクトルの最後のインデックス
    """
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

