from __future__ import annotations
import numpy as np

def knn_indices(X_lib: np.ndarray, q: np.ndarray, k: int):
    """
    kNN search with sklearn fallback to brute force.
    Returns (idx, dist).
    """
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
    
    