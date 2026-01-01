import numpy as np

def build_clean_attractor(sdppg_segments, is_ma_flags, m=6, tau=2, max_points=50000):
    """
    MAの影響が小さい区間を埋め込んだアトラクターを構成
    """
    X_list = []
    count = 0
    for seg, is_ma in zip(sdppg_segments, is_ma_flags):
        if is_ma:
            continue
        X, _ = build_embedding(seg, m=m, tau=tau)
        if X is None:
            continue
        X_list.append(X)
        count += X.shape[0]
        if count >= max_points:
            break

    if not X_list:
        return None

    X_clean = np.vstack(X_list)
    if X_clean.shape[0] > max_points:
        idx = np.random.choice(X_clean.shape[0], max_points, replace=False)
        X_clean = X_clean[idx]
    return X_clean


def _project_embedding_onto_clean(X_ma, X_clean, k=10):
    """
    MAセグメントの埋め込み点X_maを, クリーンなアトラクターX_cleanに射影
    """
    N = X_ma.shape[0]
    X_proj = np.zeros_like(X_ma)
    for i in range(N):
        d = np.linalg.norm(X_ma[i] - X_clean, axis=1)
        idx = np.argsort(d)[:k]
        d_neighbors = d[idx]
        d0 = np.min(d_neighbors) + 1e-12
        w = np.exp(-d_neighbors / d0)
        w_sum = np.sum(w)
        if w_sum <= 0:
            X_proj[i] = X_ma[i]
        else:
            w /= w_sum
            X_proj[i] = np.dot(w, X_clean[idx])
    return X_proj


def reconstruct_series_from_embedding(X_proj, L_original, m=6, tau=2):
    """
    埋め込み空間の投影結果 X_proj から 1次元の時系列を再構成する簡易版
    各時刻の第1成分を代表値とみなす。
    """
    N = X_proj.shape[0]
    x_hat = np.zeros(L_original, dtype=float)
    x_hat[:N] = X_proj[:, 0]
    if L_original > N:
        x_hat[N:] = x_hat[N - 1]
    return x_hat


def correct_ma_segments(sdppg_segments,
                        is_ma_flags,
                        m=6,
                        tau=2,
                        k=10,
                        alpha=0.7,
                        max_points=50000):
    """
    ステップ4「LLE ≤ 0 制約に基づく軌道の変形」を
    クリーンアトラクタへの投影という形で簡易実装。

    sdppg_segments: list of 1D arrays
    is_ma_flags:    同じ長さのbool配列（TrueがMAセグメント）

    戻り値: corrected_segments（同じ長さのlist）
    """
    sdppg_segments = [np.asarray(seg, dtype=float) for seg in sdppg_segments]
    is_ma_flags = np.asarray(is_ma_flags, dtype=bool)

    X_clean = build_clean_attractor(sdppg_segments, is_ma_flags,
                                    m=m, tau=tau, max_points=max_points)
    if X_clean is None:
        # クリーン区間が無い場合は何も修正できない
        return sdppg_segments

    corrected = []
    for seg, is_ma in zip(sdppg_segments, is_ma_flags):
        if not is_ma:
            corrected.append(seg.copy())
            continue

        X_ma, _ = build_embedding(seg, m=m, tau=tau)
        if X_ma is None:
            corrected.append(seg.copy())
            continue

        X_proj = _project_embedding_onto_clean(X_ma, X_clean, k=k)
        x_hat = reconstruct_series_from_embedding(X_proj,
                                                  L_original=len(seg),
                                                  m=m,
                                                  tau=tau)
        # 元の系列とブレンド（alpha: 補正の強さ）
        seg_corr = alpha * x_hat + (1.0 - alpha) * seg
        corrected.append(seg_corr)

    return corrected