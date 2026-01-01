import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KDTree

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

"""
def build_embedding(x, m=6, tau=2):

    時系列 x を m次元・遅延tauで埋め込み
    返り値:
        X : shape (N, m)
        y : 1ステップ先の値 (shape (N,))
        
    x = np.asarray(x, dtype=float)
    L = len(x)
    N = L - m * tau
    if N <= m + 1:
        return None, None

    X = np.zeros((N, m), dtype=float)
    for j in range(m):
        X[:, j] = x[j * tau : j * tau + N]
    y = x[m * tau : m * tau + N]
    return X, y
"""

def mutual_information(x, y, n_bins=32):
    """
    雑なヒストグラムベースの相互情報量推定
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    H, _, _ = np.histogram2d(x, y, bins=n_bins)
    Pxy = H / np.sum(H)
    Px = np.sum(Pxy, axis=1)
    Py = np.sum(Pxy, axis=0)

    eps = 1e-12
    Pxy = Pxy + eps
    Px = Px + eps
    Py = Py + eps

    mi = 0.0
    for i in range(Pxy.shape[0]):
        for j in range(Pxy.shape[1]):
            mi += Pxy[i, j] * np.log(Pxy[i, j] / (Px[i] * Py[j]))
    return mi

def estimate_delay_ami(x, max_lag_samples=64, n_bins=32, smooth_window=3):
    """
    AMI(average mutual information)に基づく遅延時間 tau の推定
    「最初の局所最小」を採用
    見つからなければ最小MIのlagを返す
    Fraser and Swinney 1985
    """
    x = np.asarray(x, dtype=float)
    L = len(x)

    if max_lag_samples >= L//2:
        max_lag_samples = max(1, L//2 - 1)

    lags = np.arange(1, max_lag_samples+1)
    mis = []

    L_eff = L - max_lag_samples
    x0 = x[:L_eff]

    for lag in lags:
        xlag = x[lag:lag + L_eff]
        mi = mutual_information(x0, xlag, n_bins=n_bins)
        mis.append(mi)

    mis = np.array(mis)

    mis_s = mis.copy()
    if smooth_window > 1 and len(mis) >= smooth_window:

        kernel = np.ones(smooth_window) / smooth_window
        mis_s = np.convolve(mis, kernel, mode="same")

    for i in range(1, len(mis_s)-1):
        if mis_s[i] < mis_s[i-1] and mis_s[i] <= mis_s[i+1]:
            return int(lags[i])
    
    cut = int(0.9*len(mis_s))

    if cut < 1:
        idx_min = int(np.argmin(mis_s))
    else:
        idx_min = int(np.argmin(mis_s[:cut]))
    
    return int(lags[idx_min])
    """
    x = np.asarray(x, dtype=float)
    mis = []
    lags = range(1, max_lag_samples + 1)
    for lag in lags:
        mi = mutual_information(x[:-lag], x[lag:], n_bins=n_bins)
        mis.append(mi)
    mis = np.array(mis)

    # 最初の局所最小
    for i in range(1, len(mis) - 1):
        if mis[i] < mis[i - 1] and mis[i] < mis[i + 1]:
            return lags[i]

    # 局所最小が見つからない場合は最小値
    return int(lags[np.argmin(mis)])
    """


def estimate_embedding_dimension_fnn(x, tau, m_max=10, r_tol=15.0, a_tol=2.0, theiler=0, m_min=2):
    """
    False Nearest Neighbors 法による埋め込み次元推定
    Kennel et al 1992
    """

    x = np.asarray(x, dtype=float)
    L = len(x)
    if tau < 0:
        raise ValueError("tau must be positive")
    
    if L < (m_max + 2) * tau:
        m_max = max(m_min+1, min(m_max, L // tau - 2))
        if m_max < m_min:
            return m_min
        
    x = (x-np.mean(x))/(np.std(x)+1e-12)
    A = np.std(x)+1e-12 #attractor size

    def fnn_fraction(m):
        X, _ = build_embedding(x, m=m, tau=tau)
        Xp, _ = build_embedding(x, m=m+1, tau=tau)

        if X is None or Xp is None:
            return 1.0
        
        N = min(X.shape[0], Xp.shape[0])
        if N < 2:
            return 1.0
        
        X = X[:N]
        Xp = Xp[:N]
        
        k = max(2+theiler, 5)
        tree = KDTree(X, leaf_size=40)
        dist, idx=tree.query(X, k=k)

        Rm_list = []
        nn_idx_list = []

        for i in range(N):
            for d, j in zip(dist[i], idx[i]):

                if j == i:
                    continue
                if theiler > 0 and abs(i-j) <= theiler:
                    continue

                Rm_list.append(d)
                nn_idx_list.append(j)
                break

        if len(Rm_list) < 2:
            return 1.0
        
        Rm = np.array(Rm_list)
        nn_idx = np.array(nn_idx_list, dtype=int)

        # 念のため：Xpの範囲外インデックスを捨てる
        valid = (nn_idx >= 0) & (nn_idx < Xp.shape[0])
        if np.sum(valid) < 2:
            return 1.0
        Rm = Rm[valid]
        nn_idx = nn_idx[valid]

        dist_p = np.linalg.norm(Xp - Xp[nn_idx], axis = 1)

        cond1 = (Rm > 0) & (dist_p / Rm > r_tol)
        cond2 = dist_p / A > a_tol

        return np.mean(cond1 | cond2)
    
    fractions = []
    for m in range(1, m_max):
        frac = fnn_fraction(m)
        fractions.append((m, frac))

    for m, frac in fractions:
        if m >= m_min and frac < 0.1:
            return m
        
    m_best, _ = min(
        ((m, frac) for m, frac in fractions if m >= m_min),
        key=lambda t: t[1]
    )

    return m_best
    
    """
    #N×N matrix
    
    def fnn_fraction(m):
        X, _ = build_embedding(x, m=m, tau=tau)
        if X is None:
            return 1.0
        N = X.shape[0]
        dist = np.linalg.norm(X[None, :, :] - X[:, None, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        nn_idx = np.argmin(dist, axis=1)
        Rm = dist[np.arange(N), nn_idx]

        Xp, _ = build_embedding(x, m=m+1, tau=tau)
        if Xp is None or Xp.shape[0] < N:
            return 1.0
        dist_p = np.linalg.norm(Xp - Xp[nn_idx, :], axis=1)
        A = np.max(np.linalg.norm(Xp, axis=1))

        cond1 = (Rm > 0) & (dist_p / Rm > r_tol)
        cond2 = dist_p / A > a_tol
        fnn = np.sum(cond1 | cond2)
        return fnn / float(N)

    #KDTree
    def fnn_fraction(m):
        # d 次元埋め込み
        # (d+1) 次元埋め込み
        X, _ = build_embedding(x, m=m, tau=tau)
        Xp, _ = build_embedding(x, m=m+1, tau=tau)

        if X is None or Xp is None:
            return 1.0
        
        N = min(X.shape[0], Xp.shape[0])
        if N < 2:
            return 1.0
        
        X = X[:N]
        Xp = Xp[:N]

        # KD-tree による最近傍探索（自分 + 最近傍 → k=2）
        tree = KDTree(X, leaf_size=40)
        dist, idx = tree.query(X, k=2)

        # 自分を除いた最近傍（2番目）
        Rm = dist[:, 1]
        nn_idx = idx[:, 1]

        # (d+1) 次元での距離
        dist_p = np.linalg.norm(Xp - Xp[nn_idx], axis=1)

        # アトラクタサイズ（最大ノルム）
        A = np.max(np.linalg.norm(Xp, axis=1))
        if A == 0:
            return 1.0

        # Kennel の2条件
        cond1 = (Rm > 0) & (dist_p / Rm > r_tol)
        cond2 = dist_p / A > a_tol
        return np.mean(cond1 | cond2)

    fractions = []
    for m in range(1, m_max):
        frac = fnn_fraction(m)
        fractions.append((m, frac))

    for m, frac in fractions:
        if frac < 0.1:
            return m
        
    m_best = min(fractions, key=lambda  t: t[1])[0]
    return m_best
    """

def choose_embedding_params(x, fs, max_lag_sec=2.0, m_max=10):
    """
    x から (m, tau) を自動推定する関数
    """
    max_lag_samples = int(max_lag_sec * fs)
    tau = estimate_delay_ami(x, max_lag_samples=max_lag_samples)
    m = estimate_embedding_dimension_fnn(x, tau=tau, m_max=m_max)
    return m, tau

def ami_curve_hist(x, max_lag_samples, n_bins=32, smooth_window=1):
    """
    AMI曲線 I(tau) を返す（ヒストグラム2D）
    returns: lags (1..max_lag), mis, mis_smooth
    """
    x = np.asarray(x, dtype=float)
    L = len(x)
    max_lag_samples = int(max_lag_samples)
    max_lag_samples = min(max_lag_samples, max(1, L//2-1))

    lags = np.arange(1, max_lag_samples + 1)
    L_eff = L - max_lag_samples
    x0 = x[:L_eff]

    mis = []
    for lag in lags:
        xlag = x[lag:lag+L_eff]
        mi = mutual_information(x0, xlag, n_bins=n_bins)
        mis.append(mi)
    
    mis = np.asarray(mis, dtype=float)
    
    mis_s = mis
    if smooth_window and smooth_window > 1 and len(mis) >= smooth_window:
        w = int(smooth_window)
        k = np.ones(w, dtype=float)/w
        mis_s = np.convolve(mis, k, mode="same")

    return lags, mis, mis_s

def fnn_curve(x, tau, m_max=10, r_tol=15.0, a_tol=2.0, theiler=0):
    """
    FNN曲線 FNN(m) を返す
    returns: ms (1..m_max-1), fnn_fracs
    """
    ms = np.arange(1, m_max, dtype=int)
    fracs = []
    for m in ms:
        frac = fnn_fraction_single(x, tau, m, r_tol=r_tol, a_tol=a_tol, theiler=theiler)
        fracs.append(frac)
    return ms, np.asarray(fracs, dtype=float)

def fnn_fraction_single(x, tau, m, r_tol=15, a_tol=2, theiler=0):

    x = np.asarray(x, dtype=float)
    tau = int(tau)

    x = (x - np.mean(x)) / (np.std(x)+1e-12)
    R_A = np.std(x) + 1e-12
    
    X, _ = build_embedding(x, m=m, tau=tau)
    Xp, _ = build_embedding(x, m=m+1, tau=tau)

    if X is None or Xp is None:
        return 1.0

    N = min(X.shape[0], Xp.shape[0])
    if N < 3:
        return 1.0

    X  = X[:N]
    Xp = Xp[:N]

    k = max(5, 2 + int(theiler))
    tree = KDTree(X, leaf_size=40)
    dist, idx = tree.query(X, k=k)

    Rm_list = []
    nn_list = []

    for i in range(N):
        for d, j in zip(dist[i], idx[i]):
            if j == i:
                continue
            if theiler > 0 and abs(i-j) <= theiler:
                continue
            Rm_list.append(d)
            nn_list.append(j)
            break

    if len(Rm_list) < 3:
        return 1.0

    Rm = np.asarray(Rm_list, dtype=float)
    nn = np.asarray(nn_list, dtype=int)

    valid = (nn >= 0) & (nn < N)
    if np.sum(valid) < 3:
        return 1.0
    Rm = Rm[valid]
    nn = nn[valid]

    Rm1 = np.linalg.norm(Xp - Xp[nn], axis=1)

    cond1 = (Rm > 0) & (Rm1 / Rm > r_tol)
    cond2 = (Rm1 / R_A) > a_tol
    return float(np.mean(cond1 | cond2))

def save_ami_fnn_plots(x, fs, out_prefix, max_lag_sec=2.0, n_bins=32, smooth_window=5, tau_for_fnn=None, m_max=10, theiler=0):
    x = np.asarray(x, dtype=float)
    max_lag_samples = int(round(max_lag_sec*fs))

    lags, mis, mis_s = ami_curve_hist(x, max_lag_samples, n_bins=n_bins, smooth_window=smooth_window)

    plt.figure()
    plt.plot(lags, mis, label="MsI")
    plt.plot(lags, mis_s, label="MI smoothed")
    plt.xlabel("tau (samples)")
    plt.ylabel("I(tau)")
    plt.title("AMI curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_ami.png", dpi=150)
    plt.close()

    if tau_for_fnn is not None:
        ms, fracs = fnn_curve(x, tau_for_fnn, m_max=m_max, theiler=theiler)
        plt.figure()
        plt.plot(ms, fracs, marker="o")
        plt.xlabel("m")
        plt.ylabel("FNN(m)")
        plt.title(f"FNN curve (tau={tau_for_fnn} samples)")
        plt.tight_layout()
        plt.savefig(out_prefix + "_fnn.png", dpi=150)
        plt.close()

def downsample_for_tau_only(x, fs, fs_est = 50):
    
    x = np.asarray(x, dtype=float)
    q = max(1, int(round(fs/fs_est)))

    fs_ds = fs / q
    if q == 1:
        return x, 1, fs
    
    w = q
    k = np.ones(w, dtype=float) / w
    x_f = np.convolve(x, k, mode="same")
    x_ds = x_f[::q]

    return x_ds, q, fs_ds

def estimate_delay_ami_tau_only_ds(x, fs, fs_est=50, max_lag_sec=2.0, min_lag_sec=0.03, n_bins=32, smooth_window=5, avoid_tail_ratio=0.0):
    x = np.asarray(x, dtype=float)
    x_ds, q, fs_ds = downsample_for_tau_only(x, fs, fs_est=fs_est)

    max_lag_ds = int(round(max_lag_sec * fs_ds))
    max_lag_ds = max(1, max_lag_ds)

    min_lag_ds = int(round(min_lag_sec * fs_ds))
    min_lag_ds = max(1, min_lag_ds)

    lags, _, mis_s = ami_curve_hist(x_ds, max_lag_ds, n_bins=n_bins, smooth_window=smooth_window)

    mask = lags >= min_lag_ds
    lags2 = lags[mask]
    mis2 = mis_s[mask]
    if len(mis2) < 3:
        return 1
    
    tau_ds = None
    for i in range(1, len(mis2)-1):
        if mis2[i] < mis2[i-1] and mis2[i] <= mis2[i+1]:
            tau_ds = int(lags2[i])
            break

    if tau_ds is None:
        cut = int((1.0 - float(avoid_tail_ratio)) * len(mis2))
        cut = max(1, min(cut, len(mis2)))
        tau_ds = int(lags2[int(np.argmin(mis2[:cut]))])

    tau = int(round(tau_ds*q))
    return max(1, tau)

