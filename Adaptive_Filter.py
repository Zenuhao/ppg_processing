import numpy as np

def anc_lms_ppg_acc(
    ppg,
    acc,
    mu=1e-3,
    normalize_acc=True,
    return_w_history=True,
):
    """
    ACC を参照信号とする ANC (Adaptive Noise Canceller) による
    PPG の MA 除去（LMS版）

    モデル:
        PPG(n) = clean(n) + w(n)^T * ACC(n) + v(n)

    ここで:
        MA_hat(n) = w(n)^T * ACC(n)
        clean_hat(n) = PPG(n) - MA_hat(n) = e(n)

    Parameters
    ----------
    ppg : array-like, shape (N,)
        観測PPG
    acc : array-like, shape (N,3)
        3軸加速度
    mu : float
        LMSステップサイズ
    normalize_acc : bool
        ACC各軸を標準偏差で割って正規化するかどうか
    return_w_history : bool
        True なら w(n) の履歴を返す

    Returns
    -------
    ppg_clean : (N,)
    ma_est    : (N,)
    w_hist    : (N,3) or None
    """
    ppg = np.asarray(ppg, dtype=float)
    acc = np.asarray(acc, dtype=float)

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("acc must be shape (N,3)")
    if len(ppg) != acc.shape[0]:
        raise ValueError("ppg and acc must have the same length")

    N = len(ppg)

    # ACC正規化
    acc_proc = acc.copy()
    if normalize_acc:
        std = acc_proc.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        acc_proc /= std

    # フィルタ係数: w(n) = [wx, wy, wz]^T
    w = np.zeros(3, dtype=float)

    ma_est = np.zeros(N, dtype=float)
    ppg_clean = np.zeros(N, dtype=float)
    w_hist = np.zeros((N, 3), dtype=float) if return_w_history else None

    for n in range(N):
        x_n = acc_proc[n]                 # (3,)
        y_n = np.dot(w, x_n)              # = 推定MA(n)
        e_n = ppg[n] - y_n                # = 推定clean(n)

        # LMS 更新: w(n+1) = w(n) + mu * e(n) * x(n)
        w = w + mu * e_n * x_n

        ma_est[n] = y_n
        ppg_clean[n] = e_n
        if return_w_history:
            w_hist[n] = w

    return ppg_clean, ma_est, w_hist

def anc_nlms_ppg_acc(
    ppg,
    acc,
    mu=0.5,
    eps=1e-8,
    normalize_acc=True,
    return_w_history=True,
):
    """
    ACC を参照信号とする ANC による PPG の MA 除去（NLMS版）

    更新式:
        w(n+1) = w(n) + (mu / (||x(n)||^2 + eps)) * e(n) * x(n)

    他の引数・返り値は anc_lms_ppg_acc と同様
    """
    ppg = np.asarray(ppg, dtype=float)
    acc = np.asarray(acc, dtype=float)

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("acc must be shape (N,3)")
    if len(ppg) != acc.shape[0]:
        raise ValueError("ppg and acc must have the same length")

    N = len(ppg)

    acc_proc = acc.copy()
    if normalize_acc:
        std = acc_proc.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        acc_proc /= std

    w = np.zeros(3, dtype=float)

    ma_est = np.zeros(N, dtype=float)
    ppg_clean = np.zeros(N, dtype=float)
    w_hist = np.zeros((N, 3), dtype=float) if return_w_history else None

    for n in range(N):
        x_n = acc_proc[n]
        y_n = np.dot(w, x_n)
        e_n = ppg[n] - y_n

        norm2 = np.dot(x_n, x_n) + eps
        w = w + (mu / norm2) * e_n * x_n

        ma_est[n] = y_n
        ppg_clean[n] = e_n
        if return_w_history:
            w_hist[n] = w

    return ppg_clean, ma_est, w_hist

def anc_rls_ppg_acc(
    ppg,
    acc,
    lam=0.99,
    delta=1e3,
    normalize_acc=True,
    return_w_history=True,
):
    """
    ACC を参照信号とする ANC による PPG の MA 除去（RLS版）

    コスト関数:
        J = sum_{k=0}^n lam^{n-k} e(k)^2

    更新式:
        K(n) = P(n-1)x(n) / (lam + x(n)^T P(n-1) x(n))
        w(n) = w(n-1) + K(n) e(n)
        P(n) = (1/lam)[P(n-1) - K(n)x(n)^T P(n-1)]

    Parameters
    ----------
    lam : float
        忘却係数 (0<lam<=1)
    delta : float
        初期共分散 P(0) = delta * I のスケール
    """
    ppg = np.asarray(ppg, dtype=float)
    acc = np.asarray(acc, dtype=float)

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("acc must be shape (N,3)")
    if len(ppg) != acc.shape[0]:
        raise ValueError("ppg and acc must have the same length")

    N = len(ppg)

    acc_proc = acc.copy()
    if normalize_acc:
        std = acc_proc.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        acc_proc /= std

    # 初期値
    w = np.zeros(3, dtype=float)
    P = delta * np.eye(3, dtype=float)

    ma_est = np.zeros(N, dtype=float)
    ppg_clean = np.zeros(N, dtype=float)
    w_hist = np.zeros((N, 3), dtype=float) if return_w_history else None

    for n in range(N):
        x_n = acc_proc[n].reshape(-1, 1)  # (3,1)

        # y_hat = w^T x
        y_n = float(w @ x_n[:, 0])
        e_n = ppg[n] - y_n

        # K(n)
        Px = P @ x_n                        # (3,1)
        denom = lam + float(x_n.T @ Px)     # スカラー
        K = Px / denom                      # (3,1)

        # w(n)
        w = w + (K[:, 0] * e_n)

        # P(n)
        P = (P - K @ (x_n.T @ P)) / lam

        ma_est[n] = y_n
        ppg_clean[n] = e_n
        if return_w_history:
            w_hist[n] = w

    return ppg_clean, ma_est, w_hist

def anc_kalman_ppg_acc(
    ppg,
    acc,
    q_var=1e-7,
    r_var=1e-3,
    normalize_acc=True,
    return_w_history=True,
):
    """
    ACC を参照信号とする ANC による PPG の MA 除去（Kalman版）

    状態: w(n) (3次元)
        w(n+1) = w(n) + q(n),  q ~ N(0, Q=q_var*I)
    観測: PPG(n)
        PPG(n) = x(n)^T w(n) + r(n),  r ~ N(0, R=r_var)

    Parameters
    ----------
    q_var : float
        状態ノイズ共分散のスケール (Q = q_var * I)
        → 大きいと w(n) が時間変化に素早く追従
    r_var : float
        観測ノイズ分散 R
        → 大きいと観測がノイジーだとみなされる

    他の引数・返り値は他の関数と同様
    """
    ppg = np.asarray(ppg, dtype=float)
    acc = np.asarray(acc, dtype=float)

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("acc must be shape (N,3)")
    if len(ppg) != acc.shape[0]:
        raise ValueError("ppg and acc must have the same length")

    N = len(ppg)

    acc_proc = acc.copy()
    if normalize_acc:
        std = acc_proc.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        acc_proc /= std

    # 初期状態
    w = np.zeros(3, dtype=float)
    P = np.eye(3, dtype=float) * 1e3  # 大きめの初期共分散

    Q = q_var * np.eye(3, dtype=float)
    R = r_var

    ma_est = np.zeros(N, dtype=float)
    ppg_clean = np.zeros(N, dtype=float)
    w_hist = np.zeros((N, 3), dtype=float) if return_w_history else None

    for n in range(N):
        x_n = acc_proc[n].reshape(-1, 1)  # (3,1)

        # ---- 予測ステップ ----
        w_pred = w                         # w(n|n-1)
        P_pred = P + Q                     # P(n|n-1)

        # ---- 更新ステップ ----
        # 観測予測: y_hat = x^T w_pred
        y_pred = float(w_pred @ x_n[:, 0])
        # イノベーション
        innov = ppg[n] - y_pred

        # S = x^T P_pred x + R
        S = float(x_n.T @ (P_pred @ x_n)) + R
        # Kalman gain K = P_pred x / S
        K = (P_pred @ x_n) / S            # (3,1)

        # 状態更新
        w = w_pred + (K[:, 0] * innov)
        # 共分散更新
        P = P_pred - K @ (x_n.T @ P_pred)

        # 出力
        ma_est[n] = float(w @ x_n[:, 0])
        ppg_clean[n] = ppg[n] - ma_est[n]
        if return_w_history:
            w_hist[n] = w

    return ppg_clean, ma_est, w_hist
