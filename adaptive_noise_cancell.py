import numpy as np

def prep_ppg(ppg, demean=True):
    x = np.asarray(ppg, dtype=float)
    if demean:
        x = x - float(np.mean(x))
    return x

def prep_acc(acc, demean=True, normalize=True):
    a = np.asarray(acc, dtype=float).copy()
    if demean:
        a -= a.mean(axis=0, keepdims=True)   # ★重力/姿勢DC除去
    if normalize:
        std = a.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        a /= std
    return a

def make_tapped_regressor(acc_proc, n_taps):
    """
    acc_proc: (N,3)
    return X: (N, 3*n_taps), row n = [acc[n], acc[n-1], ..., acc[n-n_taps+1]]
    """
    N = acc_proc.shape[0]
    L = int(n_taps)
    if L < 1:
        raise ValueError("n_taps must be >= 1")
    X = np.zeros((N, 3*L), dtype=float)
    for k in range(L):
        X[k:, 3*k:3*k+3] = acc_proc[:N-k, :]
    return X

def anc_lms_ppg_acc(
    ppg,
    acc,
    mu=1e-3,
    n_taps = 16,
    demean_ppg = True,
    demean_acc = True,
    normalize_acc=True,
    update_gate = 0.0,
    return_w_history=True,
):
    """
    ACCを参照信号とするANCによるPPGのMA除去, LMS
    
    モデル：PPG(n) = clean(n) + w(n)^T * ACC(n) + v(n)

    MA(n) = w(n)^T * ACC(n)
    clean(n) = PPG(n) - MA(n) = e(n)

    mu : LMSステップサイズ
    normalize_acc : ACC各軸を標準偏差で割って正規化するかどうか
    return_w_history : True なら w(n) の履歴を返す

    """
    ppg = prep_ppg(ppg, demean=demean_ppg)
    acc = np.asarray(acc, dtype=float)


    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("acc must be shape (N,3)")
    if len(ppg) != acc.shape[0]:
        raise ValueError("ppg and acc must have the same length")

    N = len(ppg)

    # ACC正規化
    acc_proc = prep_acc(acc, demean=demean_acc, normalize=normalize_acc)
    X = make_tapped_regressor(acc_proc, n_taps)
    M = X.shape[1]

    # フィルタ係数: w(n) = [wx, wy, wz]^T
    w = np.zeros(M, dtype=float)

    ma_est = np.zeros(N, dtype=float)
    ppg_clean = np.zeros(N, dtype=float)
    w_hist = np.zeros((N, M), dtype=float) if return_w_history else None

    gate = float(update_gate)

    for n in range(N):
        x_n = X[n]                 # (M,1)
        y_n = float(w @ x_n)              # 推定MA(n)
        e_n = ppg[n] - y_n                # 推定clean(n)

        if gate > 0.0:
            if float(x_n @ x_n) > gate:
                w = w + mu * e_n * x_n
        else:
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
    n_taps = 16,
    demean_ppg = True,
    demean_acc = True,
    normalize_acc=True,
    update_gate = 1e-3,
    return_w_history=True,
):
    """
    ACCを参照信号とするANCによるPPGのMA除去, NLMS

    更新式： w(n+1) = w(n) + (mu / (||x(n)||^2 + eps)) * e(n) * x(n)

    他の引数・返り値はanc_lms_ppg_accと同様
    """
    ppg = prep_ppg(ppg, demean=demean_ppg)
    acc = np.asarray(acc, dtype=float)

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("acc must be shape (N,3)")
    if len(ppg) != acc.shape[0]:
        raise ValueError("ppg and acc must have the same length")

    N = len(ppg)

    acc_proc = prep_acc(acc, demean=demean_acc, normalize=normalize_acc)
    X = make_tapped_regressor(acc_proc, n_taps)
    M = X.shape[1]

    w = np.zeros(M, dtype=float)

    ma_est = np.zeros(N, dtype=float)
    ppg_clean = np.zeros(N, dtype=float)
    w_hist = np.zeros((N, M), dtype=float) if return_w_history else None

    gate = float(update_gate)

    for n in range(N):
        x_n = X[n]
        y_n = float(w @ x_n) 
        e_n = ppg[n] - y_n

        norm2 = float(x_n @ x_n) 
        if norm2 > gate:  # ★静止/参照弱い区間の更新を抑制
            w = w + (mu / (norm2 + eps)) * e_n * x_n

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
    n_taps = 16,
    demean_ppg = True,           # ★追加：PPG平均除去
    demean_acc = True,       
    normalize_acc = True,
    update_gate = 0.0,
    return_w_history = True,
):
    """
    ACCを参照信号とするANCによるPPGのMA除去, NLMS

    コスト関数：J = sum_{k=0}^n lambda^{n-k} e(k)^2

    更新式：
        K(n) = P(n-1)x(n) / (lambda + x(n)^T P(n-1) x(n))
        w(n) = w(n-1) + K(n) e(n)
        P(n) = (1/lambda)[P(n-1) - K(n)x(n)^T P(n-1)]

    lambda : 忘却係数 (0<lambda<=1)
    delta : 初期共分散 P(0) = delta * I のスケール
    """
    ppg = prep_ppg(ppg, demean=demean_ppg)
    acc = np.asarray(acc, dtype=float)

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("acc must be shape (N,3)")
    if len(ppg) != acc.shape[0]:
        raise ValueError("ppg and acc must have the same length")

    N = len(ppg)

    acc_proc = prep_acc(acc, demean=demean_acc, normalize=normalize_acc)
    X = make_tapped_regressor(acc_proc, n_taps)
    M = X.shape[1]

    # 初期値
    w = np.zeros(M, dtype=float)
    P = delta * np.eye(M, dtype=float)

    ma_est = np.zeros(N, dtype=float)
    ppg_clean = np.zeros(N, dtype=float)
    w_hist = np.zeros((N, M), dtype=float) if return_w_history else None

    gate = float(update_gate)

    for n in range(N):
        x_n = X[n].reshape(-1, 1)  # (M,1)

        # y_hat = w^T x
        y_n = float(w @ x_n[:, 0])
        e_n = ppg[n] - y_n

        # 更新ゲート（任意）
        if gate > 0.0 and float(x_n[:, 0] @ x_n[:, 0]) <= gate:
            ma_est[n] = y_n
            ppg_clean[n] = e_n
            if return_w_history:
                w_hist[n] = w
            continue

        # K(n)
        Px = P @ x_n                        # (M,1)
        denom = lam + float(x_n.T @ Px)     # スカラー
        K = Px / denom                      # (M,1)

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
    n_taps = 16,
    demean_ppg=True,           
    demean_acc=True,          
    normalize_acc=True,
    update_gate=1e-3,  
    return_w_history=True,
):
    """
    ACCを参照信号とするANCによるPPGのMA除去, Kalman

    状態：w(n+1) = w(n) + q(n),  q ~ N(0, Q=q_var*I)
    観測：PPG(n) = x(n)^T w(n) + r(n),  r ~ N(0, R=r_var)

    q_var : 状態ノイズ共分散のスケール (Q = q_var * I)
            → 大きいと w(n) が時間変化に素早く追従
    r_var : 観測ノイズ分散 R
            → 大きいと観測がノイジーだとみなされる

    他の引数・返り値は他の関数と同様
    """
    ppg = prep_ppg(ppg, demean=demean_ppg)
    acc = np.asarray(acc, dtype=float)

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("acc must be shape (N,3)")
    if len(ppg) != acc.shape[0]:
        raise ValueError("ppg and acc must have the same length")
    
    N = len(ppg)

    acc_proc = prep_acc(acc, demean=demean_acc, normalize=normalize_acc)
    X = make_tapped_regressor(acc_proc, n_taps)
    M = X.shape[1]
    
    # 初期状態
    w = np.zeros(M, dtype=float)
    P = np.eye(M, dtype=float) * 1e3  # 大きめの初期共分散

    Q = q_var * np.eye(M, dtype=float)
    R = float(r_var)

    ma_est = np.zeros(N, dtype=float)
    ppg_clean = np.zeros(N, dtype=float)
    w_hist = np.zeros((N, M), dtype=float) if return_w_history else None

    gate = float(update_gate)

    for n in range(N):
        x_n = X[n].reshape(-1, 1)  # (M,1)

        if float(x_n[:, 0] @ x_n[:, 0]) <= gate:
            y_hat = float(w @ x_n[:, 0])
            ma_est[n] = y_hat
            ppg_clean[n] = ppg[n] - y_hat
            if return_w_history:
                w_hist[n] = w
            continue

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
        K = (P_pred @ x_n) / S            # (M,1)

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
