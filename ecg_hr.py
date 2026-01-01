import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# ----------------------------------------------------------------------
# 1. フィルタ処理
# ----------------------------------------------------------------------

def bandpass_ecg(ecg, fs, low=5.0, high=15.0, order=4):
    """
    4次バターワース・バンドパスフィルタで ECG を 5–15 Hz に制限する.

    Parameters
    ----------
    ecg : array_like
        生の ECG 波形 (1次元).
    fs : float
        サンプリング周波数 [Hz].
    low : float, optional
        低域カットオフ周波数 [Hz].
    high : float, optional
        高域カットオフ周波数 [Hz].
    order : int, optional
        フィルタ次数.

    Returns
    -------
    ecg_filt : ndarray
        フィルタ済み ECG 波形.
    """
    ecg = np.asarray(ecg)
    nyq = fs / 2.0
    wp = [low / nyq, high / nyq]
    b, a = butter(order, wp, btype="bandpass")
    # ゼロ位相フィルタリング
    ecg_filt = filtfilt(b, a, ecg)
    return ecg_filt


# ----------------------------------------------------------------------
# 2. Pan–Tompkins 前処理 (微分, 2乗, 移動窓積分)
# ----------------------------------------------------------------------

def derivative_filter(sig):
    """
    Pan–Tompkins 型 5点微分フィルタ (中央差分; オフライン想定).

    y[n] = ( -x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2] ) / 8

    Parameters
    ----------
    sig : array_like
        入力信号.

    Returns
    -------
    der : ndarray
        微分後信号.
    """
    x = np.asarray(sig)
    # パディングして端を安全に処理
    x_pad = np.pad(x, (2, 2), mode="edge")
    der = (
        -x_pad[:-4]
        - 2 * x_pad[1:-3]
        + 2 * x_pad[3:-1]
        + x_pad[4:]
    ) / 8.0
    return der


def moving_window_integration(sig, fs, window_ms=150.0):
    """
    移動窓積分 (Moving Window Integration).

    Parameters
    ----------
    sig : array_like
        入力信号 (通常は 2乗後の信号).
    fs : float
        サンプリング周波数 [Hz].
    window_ms : float, optional
        移動平均窓の長さ [ms].

    Returns
    -------
    mwi : ndarray
        移動窓積分後の信号.
    """
    x = np.asarray(sig)
    N = int(round(window_ms * fs / 1000.0))
    if N < 1:
        N = 1
    kernel = np.ones(N) / N
    # 'same' で長さを維持
    mwi = np.convolve(x, kernel, mode="same")
    return mwi


# ----------------------------------------------------------------------
# 3. Pan–Tompkins 型 R ピーク検出
# ----------------------------------------------------------------------

def pan_tompkins_qrs(ecg, fs,
                     low=5.0,
                     high=15.0,
                     mwi_ms=150.0,
                     refractory_ms=200.0):
    """
    Pan–Tompkins 法に基づく QRS (R 波) 検出.

    1. 4次バターワース・バンドパス (5–15 Hz)
    2. 微分
    3. 2乗
    4. 移動窓積分 (約 150 ms)
    5. 適応しきい値 + 不応期によるピーク選別

    Parameters
    ----------
    ecg : array_like
        生の ECG 波形.
    fs : float
        サンプリング周波数 [Hz].
    low, high : float, optional
        バンドパスのカットオフ [Hz].
    mwi_ms : float, optional
        移動窓積分の窓長 [ms].
    refractory_ms : float, optional
        不応期 (連続 R ピークの最小間隔) [ms].

    Returns
    -------
    r_peaks : ndarray (int)
        検出された R ピークのサンプルインデックス.
    info : dict
        中間信号などを格納した辞書.
        keys: 'ecg_filt', 'diff', 'squared', 'mwi'
    """
    ecg = np.asarray(ecg)

    # 1. バンドパスフィルタ
    ecg_filt = bandpass_ecg(ecg, fs, low=low, high=high, order=4)

    # 2. 微分
    diff = derivative_filter(ecg_filt)

    # 3. 2乗
    squared = diff ** 2

    # 4. 移動窓積分
    mwi = moving_window_integration(squared, fs, window_ms=mwi_ms)

    # 5. ピーク候補の検出 (単純ピーク)
    min_distance = int(round(refractory_ms * fs / 1000.0))  # 不応期
    # 初期ピークはあくまで候補 (後でしきい値でふるう)
    cand_peaks, _ = find_peaks(mwi, distance=min_distance)

    # 適応しきい値の初期化
    # 初期値は信号の統計量からざっくり決める
    init_window = int(2 * fs)  # 最初の2秒を基準に
    init_segment = mwi[:max(init_window, 1)]
    # 信号ピーク候補は上位部分，ノイズピークは下位部分として分ける
    if len(init_segment) == 0:
        raise ValueError("ECG が短すぎます。サンプル数を増やしてください。")

    # 初期ピークのしきい値 (ざっくり平均＋少し)
    th0 = np.mean(init_segment)
    SPKI = np.mean(init_segment[init_segment > th0]) if np.any(init_segment > th0) else np.max(init_segment)
    NPKI = np.mean(init_segment[init_segment <= th0]) if np.any(init_segment <= th0) else np.min(init_segment)

    # R ピークのリスト
    r_peaks = []

    last_r_index = -np.inf  # 不応期判定用

    # Pan–Tompkins 型の適応しきい値ループ
    for idx in cand_peaks:
        peak_amp = mwi[idx]

        # 現在のしきい値
        TH = NPKI + 0.25 * (SPKI - NPKI)

        if peak_amp > TH:
            # R ピーク候補 (QRS)
            # 不応期チェック
            if (idx - last_r_index) >= min_distance:
                r_peaks.append(idx)
                last_r_index = idx

            # 信号ピークレベル更新
            SPKI = 0.125 * peak_amp + 0.875 * SPKI
        else:
            # ノイズピーク
            NPKI = 0.125 * peak_amp + 0.875 * NPKI

    r_peaks = np.asarray(r_peaks, dtype=int)

    info = {
        "ecg_filt": ecg_filt,
        "diff": diff,
        "squared": squared,
        "mwi": mwi,
    }
    return r_peaks, info


# ----------------------------------------------------------------------
# 4. R ピーク列から HR を計算
# ----------------------------------------------------------------------

def compute_hr(r_peaks, fs, method="instant", window_sec=5.0):
    """
    R ピーク列から HR (bpm) を計算する.

    Parameters
    ----------
    r_peaks : array_like (int)
        R ピークのサンプルインデックス.
    fs : float
        サンプリング周波数 [Hz].
    method : {"instant", "window"}, optional
        "instant" : R-R 間隔ごとの瞬時 HR を返す.
        "window"  : 時間窓内の平均 HR を返す（簡易実装; r_peaks の中央値時刻を代表点とする）.
    window_sec : float, optional
        method="window" のときの窓幅 [s].

    Returns
    -------
    hr_bpm : ndarray
        HR (bpm).
    t_hr : ndarray
        HR が対応する時間軸 [s].
    """
    r_peaks = np.asarray(r_peaks, dtype=int)
    if len(r_peaks) < 2:
        raise ValueError("R ピークが少なすぎます。")

    # R-R 間隔 [s]
    rr = np.diff(r_peaks) / float(fs)
    hr = 60.0 / rr  # [bpm]

    if method == "instant":
        # HR を R-R 区間の中点の時間で定義
        t_rr = (r_peaks[1:] + r_peaks[:-1]) / 2.0 / float(fs)
        return hr, t_rr

    elif method == "window":
        # ごく簡単な「窓平均」版 (必要なら細かく実装し直す)
        duration = r_peaks[-1] / float(fs)
        t_edges = np.arange(0, duration + window_sec, window_sec)
        t_center = (t_edges[:-1] + t_edges[1:]) / 2.0
        hr_win = np.zeros_like(t_center)

        for i in range(len(t_center)):
            t0, t1 = t_edges[i], t_edges[i + 1]
            # この窓にかかっている R-R 区間の HR を平均
            mask = ( (r_peaks[1:] / float(fs)) >= t0 ) & \
                   ( (r_peaks[:-1] / float(fs)) <  t1 )
            if np.any(mask):
                hr_win[i] = np.mean(hr[mask])
            else:
                hr_win[i] = np.nan  # データなし

        return hr_win, t_center

    else:
        raise ValueError("method は 'instant' または 'window' を指定してください。")


# ----------------------------------------------------------------------
# 5. おまけ: 単純ピーク検出ユーティリティ
# ----------------------------------------------------------------------

def simple_peak_detect(sig, fs, min_rr_ms=200.0, height=None):
    """
    汎用ピーク検出のラッパ (Pan–Tompkins 以外に簡易に使いたい場合用).

    Parameters
    ----------
    sig : array_like
        入力信号.
    fs : float
        サンプリング周波数 [Hz].
    min_rr_ms : float, optional
        ピーク間の最小間隔 [ms].
    height : float or None, optional
        しきい値 (None の場合は平均+標準偏差などから自動設定してもよい).

    Returns
    -------
    peaks : ndarray (int)
        検出されたピークのインデックス.
    """
    x = np.asarray(sig)
    distance = int(round(min_rr_ms * fs / 1000.0))
    if height is None:
        height = np.mean(x) + 0.5 * np.std(x)
    peaks, _ = find_peaks(x, distance=distance, height=height)
    return peaks
