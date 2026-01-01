from scipy.signal import butter, filtfilt, find_peaks

def bandpass_filter(x, fs, low=0.5, high=10.0, order=2):
    """
    bandpass filter ranging from 0.5 to 10.0
    """
    x = np.asarray(x)
    nyq = fs / 2.0
    low_n = low / nyq
    high_n = high / nyq
    if high_n >= 1.0:
        high_n = 0.99
    if low_n <= 0.0:
        low_n = 0.001

    b, a = butter(order, [low_n, high_n], btype="band")
    y = filtfilt(b, a, x)
    return y

def five_point_second_derivative(x, fs):
    """
    x''(t) ≈ [x(t+2)-16x(t+1)+30x(t)-16x(t-1)+x(t-2)] / (12*T^2)
    """
    x = np.asarray(x, dtype=float)
    T = 1.0 / fs

    # 周辺部は端値パディング
    x_pad = np.pad(x, (2, 2), mode="edge")
    dd = (
        x_pad[4:]          # t+2
        - 16 * x_pad[3:-1]  # t+1
        + 30 * x_pad[2:-2]  # t
        - 16 * x_pad[1:-3]  # t-1
        + x_pad[0:-4]       # t-2
    ) / (12.0 * T * T)
    return dd

def make_sdppg(ppg, fs, low=0.5, high=10.0, order=2, detrend_order=1):
    """
    detrend -> bandpass -> SDPPG
    """
    ppg = np.asarray(ppg, dtype=float)

    # 線形トレンド除去
    if detrend_order == 1:
        ppg_dt = detrend(ppg, type="linear")
    else:
        ppg_dt = ppg

    # バンドパス
    ppg_f = bandpass_filter(ppg_dt, fs, low=low, high=high, order=order)

    # 2階微分（SDPPG）
    sdppg = five_point_second_derivative(ppg_f, fs)

    # 振幅を標準化（任意）
    std = np.std(sdppg)
    if std > 0:
        sdppg = sdppg / std

    return sdppg