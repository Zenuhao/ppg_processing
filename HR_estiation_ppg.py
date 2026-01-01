import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def butter_bandpass(x, fs, low=0.8, high=5.0, order=2):
    x = np.asarray(x, dtype=float)
    nyq = fs / 2.0
    b, a = butter(order, [(low/nyq), (high/nyq)], btype="bandpass")
    return filtfilt(b, a, x)

def ppg_estimation_hr_simple(ppg, fs, hr_max = 220, bp = (0.7, 4.0), order = 4):
    x = np.asarray(ppg, dtype=float)
    x = butter_bandpass(x, fs, low = bp[0], high=bp[1], order=order)

    min_dist = int(fs * (60.0 / hr_max))
    min_dist = max(1, min_dist)

    prom = 0.5 * np.std(x)

    peaks, props = find_peaks(x, distance=min_dist, prominence=prom)

    return peaks, props

def moving_avarage(ppg, window):
    x = np.asarray(ppg, dtype=float)
    
    if window < 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    
    kernel = np.ones(window, dtype=float) / window
    x = np.convolve(x, kernel, mode="same")
    return x

def contiguous_true_blocks(mask):

    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    blocks = []
    i = 0
    while i < n:
        if mask[i]:
            s = i
            while i < n and mask[i]:
                i += 1
            e = i
            blocks.append((s, e))
        else:
            i += 1
    
    return blocks


def elgendi(ppg, fs, w1_sec = 0.111, w2_sec = 0.667, b = 0.02, hr_max_bpm = 220.0, hr_min_bpm = 40.0, 
            bp = (0.8, 5.0), order = 2, use_abs_peak = True, return_info = True):
    
    low, high = bp
    x = np.asarray(ppg, dtype=float)
    n = len(x)

    if n < 3:
        info = {"x": x} if return_info else None
        return np.array([], dtype=int), info
    
    S = butter_bandpass(x, fs, low=low, high=high, order=order)
    Z = np.clip(S, 0.0, None)
    Q = Z**2

    w1 = max(1, int(round(w1_sec * fs)))
    w2 = max(1, int(round(w2_sec * fs)))

    if w1 % 2 == 0:
        w1 += 1
    if w2 % 2 == 0:
        w2 += 1

    ma_peak = moving_avarage(Q, w1)
    ma_beat = moving_avarage(Q, w2)

    # Elgendi: a=b*z, THR1=MAbeat+a
    z = float(np.mean(Q))
    a = float(b * z)
    th1 = ma_beat + a
    th2 = w1

    mask = ma_peak > th1

    blocks = contiguous_true_blocks(mask)
    blocks_acc = [(s, e) for (s, e) in blocks if (e - s) >= th2]

    # Elgendi: peak index = argmax(|S|) in each accepted block
    peaks = []
    for s, e in blocks_acc:
        seg = S[s:e]
        p = s + int(np.argmax(np.abs(seg) if use_abs_peak else seg))
        peaks.append(p)

    peaks = np.array(sorted(set(peaks)), dtype=int)

    if peaks.size >= 2 and hr_max_bpm is not None:
        min_dist = max(1, int(round(fs * 60.0 /hr_max_bpm)))
        filtered = []
        last = -10**18

        for p in peaks:   
            if p - last >= min_dist:
                filtered.append(p)
                last = p
            
            else:
                prev = filtered[-1]
                if np.abs(S[p]) > np.abs(S[prev]):
                    filtered[-1] = p
                    last = p
        
        peaks = np.array(filtered, dtype=int)

    info = None
    
    if return_info:
        info = {
            "x": x,
            "S": S,
            "Z": Z, 
            "Q": Q,
            "ma_peak": ma_peak,
            "ma_beat": ma_beat,
            "th1": th1,
            "block_all": blocks,
            "blocks_acc": blocks_acc,
            "params": {
                "fs": fs, "bp": bp, "order": int(order),
                "w1_sec": w1_sec, "w2_sec": w2_sec, "b": b,
                "w1": w1, "w2": w2, "th2": th2,
                "hr_max_bpm": hr_max_bpm, "hr_min_bpm": hr_min_bpm,
                "use_abs_peak": use_abs_peak,
            }
        }

    
    return peaks, info   

def ibi(
        peaks, fs, hr_min_bpm = 40.0, hr_max_bpm = 220.0, drop_outliers = True, time_ref = "second"
):
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size < 2:
        return np.array([]), np.array([]), np.array([])
    
    peaks = np.unique(peaks)
    peaks.sort()

    ibi_samples = np.diff(peaks)
    ibi_sec = ibi_samples / float(fs)
    hr_bpm = 60.0 / ibi_sec

    # time stamp of each interval
    if time_ref.lower() == "mid":
        t_ibi_sec = (peaks[1:] + peaks[:-1]) / 2.0 / float(fs)  # midpoint time
    else:
        t_ibi_sec = peaks[1:] / float(fs)  # time at the second peak (common choice)

    if drop_outliers:
        mask = np.ones_like(hr_bpm, dtype=bool)
        if hr_min_bpm is not None:
            mask &= (hr_bpm >= hr_min_bpm)
        if hr_max_bpm is not None:
            mask &= (hr_bpm <= hr_max_bpm)

        ibi_sec = ibi_sec[mask]
        t_ibi_sec = t_ibi_sec[mask]
        hr_bpm = hr_bpm[mask]

    return ibi_sec, t_ibi_sec, hr_bpm













