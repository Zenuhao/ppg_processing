import numpy as np
import pandas as pd
from HR_estiation_ppg import elgendi, ibi
from lambert_beer import remove_ma_beer_lambert

fs = 500
motion = ["run", "sit", "walk"]

params_csv = "parameter_summary_revised_ch6_sorted.csv"
df_params = pd.read_csv(params_csv)

for i in range(2, 23):
    for j in motion:
        csv = f"s{i}_{j}.csv"
        df = pd.read_csv(csv)

        time = pd.to_datetime(df["time"])
        t0 = time.iloc[0]
        time_sec = (time - t0).dt.total_seconds().to_numpy()

        acc_raw = df[["a_x", "a_y", "a_z"]].to_numpy(dtype=float)
        ppg_raw = df["pleth_6"].to_numpy(dtype=float)

        mask = (
            np.isfinite(time_sec)
            & np.isfinite(ppg_raw)
            & np.isfinite(acc_raw).all(axis=1)
        )

        t_sec = time_sec[mask]
        ppg = ppg_raw[mask]
        acc = acc_raw[mask]

        if len(ppg) < 10:
            print(f"s{i}_{j}: too short after masking")
            continue
        
        acc_x = acc[:,0]
        acc_y = acc[:,1]
        acc_z = acc[:,2]

        ppg_hat, ma = remove_ma_beer_lambert(
            ppg,
            acc_x, acc_y, acc_z,
            eps=1.0,                 # モル吸光係数（正規化）
            c=2.3e-3,                # mol/L
            k_blood=1.0,             # 補正係数
            dt=0.002                 # 500 Hz
        )
        peaks, _ = elgendi(ppg_hat, fs)
        ibi_sec, t_ibi_sec, hr_bpm = ibi(peaks, fs)
        pd.DataFrame({
            "csvname": csv,
            "t_ibi_sec": t_ibi_sec,
            "ibi_sec": ibi_sec,
            "hr_bpm": hr_bpm,
        }).to_csv(f"s{i}_{j}_lambert-beer_signals.csv", index=False)
        print(f"{i}_{j} is finished")

print("Saved all csv")


"""
for i in range(2, 23):
    k = 0
    for j in motion:
        csv = f"s{i}_{j}.csv"
        df = pd.read_csv(csv)

        time = pd.to_datetime(df["time"])
        t0 = time.iloc[0]
        time_sec = (time - t0).dt.total_seconds().to_numpy()

        acc_raw = df[["a_x", "a_y", "a_z"]].to_numpy(dtype=float)
        ppg_raw = df["pleth_6"].to_numpy(dtype=float)

        mask = (
            np.isfinite(time_sec)
            & np.isfinite(ppg_raw)
            & np.isfinite(acc_raw).all(axis=1)
        )

        t_sec = time_sec[mask]
        ppg = ppg_raw[mask]
        acc = acc_raw[mask]

        if len(ppg) < 10:
            print(f"s{i}_{j}: too short after masking")
            k += 1
            continue

        # ACC preprocessing
        acc_centered = acc - np.median(acc, axis=0, keepdims=True)
        acc_mag = np.linalg.norm(acc_centered, axis=1)

        # 動作ごとに clean とみなす割合
        q_map = {"sit": 0.30, "walk": 0.20, "run": 0.10}
        q = q_map.get(j, 0.20)
        acc_thr = float(np.quantile(acc_mag, q))
        print("acc_thr=", acc_thr, "q=", q)

        # parameter row
        row = (i - 2) * len(motion) + k
        m = int(df_params["m"].iloc[row])
        tau = int(df_params["tau"].iloc[row])
        print(f"{csv}, tau={tau}, m={m}")

        # params
        params = DenoiseParams(
            m=m, tau=tau, dt=1.0 / fs,
            theta=2.0, ridge=1e-5,
            n_iter=4,
            acc_thr=None,
            simplex_err_q=0.90,
            proj_strength=1.0,
            smooth_gamma=0.0,
        )

        params_acc = DenoiseParams(
            m=m, tau=tau, dt=1.0 / fs,
            theta=2.0, ridge=1e-5,
            n_iter=4,
            acc_thr=acc_thr,        # ★ここが「加速度あり版」
            simplex_err_q=0.90,
            proj_strength=1.0,
            smooth_gamma=0.0,
        )

        # SDPPG pre-smoothing
        gamma_map = {"sit": 5.0, "walk": 10.0, "run": 20.0}
        G = 0
        sdppg = ppg_to_sdppg(ppg, 1.0 / fs, pre_smooth_gamma=G)

        # ---- 1) Normal PPG without acc ----
        ppg_hat, info = denoise_ppg_ma(ppg, None, p=params)
        print(info["diagnostics"])
        peaks, _ = elgendi(ppg_hat, fs)
        ibi_sec, t_ibi_sec, hr_bpm = ibi(peaks, fs)
        pd.DataFrame({
            "csvname": csv,
            "t_ibi_sec": t_ibi_sec,
            "ibi_sec": ibi_sec,
            "hr_bpm": hr_bpm,
        }).to_csv(f"s{i}_{j}_attractor_signals.csv", index=False)

        # ---- 2) Normal PPG with acc ----
        ppg_hat_acc, info_acc = denoise_ppg_ma(ppg, acc_centered, p=params_acc)
        print(info_acc["diagnostics"])
        peaks_acc, _ = elgendi(ppg_hat_acc, fs)
        ibi_sec_acc, t_ibi_sec_acc, hr_bpm_acc = ibi(peaks_acc, fs)
        pd.DataFrame({
            "csvname": csv,
            "t_ibi_sec": t_ibi_sec_acc,
            "ibi_sec": ibi_sec_acc,
            "hr_bpm": hr_bpm_acc,
        }).to_csv(f"s{i}_{j}_attractor_signals_acc.csv", index=False)

        # ---- 3) SDPPG without acc ----
        sdppg_hat_sd, info_sd = denoise_ppg_ma(sdppg, None, p=params)
        print(info_sd["diagnostics"])
        ppg_hat_sd = sdppg_to_ppg(sdppg_hat_sd, 1.0 / fs, y_ref=ppg)
        peaks_sd, _ = elgendi(ppg_hat_sd, fs)
        ibi_sec_sd, t_ibi_sec_sd, hr_bpm_sd = ibi(peaks_sd, fs)
        pd.DataFrame({
            "csvname": csv,
            "t_ibi_sec": t_ibi_sec_sd,
            "ibi_sec": ibi_sec_sd,
            "hr_bpm": hr_bpm_sd,
        }).to_csv(f"s{i}_{j}_attractor_signals_sd.csv", index=False)

        # ---- 4) SDPPG with acc ----
        sdppg_hat_acc_sd, info_acc_sd = denoise_ppg_ma(sdppg, acc_centered, p=params_acc)
        print(info_acc_sd["diagnostics"])
        ppg_hat_acc_sd = sdppg_to_ppg(sdppg_hat_acc_sd, 1.0 / fs, y_ref=ppg)
        peaks_acc_sd, _ = elgendi(ppg_hat_acc_sd, fs)
        ibi_sec_acc_sd, t_ibi_sec_acc_sd, hr_bpm_acc_sd = ibi(peaks_acc_sd, fs)
        pd.DataFrame({
            "csvname": csv,
            "t_ibi_sec": t_ibi_sec_acc_sd,
            "ibi_sec": ibi_sec_acc_sd,
            "hr_bpm": hr_bpm_acc_sd,
        }).to_csv(f"s{i}_{j}_attractor_signals_acc_sd.csv", index=False)

        print(f"s{i}_{j} is finished")
        k += 1

print("Saved all csv")


"""

#Proposed Idea
"""

fs = 500
motion_list = ["run", "sit", "walk"]

params_csv = "parameter_summary_revised_ch6_sorted.csv"
df_params = pd.read_csv(params_csv)


motion = ["run", "sit", "walk"]

for i in range(2, 23):
    k = 0
    for j in motion:
        csv = f"s{i}_{j}.csv"
        df = pd.read_csv(csv)
        time = df['time']
        time = pd.to_datetime(time)
        t0 = time.iloc[0]
        time_sec = (time - t0).dt.total_seconds().to_numpy()

        acc_raw = df[["a_x", "a_y", "a_z"]].to_numpy(dtype=float)
        ppg_raw = df[f"pleth_6"].to_numpy(dtype=float)

        mask = (
                np.isfinite(time_sec)
                & np.isfinite(ppg_raw)
                & np.isfinite(acc_raw).all(axis=1)
            )
        
        t_sec = time_sec[mask]
        ppg = ppg_raw[mask]
        acc = acc_raw[mask]

        acc_centered = acc - np.median(acc, axis=0, keepdims=True)
        acc_mag = np.linalg.norm(acc_centered, axis=1)

        #動作ごとに“クリーンとみなす割合”を変える（おすすめ）
        q_map = {"sit": 0.30, "walk": 0.20, "run": 0.10}   # 例
        q = q_map.get(j, 0.20)

        acc_thr = float(np.quantile(acc_mag, q))

        print("acc_thr=", acc_thr, "q=", q)

        if len(ppg) < 10:
                print(f"s{i}_{j}: too short after masking")
                continue
        
        row = (i-2) * len(motion) + k
        m = int(df_params["m"].iloc[row])
        tau= int(df_params["tau"].iloc[row])

        print(f"s{i}_{j}, {tau}, {m}")

        k += 1

        params = DenoiseParams(
            m=m, tau=tau, dt = 1.0/fs,
            theta=2.0, ridge=1e-5,
            n_iter=4,
            acc_thr=None,
            simplex_err_q=0.90,
            proj_strength=1.0,
            smooth_gamma=2.0

        )
        
        params_acc = DenoiseParams(
            m=m, tau=tau, dt = 1.0/fs,
            theta=2.0, ridge=1e-5,
            n_iter=4,
            acc_thr=acc_thr,
            simplex_err_q=0.90,
            proj_strength=1.0,
            smooth_gamma=2.0

        )

        gamma_map = {"sit": 5.0, "walk": 10.0, "run": 20.0}
        G = gamma_map.get(j, 10.0)

        sdppg = ppg_to_sdppg(ppg, 1.0/fs, pre_smooth_gamma=G)

        # Normal PPG without acc
        ppg_hat, info = denoise_ppg_ma(ppg, None, p=params)
        print(info["diagnostics"])
        peaks, infopeak = elgendi(ppg_hat, fs)
        ibi_sec, t_ibi_sec, hr_bpm = ibi(peaks, fs)
        hr_output = pd.DataFrame({
            "csvname": csv,
            "t_ibi_sec": t_ibi_sec,
            "ibi_sec": ibi_sec,
            "hr_bpm":hr_bpm
        })
        hr_output.to_csv(f"s{i}_{j}_attractor_signals.csv", index=False)

        # Normal PPG with acc
        ppg_hat_acc, info_acc = denoise_ppg_ma(ppg, acc_centered, p=params_acc)
        print(info_acc["diagnostics"])
        peaks_acc, infopeak_acc = elgendi(ppg_hat_acc, fs)
        ibi_sec_acc, t_ibi_sec_acc, hr_bpm_acc = ibi(peaks_acc, fs)
        hr_output_acc = pd.DataFrame({
            "csvname": csv,
            "t_ibi_sec": t_ibi_sec_acc,
            "ibi_sec": ibi_sec_acc,
            "hr_bpm":hr_bpm_acc
        })
        hr_output_acc.to_csv(f"s{i}_{j}_attractor_signals_acc.csv", index=False)

        # Second derivative PPG witout acc
        sdppg_hat_sd, info_sd = denoise_ppg_ma(sdppg, None, p=params)
        print(info_sd["diagnostics"])
        ppg_hat_sd = sdppg_to_ppg(sdppg_hat_sd, 1.0/fs, y_ref=ppg)
        peaks_sd, infopeak_sd = elgendi(ppg_hat_sd, fs)
        ibi_sec_sd, t_ibi_sec_sd, hr_bpm_sd = ibi(peaks_sd, fs)
        hr_output_sd = pd.DataFrame({
            "csvname": csv,
            "t_ibi_sec": t_ibi_sec_sd,
            "ibi_sec": ibi_sec_sd,
            "hr_bpm":hr_bpm_sd
        })
        hr_output_sd.to_csv(f"s{i}_{j}_attractor_signals_sd.csv", index=False)

        # Second derivative PPG with acc
        sdppg_hat_acc_sd, info_acc_sd = denoise_ppg_ma(sdppg, acc_centered, p=params_acc)
        print(info_acc_sd["diagnostics"])
        ppg_hat_acc_sd = sdppg_to_ppg(sdppg_hat_acc_sd, 1.0/fs, y_ref=ppg)        
        peaks_acc_sd, infopeak_acc_sd = elgendi(ppg_hat_acc_sd, fs)
        ibi_sec_acc_sd, t_ibi_sec_acc_sd, hr_bpm_acc_sd = ibi(peaks_acc_sd, fs)
        hr_output_acc_sd = pd.DataFrame({
            "csvname": csv,
            "t_ibi_sec": t_ibi_sec_acc_sd,
            "ibi_sec": ibi_sec_acc_sd,
            "hr_bpm":hr_bpm_acc_sd
        })
        hr_output_acc_sd.to_csv(f"s{i}_{j}_attractor_signals_acc_sd.csv", index=False)

        print(f"s{i}_{j} is finished")
        print(f"{hr_output}")

print("Saved all csv")    


""" 

#stft

"""

for i in range(2, 23):
    for j in motion_list:
        csv = f"s{i}_{j}.csv"
        df = pd.read_csv(csv)
        time = df['time']
        time = pd.to_datetime(time)
        t0 = time.iloc[0]
        time_sec = (time - t0).dt.total_seconds().to_numpy()

        acc_raw = df[["a_x", "a_y", "a_z"]].to_numpy(dtype=float)
        ppg_raw = df[f"pleth_6"].to_numpy(dtype=float)
        mask = (
            np.isfinite(time_sec)
            & np.isfinite(ppg_raw)
            & np.isfinite(acc_raw).all(axis=1)
        )

        t_sec = time_sec[mask]
        ppg = ppg_raw[mask]
        acc = acc_raw[mask]

        if len(ppg) < 10:
            print(f"s{i}_{j}: too short after masking")
            continue

        ppg_clean, info = stft_hr_track_reconstruct(ppg, fs)

        info["hr_track_bpm"]

        L = min(len(t_sec), len(ppg), len(ppg_clean))
        t_sec = t_sec[:L]
        ppg = ppg[:L]
        ppg_clean = ppg_clean[:L]

        peaks, infopeak = elgendi(ppg_clean, fs)
        ibi_sec, t_ibi_sec, hr_bpm = ibi(peaks, fs)

        out_sig = pd.DataFrame({
            "csvname": csv,
            "channel": i,
            "t_sec": t_sec,
            "ppg_raw": ppg,
            "ppg_clean": ppg_clean,
        })

        hr_output = pd.DataFrame({
            "csvname": csv,
            "channel": i,
            "t_ibi_sec": t_ibi_sec,
            "ibi_sec": ibi_sec,
            "hr_bpm":hr_bpm
        })
        out_sig.to_csv(f"s{i}_{j}_fft_signals.csv", index=False)
        hr_output.to_csv(f"s{i}_{j}_fft_hr_output.csv", index=False)

        print(f"s{i}_{j} finished")

print("Saved all csv")    

"""


# DAE 

"""
def frame_signal_1d(x: np.ndarray, win_len: int, hop: int):
    1D信号を (Nw, win_len) の窓行列にする
    x = np.asarray(x, dtype=np.float32)
    N = len(x)
    if N < win_len:
        return np.empty((0, win_len), dtype=np.float32), np.empty((0,), dtype=int)
    starts = np.arange(0, N - win_len + 1, hop, dtype=int)
    X = np.stack([x[s:s+win_len] for s in starts], axis=0)
    return X, starts

def overlap_add_mean(frames: np.ndarray, starts: np.ndarray, N: int):
    (Nw, win_len) を重ね合わせ平均で 1D に戻す
    if len(starts) == 0:
        return np.zeros(N, dtype=np.float32)
    win_len = frames.shape[1]
    y = np.zeros(N, dtype=np.float32)
    w = np.zeros(N, dtype=np.float32)
    for k, s in enumerate(starts):
        y[s:s+win_len] += frames[k]
        w[s:s+win_len] += 1.0
    return y / np.maximum(w, 1e-8)

for i in range(2, 23):
    for j in motion_list:
        csv = f"s{i}_{j}.csv"
        df = pd.read_csv(csv)
        time = df['time']
        time = pd.to_datetime(time)
        t0 = time.iloc[0]
        time_sec = (time - t0).dt.total_seconds().to_numpy()

        acc_raw = df[["a_x", "a_y", "a_z"]].to_numpy(dtype=float)
        ppg_raw = df[f"pleth_6"].to_numpy(dtype=float)
        mask = (
            np.isfinite(time_sec)
            & np.isfinite(ppg_raw)
            & np.isfinite(acc_raw).all(axis=1)
        )

        t_sec = time_sec[mask]
        ppg = ppg_raw[mask]
        acc = acc_raw[mask]

        if len(ppg) < 10:
            print(f"s{i}_{j}: too short after masking")
            continue

        win_len = cfg.d_in                    # D と一致させる
        hop = int(1.0 * fs)                   # 例：1秒刻み（好きに調整OK）
        X_ppg, starts = frame_signal_1d(ppg, win_len=win_len, hop=hop)

        if X_ppg.shape[0] == 0:
            print(f"{csv}: too short for windowing (len={len(ppg)}, win_len={win_len})")
            continue

        X_clean = dae_denoise(
            model,
            X_ppg,                              # (Nw, D)
            normalize="window_zscore",
            rescale=True,
        )

        ppg_clean = overlap_add_mean(X_clean, starts, N=len(ppg))

        peaks, info = elgendi(ppg_clean, fs)
        ibi_sec, t_ibi_sec, hr_bpm = ibi(peaks, fs)

        out_sig = pd.DataFrame({
            "csvname": csv,
            "channel": i,
            "t_sec": t_sec,
            "ppg_raw": ppg,
            "ppg_clean": ppg_clean
        })

        hr_output = pd.DataFrame({
            "csvname": csv,
            "channel": i,
            "t_ibi_sec": t_ibi_sec,
            "ibi_sec": ibi_sec,
            "hr_bpm":hr_bpm
        })
        out_sig.to_csv(f"s{i}_{j}_dae_signals.csv", index=False)
        hr_output.to_csv(f"s{i}_{j}_dae_hr_output.csv", index=False)

        print(f"s{i}_{j} finished")

print("Saved all csv")  


"""

# MA cancelling by ANC & HR estimation from clean PPG
"""
for i in range(2, 23):
    for j in motion_list:
        csv = f"s{i}_{j}.csv"
        df = pd.read_csv(csv)
        time = df['time']
        time = pd.to_datetime(time)
        t0 = time.iloc[0]
        time_sec = (time - t0).dt.total_seconds().to_numpy()

        acc_raw = df[["a_x", "a_y", "a_z"]].to_numpy(dtype=float)
        ppg_raw = df[f"pleth_6"].to_numpy(dtype=float)
        mask = (
            np.isfinite(time_sec)
            & np.isfinite(ppg_raw)
            & np.isfinite(acc_raw).all(axis=1)
        )

        t_sec = time_sec[mask]
        ppg = ppg_raw[mask]
        acc = acc_raw[mask]

        if len(ppg) < 10:
            print(f"s{i}_{j}: too short after masking")
            continue

        ppg_clean, ma_est, _ = anc_lms_ppg_acc(ppg, acc, n_taps=50, return_w_history=False)

        peaks, info = elgendi(ppg_clean, fs)
        ibi_sec, t_ibi_sec, hr_bpm = ibi(peaks, fs)

        out_sig = pd.DataFrame({
            "csvname": csv,
            "channel": i,
            "t_sec": t_sec,
            "ppg_raw": ppg,
            "ppg_clean": ppg_clean,
            "ma_est": ma_est,
        })

        hr_output = pd.DataFrame({
            "csvname": csv,
            "channel": i,
            "t_ibi_sec": t_ibi_sec,
            "ibi_sec": ibi_sec,
            "hr_bpm":hr_bpm
        })
        out_sig.to_csv(f"s{i}_{j}_anc_signals.csv", index=False)
        hr_output.to_csv(f"s{i}_{j}_hr_output.csv", index=False)

        print(f"s{i}_{j} finished")

print("Saved all csv")    
"""
    
# HR estimation from PPG 

"""
for csv in csvs:
    df = pd.read_csv(csv)
    time = df['time']
    time = pd.to_datetime(time)
    t0 = time.iloc[0]
    time = time - t0
    time = time.dt.total_seconds()
    for i in range(1,7):
        ppg = df[f"pleth_{i}"].to_numpy(dtype=float)
        ppg = ppg[np.isfinite(ppg)]

        mask = np.isfinite(ppg) & np.isfinite(time.to_numpy())
        ppg = ppg[mask]
        time = time.to_numpy()[mask]

        peaks, info = elgendi(ppg, fs)
        ibi_sec, t_ibi_sec, hr_bpm = ibi(peaks, fs)

        for k in range(len(hr_bpm)):
            results.append({
                "csvname": csv,
                "channel": i,
                "beat_idx": k,
                "t_ibi_sec": float(t_ibi_sec[k]),
                "ibi_sec": float(ibi_sec[k]),
                "hr_bpm": float(hr_bpm[k]),
                "n_peaks": int(len(peaks)),
            })
        print(f"{csv}_{i} is finished")
        
        peaks_s, props_s = ppg_estimation_hr_simple(ppg, fs)  # props_s は使わなくてもOK
        ibi_s, t_ibi_s, hr_s = ibi(peaks_s, fs)
        
        for k in range(len(hr_s)):
            results.append({
                "csvname": csv,
                "channel": i,
                "beat_idx": k,
                "t_ibi_sec": float(t_ibi_s[k]),
                "ibi_sec": float(ibi_s[k]),
                "hr_bpm": float(hr_s[k]),
                "n_peaks": int(len(peaks_s)),
            })

        print(f"{csv}_ch{i} finished: elgendi_peaks={len(peaks_s)}, simple_peaks={len(peaks_s)}")

df_parameters = pd.DataFrame(results)
df_parameters.to_csv("heart_rate_ppg_simple.csv", index=False)
print("Saved parameter summary")
"""


# Calculation of Tau and Embedding Dimension
"""
for csv in csvs:
    df = pd.read_csv(csv)
    time = df['time']
    time = pd.to_datetime(time)
    t0 = time.iloc[0]
    time = time - t0
    time = time.dt.total_seconds()
    for i in range(1,7):
        ppg = df[f"pleth_{i}"].to_numpy(dtype=float)
        ppg = ppg[np.isfinite(ppg)]


        #tau = estimate_delay_ami(ppg, max_lag_samples=max_len_samples, n_bins=32, smooth_window=5)
        #m = estimate_embedding_dimension_fnn(ppg, tau, m_max=10, r_tol=15.0, a_tol=2.0, theiler=0, m_min=2)

        tau = estimate_delay_ami_tau_only_ds(
            ppg,
            fs=fs,
            fs_est=50,         # τ推定用のfs（25〜50推奨）
            max_lag_sec=2.0,   # A: 秒で探索範囲を与える
            min_lag_sec=0.03,  # B: PPG用に短すぎるτを避ける（例 30ms）
            n_bins=32,
            smooth_window=5,
            avoid_tail_ratio=0.0,
        )

        m = estimate_embedding_dimension_fnn(ppg, tau=tau, m_max=10, r_tol=15.0, a_tol=2.0, m_min=2, theiler=0)

        # ---- (6) 診断プロット（必要なときだけON推奨）----
        # 重ければコメントアウトしてOK
        out_prefix = f"diag_{csv}_pleth_{i}"
        save_ami_fnn_plots(
            ppg, fs=fs, out_prefix=out_prefix,
            max_lag_sec=2.0, n_bins=32, smooth_window=5,
            tau_for_fnn=tau, m_max=10, theiler=0
        )

        results.append({
            "csvname": csv,
            "channel": i,
            "tau": tau,
            "m": m,
            "tau_sec": tau / fs
        })
        print(f"{csv}_{i} is finished")

df_parameters = pd.DataFrame(results)
df_parameters.to_csv("parameter_summary_revised.csv", index=False)
print("Saved parameter summary")
"""

#HR estimation from ECG
"""
for csv in csvs:
    df = pd.read_csv(csv)

    time = df['time']
    time = pd.to_datetime(time)
    t0 = time.iloc[0]
    time = time - t0
    time = time.dt.total_seconds()
    ecg = df['ecg']
    r_peaks, info = pan_tompkins_qrs(ecg, fs)
    hr_bpm, t_hr = compute_hr(r_peaks, fs)

    df_hr = pd.DataFrame({
        't_hr_sec': t_hr,     # HR が定義される時刻 [s]
        'hr_bpm': hr_bpm      # 心拍数 [bpm]
    })

     # 出力ファイル名: 元ファイル名 + "_hr.csv"
    in_path = Path(csv)
    out_path = in_path.with_name(in_path.stem + '_hr.csv')

    # CSV に保存
    df_hr.to_csv(out_path, index=False)

    print(f"Saved HR result to: {out_path}")

    plt.figure(figsize=(10, 4))
    plt.plot(t_hr, hr_bpm, label='HR (bpm)')
    plt.xlabel('Time [s]')
    plt.ylabel('Heart Rate [bpm]')
    plt.title(f'Heart Rate: {csv}')
    plt.grid(True)
    plt.tight_layout()

    img_path = in_path.with_name(in_path.stem + '_hr.png')
    plt.savefig(img_path, dpi=200)
    plt.close()

    print(f"Saved HR plot to: {img_path}")
"""
