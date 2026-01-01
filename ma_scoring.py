def add_ma_scores(df, cols=("D2", "L1", "simplex_err", "acc_rms"), quantile_thr=0.8):
    """
    各指標のZスコアを計算し、MA_score = z^2の和
    上位quantile_thrをis_MA_percentile=Trueとする
    """
    df = df.copy()
    z_cols = []
    for col in cols:
        mu = df[col].mean()
        sd = df[col].std(ddof=0)
        if sd == 0 or np.isnan(sd):
            df[col + "_z"] = np.nan
        else:
            df[col + "_z"] = (df[col] - mu) / sd
        z_cols.append(col + "_z")

    df["MA_score"] = df[z_cols].pow(2).sum(axis=1)

    thr = df["MA_score"].quantile(quantile_thr)
    df["is_MA_percentile"] = df["MA_score"] > thr
    return df


def flag_ma_segments(df,
                     acc_rms_col="acc_rms",
                     L1_col="L1",
                     simplex_col="simplex_err",
                     acc_rms_z_col="acc_rms_z",
                     simplex_z_col="simplex_err_z"):
    """
    加速度が大きい（acc_rms_z > 0）
    L1が0以上（発散傾向）
    Simplex誤差が大きい（simplex_err_z > 0）
    を満たすセグメントを is_MA_logic=True とする
    """
    df = df.copy()

    cond_acc = df[acc_rms_z_col] > 0
    cond_L1  = df[L1_col] > 0
    cond_smp = df[simplex_z_col] > 0

    df["is_MA_logic"] = cond_acc & cond_L1 & cond_smp
    return df

