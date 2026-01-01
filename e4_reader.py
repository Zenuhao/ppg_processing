import pandas as pd
def read_e4_bvp(path):
    """
    Empatica E4 の BVP.csv / Sx_BVP.csv を読む:
      1行目: 開始時刻（UNIX time, など）
      2行目: サンプリング周波数 (64 Hz を想定)
      3行目以降: BVP 値
    """
    df = pd.read_csv(path, header=None)
    fs = float(df.iloc[1, 0])
    sig = df.iloc[2:, 0].astype(float).values
    return sig, fs


def read_e4_acc(path):
    """
    Empatica E4 の ACC.csv / Sx_ACC.csv を読む:
      1行目: 開始時刻
      2行目: サンプリング周波数 (32 Hz を想定)
      3行目以降: x,y,z の3列
    """
    df = pd.read_csv(path, header=None)
    fs = float(df.iloc[1, 0])
    acc = df.iloc[2:, :3].astype(float).values
    return acc, fs