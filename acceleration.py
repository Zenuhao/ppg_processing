import numpy as np
def compute_acc_rms(acc):
    """
    acc: shape(N,3)の加速度g
    DC(重力成分)を引いたRMS
    """
    acc = np.asarray(acc, dtype=float)
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("acc must be shape (N,3)")
    mag = np.linalg.norm(acc, axis=1)
    mag_dc = mag - np.mean(mag)
    rms = np.sqrt(np.mean(mag_dc ** 2))
    return float(rms)