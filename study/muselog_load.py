"""Turn a MuseLog phone CSV (every packet type interleaved, ~100 columns, 1 GB a night) into one
compact .npz the analysis scripts can open in a second:

  t          seconds since the first EEG sample (phone clock, per sample)
  eeg        (n, 6) float32 uV: TP9, AF7, AF8, TP10, AUX_LEFT ("DRL" column), AUX_RIGHT ("REF")
             - on a Muse 2 the micro-USB electrode arrives as AUX_RIGHT; the Athena sends 0 there
  hsi        (n, 4) float32 horseshoe quality per electrode (1 good, 2 ok, 4 off), forward-filled
  good       (n, 4) float32 the SDK's artifact-free flag per electrode, forward-filled
  acc, gyro  (m, 4): t, x, y, z at the IMU rate
  optics     (k, 17): t + the 16 Athena optical channels (730/850 nm, red, ambient), empty on a Muse 2
  misc       (j, 5): t, battery %, HR bpm, HRV RMSSD ms, HR signal ok (nan when absent)
  clock0     ISO wall-clock of the first EEG sample; fs_est = samples / elapsed

    python study/muselog_load.py data/muselog/20261028/muse_session_20261028_034217_Muse1.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = ["TP9_RAW", "AF7_RAW", "AF8_RAW", "TP10_RAW", "DRL", "REF"]
HSI = [f"{c}_CONNECTION_STRENGTH(HSI)" for c in ("TP9", "AF7", "AF8", "TP10")]
GOOD = [f"{c}_ARTIFACT_FREE(IS_GOOD)" for c in ("TP9", "AF7", "AF8", "TP10")]
ACC = ["ACCEL_X", "ACCEL_Y", "ACCEL_Z"]
GYR = ["GYRO_X", "GYRO_Y", "GYRO_Z"]
OPT = ["730nm_LEFT_OUTER", "730nm_RIGHT_OUTER", "850nm_LEFT_OUTER", "850nm_RIGHT_OUTER",
       "730nm_LEFT_INNER", "730nm_RIGHT_INNER", "850nm_LEFT_INNER", "850nm_RIGHT_INNER",
       "RED_LEFT_OUTER", "RED_RIGHT_OUTER", "AMBIENT_LEFT_OUTER", "AMBIENT_RIGHT_OUTER",
       "RED_LEFT_INNER", "RED_RIGHT_INNER", "AMBIENT_LEFT_INNER", "AMBIENT_RIGHT_INNER"]
MISC = ["BATTERY_PERCENT", "HR_BPM", "HRV_RMSSD_MS", "HR_SIGNAL_OK"]


def convert(path, out=None):
    path = Path(path)
    out = Path(out) if out else path.with_suffix(".npz")
    cols = ["PACKET_TYPE", "CLOCK_TIME"] + RAW + HSI + GOOD + ACC + GYR + OPT + MISC
    eeg_t, eeg, hsi, good, acc, gyr, opt, misc = [], [], [], [], [], [], [], []
    types = {}
    for c in pd.read_csv(path, usecols=lambda c: c in cols, chunksize=400_000, low_memory=False):
        for k, v in c.PACKET_TYPE.value_counts().items():
            types[k] = types.get(k, 0) + int(v)
        t = pd.to_datetime(c.CLOCK_TIME).astype("int64").to_numpy() / 1e9
        e = c.PACKET_TYPE == "EEG"
        eeg_t.append(t[e])
        eeg.append(c.loc[e, RAW].to_numpy(np.float32))
        hsi.append(c.loc[e, HSI].to_numpy(np.float32))
        good.append(c.loc[e, GOOD].to_numpy(np.float32))
        for names, store in ((ACC, acc), (GYR, gyr), (OPT, opt), (MISC, misc)):
            have = [n for n in names if n in c.columns]
            if not have:
                continue
            m = c[have].notna().any(axis=1).to_numpy()
            if m.any():
                block = np.full((m.sum(), len(names)), np.nan, np.float32)
                for j, n in enumerate(names):
                    if n in c.columns:
                        block[:, j] = c.loc[m, n].to_numpy(np.float32)
                store.append(np.column_stack([t[m], block]))
    t = np.concatenate(eeg_t)
    t0 = t[0]
    stack = lambda L, w: (np.concatenate(L) if L else np.zeros((0, w), np.float32))
    hsi_a, good_a = stack(hsi, 4), stack(good, 4)
    for a in (hsi_a, good_a):                     # forward-fill the per-second quality onto samples
        for j in range(a.shape[1]):
            col = a[:, j]
            idx = np.where(np.isfinite(col), np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            a[:, j] = col[idx]
    def rel(L, w):
        a = stack(L, w + 1)
        if len(a):
            a[:, 0] -= t0
        return a
    fs_est = len(t) / (t[-1] - t0) if len(t) > 1 else 0.0
    np.savez_compressed(out, t=(t - t0).astype(np.float64), eeg=stack(eeg, 6), hsi=hsi_a, good=good_a,
                        acc=rel(acc, 3), gyro=rel(gyr, 3), optics=rel(opt, 16), misc=rel(misc, 4),
                        clock0=np.array(pd.Timestamp(t0, unit="s").isoformat(timespec="milliseconds")),
                        fs_est=np.array(fs_est), packet_types=np.array(sorted(types.items()), dtype=object))
    print(f"{path.name}: {len(t)} EEG samples, {(t[-1] - t0) / 60:.1f} min, {fs_est:.1f} Hz; packets {types}")
    print("   start", pd.Timestamp(t0, unit="s"), "end", pd.Timestamp(t[-1], unit="s"), "->", out.name)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        convert(p)
