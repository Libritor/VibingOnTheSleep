"""Score a MuseLog phone recording with the rig's detector, as if the rig had kept running.

The detector is calibrated on a stretch of known eyes-closed wake given by wall-clock times
(by default nap 2's calibration, the cleanest one recorded), then fed every 5 s from the end of
that stretch to the end of the recording, causally, exactly as the live rig would. Also writes
per-30-s physiology the classifier does not see as a whole: relative delta, theta, alpha and
sigma power, slow eye movement power, electrode quality (HSI) and head angle.

    python study/muselog_night.py data/muselog/muse_session_20261027_012222_Muse1.csv \
        --calib-start "2026-10-27 02:02:10" --calib-end "2026-10-27 02:05:10" --out study/participants/P-0001/pilot
"""
import argparse
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "detector"))
sys.path.insert(0, str(ROOT / "rig"))
from features import band_feats, eog_feats  # noqa: E402
from model import LiveDetector  # noqa: E402
from sources import contact_ok, derive  # noqa: E402

FS = 256
CH = ["TP9_RAW", "AF7_RAW", "AF8_RAW", "TP10_RAW"]
HSI = ["TP9_CONNECTION_STRENGTH(HSI)", "AF7_CONNECTION_STRENGTH(HSI)",
       "AF8_CONNECTION_STRENGTH(HSI)", "TP10_CONNECTION_STRENGTH(HSI)"]
ACC = ["ACCEL_X", "ACCEL_Y", "ACCEL_Z"]


def read(path):
    cols = ["PACKET_TYPE", "CLOCK_TIME"] + HSI + CH + ACC
    parts = []
    for c in pd.read_csv(path, usecols=cols, chunksize=500_000, low_memory=False):
        parts.append(c[c.PACKET_TYPE == "EEG"].drop(columns="PACKET_TYPE"))
    d = pd.concat(parts, ignore_index=True)
    d["time"] = pd.to_datetime(d.CLOCK_TIME)
    d[ACC] = d[ACC].ffill()
    return d.drop(columns="CLOCK_TIME")


def angle(a, b):
    c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--calib-start", required=True)
    ap.add_argument("--calib-end", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = read(a.csv)
    t0 = d.time.iloc[0]
    sec = (d.time - t0).dt.total_seconds().to_numpy()
    X = d[CH].to_numpy(float)
    cs = (pd.Timestamp(a.calib_start) - t0).total_seconds()
    ce = (pd.Timestamp(a.calib_end) - t0).total_seconds()
    print(f"{len(d)} EEG rows, {sec[-1] / 60:.1f} min, {len(d) / sec[-1]:.1f} Hz; calibration {cs:.0f}-{ce:.0f} s")

    def win(end, n=30 * FS):
        j = np.searchsorted(sec, end, side="right")
        return None if j < n else (X[j - n:j], j)

    lad = LiveDetector(joblib.load(ROOT / "models" / "nap_base3min.joblib"), step_s=5.0)
    rows = []
    acc = d[ACC].to_numpy(float)
    hsi = d[HSI].to_numpy(float)
    for i, e in enumerate(np.arange(cs + 30.0, sec[-1], 5.0)):
        w = win(e)
        if w is None:
            continue
        x, j = w
        ok = contact_ok(x, FS)
        eeg, eog = derive(x)
        p = lad.push(eeg, eog, FS) if ok else None
        row = {"t_s": round(e, 1), "clock": (t0 + pd.Timedelta(seconds=e)).strftime("%H:%M:%S"),
               "contact_ok": ok, "calibrating": not lad.calibrated or e <= ce + 1e-6,
               "hsi_mean": float(np.nanmean(hsi[j - 30 * FS:j])),
               "acc": np.nanmedian(acc[j - FS:j], axis=0)}
        if p:
            row.update({f"p_{k}": v for k, v in p.items()})
        if i % 6 == 0:                                # physiology once per 30-s epoch
            b = band_feats(eeg - eeg.mean(), FS)
            o = eog_feats(eog - eog.mean(), FS)
            row.update({k: b[k] for k in ("rel_delta", "rel_theta", "rel_alpha", "rel_sigma", "log_total")})
            row.update({"log_sem": o["log_sem"], "rem_count": o["rem_count"], "eog_sd": float(eog.std())})
        rows.append(row)
    T = pd.DataFrame(rows)
    cal = T[(T.t_s >= cs + 30) & (T.t_s <= ce)]
    up = np.median(np.stack(cal.acc.to_numpy()), axis=0)
    T["angle_vs_calib"] = [angle(v, up) if np.all(np.isfinite(v)) else np.nan for v in T.acc]
    T = T.drop(columns="acc")
    stages = ["W", "N1", "N2", "N3", "REM"]
    have = T.dropna(subset=["p_W"])
    T.loc[have.index, "stage"] = have[[f"p_{s}" for s in stages]].to_numpy().argmax(1)
    T["stage"] = T.stage.map(dict(enumerate(stages)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "muselog_night_timeline.csv", index=False)

    after = T[(T.t_s > ce) & T.p_W.notna()]
    print(f"scored after calibration: {len(after)} windows, {len(after) * 5 / 60:.1f} min; "
          f"contact ok {T[T.t_s > ce].contact_ok.mean():.2f}")
    print("argmax stage share:", after.stage.value_counts(normalize=True).round(3).to_dict())
    # 30-s epochs: majority of the six 5-s windows
    after = after.assign(epoch=((after.t_s - ce) // 30).astype(int))
    ep = after.groupby("epoch").agg(clock=("clock", "first"), stage=("stage", lambda s: s.mode().iloc[0]),
                                    p_sleep=("p_W", lambda s: 1 - s.mean()))
    ep.to_csv(out / "muselog_night_epochs.csv")
    runs = (ep.stage != ep.stage.shift()).cumsum()
    print("\nstage runs (30-s epochs):")
    for _, g in ep.groupby(runs):
        print(f"  {g.clock.iloc[0]}  {g.stage.iloc[0]:3s}  {len(g) * 0.5:5.1f} min  P(sleep) {g.p_sleep.mean():.2f}")


if __name__ == "__main__":
    main()
