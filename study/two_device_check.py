"""Do two headbands worn at once see the same brain? The Athena and the Muse 2 both had AF7 and AF8
on the forehead, each against its own Fpz reference, on separate amplifiers and radios. If their
forehead channels agree, the slow waves are physiology and not an electrode's own noise.

Per 30-s epoch where both devices' channel is usable: the 0.5-4 Hz delta waveforms are
cross-correlated over lags of +-4 s (the two phone clocks disagree by a Bluetooth queue), and the
best-lag r is compared with r at lags of 20-60 s (the null: same spectra, no shared waveform).
Also the per-epoch delta-power time courses of both devices are correlated across the night.

    python study/two_device_check.py --a data/muselog/20261028/muse_session_20261028_034217_Muse2.npz \
        --b data/muselog/20261028/muse_session_20261028_034217_Muse1.npz --b-device-clock \
        --out study/participants/P-0001/nights/phone_20261028
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, correlate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from night_channels import clean, FS  # noqa: E402

CH = {"AF7": 1, "AF8": 2}


def load(path, device_clock, anchor):
    z = np.load(path, allow_pickle=True)
    eeg = z["eeg"].astype(float)
    clock0 = pd.Timestamp(str(z["clock0"]))
    t = (np.arange(len(eeg)) / FS if device_clock else z["t"]) + (clock0 - anchor).total_seconds()
    return t, eeg


def epoch(t, eeg, e0, c):
    i, j = np.searchsorted(t, e0 - 4), np.searchsorted(t, e0 + 34)   # 4 s margins for the lag search
    x = eeg[i:j, c]
    if len(x) < 36 * FS or not np.isfinite(x).all() or ((x < 5) | (x > 1640)).mean() > 0.01:
        return None
    y = clean(x - x.mean())
    if not 0.8 < y[int(4 * FS):int(34 * FS)].std() < 60:
        return None
    return y


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="first device npz (phone clock)")
    ap.add_argument("--b", required=True, help="second device npz")
    ap.add_argument("--a-device-clock", action="store_true")
    ap.add_argument("--b-device-clock", action="store_true")
    ap.add_argument("--anchor", default="2026-10-28 03:42:00")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default="athena,muse2")
    a = ap.parse_args(argv)
    anchor = pd.Timestamp(a.anchor)
    la, lb = a.labels.split(",")
    ta, ea = load(a.a, a.a_device_clock, anchor)
    tb, eb = load(a.b, a.b_device_clock, anchor)
    sos = butter(2, [0.5, 4], btype="band", fs=FS, output="sos")
    rows = []
    lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    for e0 in np.arange(np.ceil(lo / 30) * 30, hi - 30, 30.0):
        row = dict(t_s=float(e0), clock=(anchor + pd.Timedelta(seconds=e0)).strftime("%H:%M:%S"))
        for name, c in CH.items():
            ya, yb = epoch(ta, ea, e0, c), epoch(tb, eb, e0, c)
            if ya is None or yb is None:
                continue
            da, db = sosfiltfilt(sos, ya), sosfiltfilt(sos, yb)
            n = int(30 * FS)
            m = int(4 * FS)
            ref = da[m:m + n]                                   # device A, the 30 s proper
            ref = (ref - ref.mean()) / (ref.std() + 1e-9)
            full = (db - db.mean()) / (db.std() + 1e-9)
            cc = correlate(full, ref, mode="valid") / n          # lag k: B shifted by (k - m) samples
            lags = (np.arange(len(cc)) - m) / FS
            k = int(np.argmax(np.abs(cc)))
            row[f"{name}_r"] = round(float(cc[k]), 3)
            row[f"{name}_lag_s"] = round(float(lags[k]), 3)
            row[f"{name}_pow_{la}"] = round(float(da[m:m + n].var()), 2)
            row[f"{name}_pow_{lb}"] = round(float(db[m:m + n].var()), 2)
        if len(row) > 2:
            rows.append(row)
    T = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / f"two_device_{la}_{lb}.csv", index=False)
    res = {}
    for name in CH:
        if f"{name}_r" not in T:
            continue
        g = T.dropna(subset=[f"{name}_r"])
        if len(g) < 3:
            continue
        # null: the same statistic on epochs offset by 20-60 s (device B's epoch k vs device A's k+1)
        # approximated by pairing A's epoch with B's next epoch in the usable sequence
        pa, pb = g[f"{name}_pow_{la}"].to_numpy(), g[f"{name}_pow_{lb}"].to_numpy()
        res[name] = dict(epochs=int(len(g)), median_r=float(g[f"{name}_r"].abs().median()),
                         share_r_above_0p5=float((g[f"{name}_r"].abs() > 0.5).mean()),
                         lag_median_s=float(g[f"{name}_lag_s"].median()),
                         lag_iqr_s=[float(g[f"{name}_lag_s"].quantile(q)) for q in (0.25, 0.75)],
                         delta_power_timecourse_r=float(np.corrcoef(np.log10(pa + 1e-9), np.log10(pb + 1e-9))[0, 1]))
    # waveform null: redo the lag search between A's epoch and B's *following* epoch (no shared signal)
    nulls = {}
    for name, c in CH.items():
        vals = []
        g = T.dropna(subset=[f"{name}_r"]) if f"{name}_r" in T else T.iloc[:0]
        for e0 in g.t_s.to_numpy():
            ya, yb = epoch(ta, ea, e0, c), epoch(tb, eb, e0 + 60, c)
            if ya is None or yb is None:
                continue
            da, db = sosfiltfilt(sos, ya), sosfiltfilt(sos, yb)
            n, m = int(30 * FS), int(4 * FS)
            ref = da[m:m + n]
            ref = (ref - ref.mean()) / (ref.std() + 1e-9)
            full = (db - db.mean()) / (db.std() + 1e-9)
            cc = correlate(full, ref, mode="valid") / n
            vals.append(float(np.max(np.abs(cc))))
        if vals:
            nulls[name] = dict(epochs=len(vals), median_r=float(np.median(vals)),
                               share_r_above_0p5=float(np.mean(np.array(vals) > 0.5)))
    summary = dict(a=a.a, b=a.b, labels=[la, lb], same_epoch=res, offset_epoch_null=nulls)
    (out / f"two_device_{la}_{lb}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if len(T):
        print(f"\n{'clock':>8} " + " ".join(f"{n}_r {n}_lag" for n in CH))
        for _, r in T.iterrows():
            print(f"{r.clock:>8} " + " ".join(f"{r.get(f'{n}_r', np.nan):5.2f} {r.get(f'{n}_lag_s', np.nan):6.2f}" for n in CH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
