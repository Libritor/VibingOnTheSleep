"""The rig's live copy (night_watch raw.csv: t, TP9, AF7, AF8, TP10, AUX, ax, ay, az) in the same
.npz layout as muselog_load.py, so one analysis runs on either source. The OSC copy has no HSI
(filled with nan) and its AUX goes into the AUX_RIGHT slot.

    python study/rawcsv_to_npz.py study/participants/P-0001/nights/20261028_034419
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def convert(night):
    night = Path(night)
    d = pd.read_csv(night / "raw.csv", low_memory=False)
    t = d.t.to_numpy(float)
    t0 = t[0]
    eeg = np.full((len(d), 6), np.nan, np.float32)
    eeg[:, :4] = d[["TP9", "AF7", "AF8", "TP10"]].to_numpy(np.float32)
    eeg[:, 5] = pd.to_numeric(d.AUX, errors="coerce").to_numpy(np.float32)
    acc = np.column_stack([t - t0, d[["ax", "ay", "az"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)])
    acc = acc[np.isfinite(acc[:, 1:]).all(1)]
    man = json.loads((night / "manifest.json").read_text())
    started = pd.Timestamp(man["started"])
    # raw.csv's t is time.time(); the manifest's 'started' is the local wall clock of the same moment
    clock0 = (started + pd.Timedelta(seconds=t0 - pd.Timestamp(started).timestamp())) if False else \
        pd.Timestamp(t0, unit="s", tz="UTC").tz_convert("America/Toronto").tz_localize(None)
    fs_est = len(t) / (t[-1] - t0)
    out = night / "raw.npz"
    np.savez_compressed(out, t=t - t0, eeg=eeg, hsi=np.full((len(d), 4), np.nan, np.float32),
                        good=np.full((len(d), 4), np.nan, np.float32), acc=acc,
                        gyro=np.zeros((0, 4), np.float32), optics=np.zeros((0, 17), np.float32),
                        misc=np.zeros((0, 5), np.float32), clock0=np.array(clock0.isoformat(timespec="milliseconds")),
                        fs_est=np.array(fs_est), packet_types=np.array([("OSC", len(d))], dtype=object))
    print(f"{night.name}: {len(d)} samples, {(t[-1] - t0) / 60:.1f} min, {fs_est:.1f} Hz, starts {clock0} -> {out}")
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        convert(p)
