"""Steady-state scan: is there any steady-state (frequency-tagged, flicker-driven) response in a
night's EEG? For each 30-s epoch and each electrode the Welch spectrum (8-s Hann windows,
0.125 Hz bins) is turned into the SSVEP signal-to-noise ratio of Norcia et al. (2015): the power
in a bin divided by the mean power of its neighbours within 0.375-1.5 Hz on both sides. A real
steady-state visual response is a narrow line with an SNR of several, often with a harmonic at
twice the frequency, present while the flicker is on and absent otherwise. Known hardware lines
(the Muse 2's 21.33 Hz Bluetooth packet rate and its multiples) are listed, not hidden.

Writes <out>/ssvep_scan_<dev>_<ch>.npz (snr, psd, usable, epoch clocks, f) and
<out>/ssvep_scan_summary.json (persistent lines per channel, transient peaks per epoch).

    python study/ssvep_scan.py --out study/participants/P-0001/nights/phone_20261028
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from night_channels import clean, FS  # noqa: E402

ANCHOR = pd.Timestamp("2026-10-28 03:42:00")
SRC = {
    "muse2": ("data/muselog/20261028/muse_session_20261028_034217_Muse1.npz", True, {"Oz": 5, "AF7": 1, "AF8": 2}),
    "athena": ("data/muselog/20261028/muse_session_20261028_034217_Muse2.npz", False, {"AF7": 1, "AF8": 2}),
}
# Hardware lines on this Muse 2 (12-sample BLE packets at 256 Hz = 21.33 Hz) and their relatives.
HARDWARE = [10.667, 21.333, 32.0, 42.667]
EPOCH = 30.0
NPERSEG = 2048  # 8 s -> 0.125 Hz bins


def snr_spectrum(P):
    """Norcia SSVEP SNR: bin power over the mean of bins 3..12 away (0.375-1.5 Hz) on both sides."""
    n = len(P)
    out = np.full(n, np.nan)
    for k in range(12, n - 12):
        nb = np.r_[P[k - 12:k - 2], P[k + 3:k + 13]]
        out[k] = P[k] / nb.mean()
    return out


def load(dev):
    path, devclock, chans = SRC[dev]
    z = np.load(path, allow_pickle=True)
    eeg = z["eeg"].astype(float)
    clock0 = pd.Timestamp(str(z["clock0"]))
    t = (np.arange(len(eeg)) / FS if devclock else z["t"]) + (clock0 - ANCHOR).total_seconds()
    return t, eeg, chans


def scan(dev, out):
    t, eeg, chans = load(dev)
    n_ep = int((t[-1] - t[0]) // EPOCH)
    f = None
    res = {}
    for ch, col in chans.items():
        snr, psd, usable, clocks = [], [], [], []
        for e in range(n_ep):
            e0 = t[0] + e * EPOCH
            i, j = np.searchsorted(t, e0), np.searchsorted(t, e0 + EPOCH)
            x = eeg[i:j, col]
            if len(x) < EPOCH * FS * 0.9:
                continue
            rail = float(((x < 5) | (x > 1640)).mean())
            y = clean(x - x.mean())
            ok = rail < 0.01 and 0.8 < y.std() < 60
            ff, P = welch(y, fs=FS, nperseg=NPERSEG, noverlap=NPERSEG // 2, window="hann")
            keep = (ff >= 1) & (ff <= 60)
            if f is None:
                f = ff[keep]
            s = snr_spectrum(P)
            snr.append(s[keep])
            psd.append(P[keep])
            usable.append(ok)
            clocks.append((ANCHOR + pd.Timedelta(seconds=e0 + EPOCH / 2)).strftime("%H:%M:%S"))
        snr, psd, usable = np.array(snr), np.array(psd), np.array(usable)
        np.savez_compressed(out / f"ssvep_scan_{dev}_{ch}.npz", f=f, snr=snr, psd=psd, usable=usable,
                            clock=np.array(clocks))
        res[ch] = dict(f=f, snr=snr, psd=psd, usable=usable, clock=np.array(clocks))
        print(f"{dev} {ch}: {usable.sum()} usable of {len(usable)} epochs")
    return res


def near_hardware(fr, tol=0.4):
    return any(abs(fr - h) <= tol for h in HARDWARE)


def summarise(dev, ch, r):
    f, snr, usable, clock = r["f"], r["snr"], r["usable"], r["clock"]
    if usable.sum() == 0:
        return {"usable_epochs": 0}
    S = snr[usable]
    med = np.nanmedian(S, axis=0)
    frac = np.nanmean(S > 3, axis=0)
    # persistent lines: local maxima of the median SNR spectrum above 3 (4.8 dB), 3-45 Hz
    lines = []
    band = (f >= 3) & (f <= 45)
    for k in np.where(band)[0]:
        if med[k] >= 3 and med[k] == np.nanmax(med[max(0, k - 4):k + 5]):
            lines.append({"hz": round(float(f[k]), 3), "median_snr_db": round(10 * np.log10(med[k]), 1),
                          "share_epochs_snr_gt3": round(float(frac[k]), 2), "hardware": near_hardware(f[k])})
    # transient peaks: per usable epoch, the strongest bin in 4-45 Hz away from the hardware lines
    mask = (f >= 4) & (f <= 45) & ~np.array([near_hardware(x) for x in f])
    peaks = []
    for e in np.where(usable)[0]:
        s = np.where(mask, snr[e], np.nan)
        k = int(np.nanargmax(s))
        if s[k] >= 10:
            k2 = int(np.argmin(abs(f - 2 * f[k])))
            peaks.append({"clock": str(clock[e]), "hz": round(float(f[k]), 3), "snr_db": round(10 * np.log10(s[k]), 1),
                          "harmonic_snr_db": round(10 * np.log10(snr[e][k2]), 1) if 2 * f[k] <= 60 else None})
    strongest = max(peaks, key=lambda p: p["snr_db"]) if peaks else None
    return {"usable_epochs": int(usable.sum()), "epochs": int(len(usable)), "persistent_lines": lines,
            "transient_peaks_snr_ge_10dB": len(peaks), "strongest_transient": strongest,
            "transient_examples": peaks[:12]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for dev in SRC:
        res = scan(dev, out)
        for ch, r in res.items():
            summary[f"{dev}_{ch}"] = summarise(dev, ch, r)
    (out / "ssvep_scan_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
