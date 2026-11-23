"""Thirty-second raw traces at chosen wall-clock moments, so a reader can see the waves the epoch
table counts: slow waves, spindles, eye movements, the 60 Hz on the Oz cup. Each panel shows one
channel of one device after the 60 Hz notch and 0.3 Hz high-pass (the raw sd is in the label).
Optionally overlays the same channel from a second device shifted by a lag, to show two
headbands seeing the same wave.

    python study/night_snapshots.py --out docs/night_2026-10-28_snapshots.pdf \
        --panel "muse2:AF8:04:40:30" --panel "athena:AF8:04:40:30:-0.77" ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from night_channels import clean, FS, NAMES, COLS  # noqa: E402

SRC = {
    "muse2": ("data/muselog/20261028/muse_session_20261028_034217_Muse1.npz", True),
    "athena": ("data/muselog/20261028/muse_session_20261028_034217_Muse2.npz", False),
    "muse2osc": ("study/participants/P-0001/nights/20261028_034419/raw.npz", True),
}
ANCHOR = pd.Timestamp("2026-10-28 03:42:00")
_cache = {}


def load(dev):
    if dev not in _cache:
        path, devclock = SRC[dev]
        z = np.load(path, allow_pickle=True)
        eeg = z["eeg"].astype(float)
        clock0 = pd.Timestamp(str(z["clock0"]))
        t = (np.arange(len(eeg)) / FS if devclock else z["t"]) + (clock0 - ANCHOR).total_seconds()
        _cache[dev] = (t, eeg)
    return _cache[dev]


def trace(dev, ch, clock, seconds=30, lag=0.0):
    t, eeg = load(dev)
    e0 = (pd.Timestamp("2026-10-28 " + clock) - ANCHOR).total_seconds() + lag
    i, j = np.searchsorted(t, e0 - 2), np.searchsorted(t, e0 + seconds + 2)
    x = eeg[i:j, COLS[NAMES.index(ch)]]
    y = clean(x - x.mean())
    k = int(2 * FS)
    return np.arange(len(y) - 2 * k) / FS, y[k:-k], float(x.std())


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7})
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True,
                    help="dev:channel:HH:MM:SS[:lag_s][:overlay_dev] e.g. muse2:AF8:04:40:30 or athena:AF8:04:40:30:-0.77:muse2")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    n = len(a.panel)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.25 * n + 0.4), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, spec in zip(axes, a.panel):
        parts = spec.split(":")
        dev, ch, clock = parts[0], parts[1], ":".join(parts[2:5])
        lag = float(parts[5]) if len(parts) > 5 else 0.0
        over = parts[6] if len(parts) > 6 else None
        tt, y, sd = trace(dev, ch, clock, a.seconds)
        ax.plot(tt, y, color="#1f4e79", lw=0.5, label=f"{dev} {ch}")
        if over:
            t2, y2, _ = trace(over, ch, clock, a.seconds, lag)
            ax.plot(t2, y2, color="#c0392b", lw=0.5, alpha=0.8, label=f"{over} {ch} shifted {lag:+.2f} s")
            ax.legend(fontsize=6, frameon=False, loc="upper right", ncol=2)
        ax.set_ylabel(f"{dev} {ch}\n{clock}\nuV", fontsize=6)
        ax.text(0.01, 0.95, f"raw sd {sd:.0f} uV, cleaned sd {y.std():.1f} uV", transform=ax.transAxes,
                fontsize=6, va="top", color="#555")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("seconds")
    axes[-1].set_xlim(0, a.seconds)
    fig.tight_layout()
    out = Path(a.out)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
