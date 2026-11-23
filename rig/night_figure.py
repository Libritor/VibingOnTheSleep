"""The night as one picture: hypnogram and stage probabilities from detector.csv, spectrograms of
the frontal EEG and of the fifth electrode (AUX, a cup at Oz) from raw.csv, the head angle from the
accelerometer, and the moments the triggers would have fired. Vector PDF with the text embedded,
plus a PNG.

    python rig/night_figure.py study/participants/P-0001/nights/<stamp>
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

FS = 256.0
STAGES = ("W", "N1", "N2", "N3", "REM")
ORDER = {"N3": 0, "N2": 1, "N1": 2, "REM": 3, "W": 4}


def read_raw(path):
    """(t, eeg frontal-vs-mastoid, aux or None, angle or None) downsampled to 64 Hz."""
    t, eeg, aux, acc = [], [], [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        head = next(r)
        ia = head.index("AUX") if "AUX" in head else None
        for i, row in enumerate(r):
            if i % 4:
                continue
            try:
                tp9, af7, af8, tp10 = map(float, row[1:5])
            except ValueError:
                continue
            t.append(float(row[0]))
            eeg.append((af7 + af8) / 2 - (tp9 + tp10) / 2)
            a = row[ia] if ia is not None else ""
            aux.append(float(a) - (tp9 + tp10) / 2 if a not in ("", "nan") else np.nan)
            try:
                acc.append([float(v) for v in row[-3:]])
            except ValueError:
                acc.append([np.nan] * 3)
    t, eeg, aux, acc = map(np.array, (t, eeg, aux, acc))
    angle = None
    if np.isfinite(acc).all(axis=1).any():
        ref = np.nanmedian(acc[:64 * 180], axis=0)        # the first three minutes = lying as calibrated
        dot = (acc @ ref) / (np.linalg.norm(acc, axis=1) * np.linalg.norm(ref) + 1e-9)
        angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    return t - t[0], eeg, (aux if np.isfinite(aux).any() else None), angle


def main(night):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7})
    night = Path(night)
    det = list(csv.DictReader(open(night / "detector.csv", newline="")))
    t_d = np.array([float(r["t_s"]) for r in det]) / 3600
    P = {s: np.array([float(r[f"p_{s}"]) if r[f"p_{s}"] else np.nan for r in det]) for s in STAGES}
    stage = np.array([ORDER.get(r["stage"], np.nan) if r["stage"] else np.nan for r in det], float)
    cues = [float(r["t_s"]) / 3600 for r in det if r["would_cue"]]
    events = list(csv.DictReader(open(night / "events.csv", newline="")))
    marks = {e["event"]: float(e["t_s"]) / 3600 for e in events
             if e["event"] in ("calibrated", "sleep_onset_trigger", "n2_guard")}
    t, eeg, aux, angle = read_raw(night / "raw.csv")
    hours = max(t[-1] / 3600, t_d[-1] if len(t_d) else 0, 0.1)

    n = 3 + (aux is not None) + (angle is not None)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.35 * n + 0.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.1, 1, 1.2] + [1.2] * (n - 3), hspace=0.35))
    ax = axes[0]
    ax.step(t_d, stage, where="post", color="#1f4e79", lw=0.9)
    ax.set_yticks(range(5)); ax.set_yticklabels(["N3", "N2", "N1", "REM", "W"]); ax.set_ylim(-0.5, 4.5)
    ax.set_ylabel("stage")
    for name, x in marks.items():
        ax.axvline(x, color="#c0392b", lw=0.8, ls="--")
        ax.text(x, 4.3, name.replace("_", " "), fontsize=6, color="#c0392b", rotation=90, va="top", ha="right")
    for x in cues:
        ax.axvline(x, color="#2e7d32", lw=0.8, alpha=0.7)
    ax.set_title(f"{night.parent.parent.name}, night {night.name}: detector argmax (dashed: triggers the "
                 f"rig would have fired; green: REM cues it would have played)", loc="left", fontsize=7.5)
    ax = axes[1]
    ax.plot(t_d, 1 - P["W"], color="#1f4e79", lw=0.8, label="P(sleep)")
    ax.plot(t_d, P["REM"], color="#2e7d32", lw=0.8, label="P(REM)")
    ax.plot(t_d, P["N2"] + P["N3"], color="#8e44ad", lw=0.8, label="P(N2+N3)")
    ax.set_ylim(0, 1); ax.set_ylabel("probability"); ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper right")

    def spec(ax, x, label):
        ok = np.isfinite(x)
        x = np.where(ok, x, 0.0)
        ax.specgram(x - np.nanmean(x), NFFT=512, Fs=64.0, noverlap=256, cmap="magma", vmin=-20, vmax=30,
                    xextent=(0, t[-1] / 3600))
        ax.set_ylim(0, 30); ax.set_ylabel(f"{label}\nHz")
    spec(axes[2], eeg, "frontal EEG")
    k = 3
    if aux is not None:
        spec(axes[k], aux, "AUX (Oz)"); k += 1
    if angle is not None:
        axes[k].plot(t / 3600, angle, color="#555", lw=0.6)
        axes[k].set_ylabel("head angle\n(deg)"); axes[k].set_ylim(0, 90); k += 1
    axes[-1].set_xlabel("hours since the recording started"); axes[-1].set_xlim(0, hours)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(night / "night.pdf", bbox_inches="tight")
    fig.savefig(night / "night.png", dpi=130, bbox_inches="tight")
    print("wrote", night / "night.pdf")
    summ = night / "summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        print({k: s[k] for k in ("minutes_scored", "stage_share", "would_be_rem_cues") if k in s})


if __name__ == "__main__":
    main(sys.argv[1])
