"""Figure and state table for the steady-state scan (ssvep_scan.py). A visual steady-state
response must be present while the eyes are open on a flickering display and absent while the
eyes are closed and during sleep; a hardware or environmental line is present regardless. The
figure shows (a) the occipital cup's spectrum in two eyes-closed states, (b) the forehead
spectra while sitting with the phone, (c) the night-long time course of the three narrow lines
the scan found, against the state bands.

    python study/ssvep_report.py --in study/participants/P-0001/nights/phone_20261028 \
        --out study/participants/P-0001/nights/phone_20261028/ssvep_scan.pdf
"""
import argparse
import json
from pathlib import Path

import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
STATES = [("sitting, phone", "03:42:30", "03:48:00"), ("lying, eyes closed", "03:51:00", "03:56:30"),
          ("N3 sleep", "04:40:00", "05:00:00")]
# The Oz cup was in contact 03:50:30-04:29 only, so its sleep comparison uses the N2/N3 stretch before the turn.
OZ_STATES = [("lying, eyes closed", "03:51:00", "03:56:30"), ("asleep (N2/N3)", "04:16:00", "04:29:00")]
LINES = [("24.0 Hz", "muse2_AF7", 24.0, STATES), ("40.0 Hz", "athena_AF8", 40.0, STATES),
         ("44.0 Hz", "muse2_Oz", 44.0, [STATES[0]] + OZ_STATES)]


def secs(clock):
    h, m, s = clock.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load(indir, key):
    z = np.load(Path(indir) / f"ssvep_scan_{key}.npz")
    t = np.array([secs(str(c)) for c in z["clock"]])
    return z["f"], z["snr"], z["usable"], t


def state_median(f, snr, usable, t, a, b):
    m = usable & (t >= secs(a)) & (t <= secs(b))
    return (np.nanmedian(snr[m], axis=0) if m.sum() else None), int(m.sum())


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.5, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID})
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    keys = ["muse2_Oz", "muse2_AF7", "muse2_AF8", "athena_AF7", "athena_AF8"]
    D = {k: load(a.indir, k) for k in keys}

    # state table: median SNR (dB) of each line in each state, per channel
    table = {}
    for name, key, hz, states in LINES:
        f, snr, usable, t = D[key]
        k = int(np.argmin(abs(f - hz)))
        row = {}
        for st, s0, s1 in states:
            med, n = state_median(f, snr, usable, t, s0, s1)
            row[st] = {"snr_db": None if med is None else round(float(10 * np.log10(med[k])), 1), "epochs": n}
        table[f"{name} on {key}"] = row
    print(json.dumps(table, indent=1))
    Path(a.out).with_name("ssvep_state_table.json").write_text(json.dumps(table, indent=1), encoding="utf-8")

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6))
    for ax in axes:
        ax.grid(True, color=GRID, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2)

    # (a) Oz cup, eyes-closed states
    ax = axes[0]
    f, snr, usable, t = D["muse2_Oz"]
    for (st, s0, s1), col in zip(OZ_STATES, (BLUE, ORANGE)):
        med, n = state_median(f, snr, usable, t, s0, s1)
        if med is not None:
            ax.plot(f, 10 * np.log10(med), color=col, lw=1.2, label=f"{st}, {s0[:5]}-{s1[:5]} ({n} epochs)")
    ax.set_xlim(2, 50)
    ax.set_ylabel("steady-state SNR, dB")
    ax.set_title("(a) Oz cup (the only electrode over visual cortex): eyes closed, median spectrum",
                 loc="left", fontsize=8, color=INK)
    ax.annotate("10 Hz alpha (eyes closed)", (10.1, 3), xytext=(13, 8), fontsize=7, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.annotate("44.0 Hz line, present in every epoch,\nasleep or awake: hardware, not visual", (43.6, 21),
                xytext=(27, 16), fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.text(0.01, 0.04, "not in contact while sitting with the phone (03:42-03:50)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7, color=INK2)
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    # (b) forehead, sitting with the phone
    ax = axes[1]
    s0, s1 = STATES[0][1], STATES[0][2]
    for key, col, lab in (("muse2_AF7", BLUE, "Muse 2 AF7"), ("athena_AF7", ORANGE, "Athena AF7"),
                          ("athena_AF8", AQUA, "Athena AF8")):
        f, snr, usable, t = D[key]
        med, n = state_median(f, snr, usable, t, s0, s1)
        if med is not None:
            ax.plot(f, 10 * np.log10(med), color=col, lw=1.2, label=f"{lab} ({n} epochs)")
    ax.set_xlim(2, 50)
    ax.set_ylabel("steady-state SNR, dB")
    ax.set_title("(b) forehead electrodes while sitting with the phone, 03:42-03:48, median spectrum",
                 loc="left", fontsize=8, color=INK)
    for hz, txt, ha in ((21.33, "21.3 Hz Bluetooth ", "right"), (24.0, " 24 Hz", "left"),
                        (40.0, "40 Hz ", "right"), (42.67, " 42.7 Hz Bluetooth", "left")):
        ax.axvline(hz, color=GRID, lw=0.8, zorder=0)
        ax.text(hz, 29, txt, fontsize=6.5, color=INK2, ha=ha, va="bottom")
    ax.set_ylim(top=34)
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    # (c) time course of the three lines
    ax = axes[2]
    for (name, key, hz, _), col in zip(LINES, (BLUE, ORANGE, AQUA)):
        f, snr, usable, t = D[key]
        k = int(np.argmin(abs(f - hz)))
        y = np.where(usable, 10 * np.log10(snr[:, k]), np.nan)
        ax.plot((t - secs("03:42:00")) / 60, y, color=col, lw=1.2, label=f"{name} on {key.replace('_', ' ')}")
    for (st, s0, s1), yy, ha in zip(STATES, (27, 30, 27), ("left", "left", "center")):
        x0, x1 = (secs(s0) - secs("03:42:00")) / 60, (secs(s1) - secs("03:42:00")) / 60
        ax.axvspan(x0, x1, color=GRID, alpha=0.6, lw=0, zorder=0)
        ax.text(x0 if ha == "left" else (x0 + x1) / 2, yy, st, ha=ha, va="bottom", fontsize=6.5, color=INK2)
    ax.axhline(4.8, color=INK2, lw=0.6, ls=(0, (3, 3)))
    ax.text(114, 5.2, "SNR 3 (4.8 dB)", fontsize=6.5, color=INK2, ha="right")
    ax.set_xlim(0, 115)
    ax.set_ylim(-3, 33)
    ax.set_xlabel("minutes after 03:42")
    ax.set_ylabel("line SNR, dB")
    ax.set_title("(c) the three narrow lines through the night: none switches off with the eyes",
                 loc="left", fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.88))

    fig.suptitle("Steady-state scan of the 2026-10-28 night: no flicker-driven visual response",
                 x=0.01, ha="left", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(a.out)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
