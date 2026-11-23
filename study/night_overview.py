"""The first two-headband night as one figure, from the per-epoch tables night_channels.py wrote
(Athena phone copy, Muse 2 phone copy to 05:23 then the rig's live copy to 07:53) and the
two-device agreement table. Panels: a rule-based state ribbon and the usability of every
electrode; spectrograms of the Muse 2's AF8 and of the Oz cup; relative delta of the three good
channels with the Oz slow-wave amplitude; spindles and eye movements; the epoch-by-epoch
agreement of the two headbands' delta waveforms; heart rate (raw optics and the SDK's), breathing
rate from both accelerometers; head angle and movement. Vector PDF with the text embedded.

State rule (a sketch, not a validated stager; the validated detector needs the mastoids, which
were noise all night): per epoch, the mean relative delta of the usable AF8 channels (both
devices): off = no usable forehead channel; move = accelerometer sd above 5 milli-g; W/N1 below
0.45; N2-like 0.45-0.70; N3-like 0.70 and above.

    python study/night_overview.py study/participants/P-0001/nights/phone_20261028
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import decimate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "study"))
from night_channels import clean, FS  # noqa: E402

ANCHOR = pd.Timestamp("2026-10-28 03:42:00")
STATES = ["off", "move", "W/N1", "N2-like", "N3-like"]
COLORS = {"off": "#d0d0d0", "move": "#f0b429", "W/N1": "#9ecae1", "N2-like": "#4292c6", "N3-like": "#08306b"}


def hours(t_s):
    return np.asarray(t_s) / 3600.0


def merged_muse2(out):
    P = pd.read_csv(out / "channels_muse2_phone.csv")
    L = pd.read_csv(out / "channels_muse2_osc.csv")
    return pd.concat([P, L[L.t_s > P.t_s.max()]], ignore_index=True).sort_values("t_s").reset_index(drop=True)


def state_table(A, M):
    T = pd.DataFrame({"t_s": sorted(set(A.t_s) | set(M.t_s))})
    a = A.set_index("t_s")
    m = M.set_index("t_s")
    rd, mv, nsp, eye = [], [], [], []
    for t in T.t_s:
        vals = []
        for D in (a, m):
            if t in D.index and D.at[t, "AF8_usable"] == 1 and np.isfinite(D.at[t, "AF8_rel_delta"]):
                vals.append(D.at[t, "AF8_rel_delta"])
        rd.append(np.mean(vals) if vals else np.nan)
        mv.append(np.nanmax([D.at[t, "move"] if t in D.index and "move" in D else np.nan for D in (a, m)]))
        nsp.append(np.nansum([D.at[t, "AF8_spindles"] if t in D.index and "AF8_spindles" in D else np.nan for D in (a, m)]))
        eye.append(np.nanmax([D.at[t, "EOG_eye"] if t in D.index and "EOG_eye" in D else np.nan for D in (a, m)]))
    T["rel_delta"], T["move"], T["spindles"], T["eog_eye"] = rd, mv, nsp, eye
    st = []
    for r in T.itertuples():
        if not np.isfinite(r.rel_delta):
            st.append("off")
        elif np.isfinite(r.move) and r.move > 5:
            st.append("move")
        elif r.rel_delta < 0.45:
            st.append("W/N1")
        elif r.rel_delta < 0.70:
            st.append("N2-like")
        else:
            st.append("N3-like")
    T["state"] = st
    T["clock"] = [(ANCHOR + pd.Timedelta(seconds=float(t))).strftime("%H:%M") for t in T.t_s]
    return T


def spectro(ax, npz, col, t_end_h, label, vmax=20):
    z = np.load(npz, allow_pickle=True)
    x = z["eeg"][:, col].astype(float)
    off = (pd.Timestamp(str(z["clock0"])) - ANCHOR).total_seconds()
    x = np.where(np.isfinite(x), x, 0.0)
    rail = (x < 5) | (x > 1640)
    y = clean(x - x.mean())
    y[rail] = 0.0
    y = np.clip(y, -150, 150)
    y64 = decimate(y, 4, zero_phase=True)
    ax.specgram(y64, NFFT=512, Fs=64.0, noverlap=384, cmap="magma", vmin=-15, vmax=vmax,
                xextent=(off / 3600, off / 3600 + len(y64) / 64 / 3600))
    ax.set_ylim(0, 30)
    ax.set_ylabel(f"{label}\nHz")


def main(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7})
    out = Path(out)
    A = pd.read_csv(out / "channels_athena_phone.csv")
    M = merged_muse2(out)
    D = pd.read_csv(out / "two_device_athena_muse2.csv")
    S = state_table(A, M)
    S.to_csv(out / "state_sketch.csv", index=False)
    t_max = max(A.t_s.max(), M.t_s.max()) + 30

    fig, axes = plt.subplots(8, 1, figsize=(7.2, 10.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.6, 1.1, 1.1, 1, 0.9, 0.8, 0.9, 0.8], hspace=0.28))
    # (a) state ribbon + usability matrix
    ax = axes[0]
    rows = [("Athena", A, ["TP9", "AF7", "AF8", "TP10"]), ("Muse 2", M, ["TP9", "AF7", "AF8", "TP10", "Oz"])]
    labels, mat = [], []
    for dev, T, chs in rows:
        for ch in chs:
            labels.append(f"{dev} {ch}")
            v = np.full(int(t_max // 30), np.nan)
            idx = (T.t_s // 30).astype(int)
            v[idx] = T[f"{ch}_usable"].to_numpy()
            mat.append(v)
    mat = np.array(mat)
    ax.imshow(mat, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#e8e8e8", "#2e7d32"]), vmin=0, vmax=1,
              extent=(0, t_max / 3600, len(labels) + 0.5, 0.5), interpolation="nearest")
    for k, r in S.iterrows():
        ax.add_patch(plt.Rectangle((r.t_s / 3600, -0.9), 30 / 3600, 1.2, color=COLORS[r.state], lw=0))
    ax.set_ylim(len(labels) + 0.5, -1.0)
    ax.set_yticks([-0.3] + list(range(1, len(labels) + 1)))
    ax.set_yticklabels(["state"] + labels, fontsize=6)
    ax.legend(handles=[Patch(color=COLORS[s], label=s) for s in STATES] + [Patch(color="#2e7d32", label="electrode usable")],
              fontsize=5.5, frameon=False, ncol=6, loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title("P-0001, night of 2026-10-28: Muse Athena and Muse 2 (with an Oz cup) worn together; "
                 "phone recordings", loc="left", fontsize=7.5, pad=14)
    # (b), (c) spectrograms from the rig's live copy of the Muse 2 (device clock)
    live = ROOT / "study/participants/P-0001/nights/20261028_034419/raw.npz"
    spectro(axes[1], live, 2, t_max / 3600, "Muse 2 AF8")
    spectro(axes[2], live, 5, t_max / 3600, "Oz cup", vmax=25)
    # (d) relative delta
    ax = axes[3]
    for T, ch, lab, col in ((A, "AF8", "Athena AF8", "#c0392b"), (M, "AF8", "Muse 2 AF8", "#1f4e79"), (M, "Oz", "Oz cup", "#2e7d32")):
        g = T[T[f"{ch}_usable"] == 1]
        ax.plot(hours(g.t_s), g[f"{ch}_rel_delta"], ".", ms=2, color=col, label=lab)
    ax.set_ylim(0, 1); ax.set_ylabel("relative delta\n0.5-4 Hz")
    ax2 = ax.twinx()
    g = M[M.Oz_usable == 1]
    ax2.plot(hours(g.t_s), g.Oz_sw_p90, "-", lw=0.7, color="#7f8c8d", label="Oz slow-wave p-p (90th pct)")
    ax2.set_ylabel("uV", color="#7f8c8d"); ax2.set_ylim(0, 160)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=3)
    ax2.legend(fontsize=5.5, frameon=False, loc="upper right")
    # (e) spindles and eye movements
    ax = axes[4]
    for T, lab, col, w in ((A, "Athena AF8 spindles", "#c0392b", 30 / 3600), (M, "Muse 2 AF8 spindles", "#1f4e79", 30 / 3600)):
        g = T[T.AF8_usable == 1]
        ax.bar(hours(g.t_s) + (w / 2 if lab.startswith("Muse") else 0), g.AF8_spindles, width=w / 2, color=col, label=lab)
    ax.set_ylabel("spindles\nper 30 s"); ax.set_ylim(0, 6)
    ax2 = ax.twinx()
    for T, lab, col in ((A, "Athena EOG eye movements", "#e67e22"), (M, "Muse 2 EOG eye movements", "#8e44ad")):
        if "EOG_eye" in T:
            g = T.dropna(subset=["EOG_eye"])
            ax2.plot(hours(g.t_s), g.EOG_eye, "-", lw=0.6, color=col, label=lab, alpha=0.8)
    ax2.set_ylabel("eye moves\nper 30 s"); ax2.set_ylim(0, 25)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=2)
    ax2.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=2)
    # (f) two-device agreement
    ax = axes[5]
    null = json.loads((out / "two_device_athena_muse2.json").read_text())["offset_epoch_null"]
    for ch, col in (("AF8", "#1f4e79"), ("AF7", "#c0392b")):
        g = D.dropna(subset=[f"{ch}_r"])
        ax.plot(hours(g.t_s), g[f"{ch}_r"].abs(), ".", ms=2.5, color=col, label=f"{ch}: |r| of the two headbands' 0.5-4 Hz waveforms")
        if ch in null:
            ax.axhline(null[ch]["median_r"], color=col, lw=0.6, ls="--")
    ax.text(0.01, 0.05, "dashed: the same statistic for epochs 60 s apart (no shared signal)", transform=ax.transAxes, fontsize=5.5, color="#555")
    ax.set_ylim(0, 1); ax.set_ylabel("Athena vs Muse 2\nagreement")
    ax.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=2)
    # (g) heart and breathing
    ax = axes[6]
    g = A.dropna(subset=["hr_ppg"])
    g = g[g.hr_ppg_snr >= 10]
    ax.plot(hours(g.t_s), g.hr_ppg, ".", ms=2.5, color="#c0392b", label="heart rate, Athena raw optics (this analysis)")
    g = A.dropna(subset=["hr"])
    ax.plot(hours(g.t_s), g.hr, "-", lw=0.6, color="#e6a0a0", label="heart rate, SDK's HR_BPM")
    ax.set_ylim(40, 100); ax.set_ylabel("beats/min")
    ax2 = ax.twinx()
    for T, lab, col in ((A, "breathing, Athena IMU", "#2e7d32"), (M, "breathing, Muse 2 IMU", "#1f4e79")):
        g = T.dropna(subset=["breath_bpm"])
        g = g[g.breath_snr >= 10]
        ax2.plot(hours(g.t_s), g.breath_bpm, ".", ms=1.8, color=col, label=lab, alpha=0.7)
    ax2.set_ylim(8, 24); ax2.set_ylabel("breaths/min")
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=2)
    ax2.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=2)
    # (h) head angle and movement
    ax = axes[7]
    for T, lab, col in ((A, "Athena", "#c0392b"), (M, "Muse 2", "#1f4e79")):
        g = T.dropna(subset=["head_angle"])
        ax.plot(hours(g.t_s), g.head_angle, "-", lw=0.7, color=col, label=f"head angle, {lab}")
    ax.set_ylim(0, 120); ax.set_ylabel("head angle\n(deg)")
    ax2 = ax.twinx()
    g = M.dropna(subset=["move"])
    ax2.plot(hours(g.t_s), g.move, "-", lw=0.5, color="#7f8c8d", label="movement (sd |acc|, milli-g)")
    ax2.set_ylim(0, 20); ax2.set_ylabel("milli-g", color="#7f8c8d")
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=2)
    ax2.legend(fontsize=5.5, frameon=False, loc="upper right")
    ticks = np.arange(0, t_max / 3600 + 0.01, 0.5)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([(ANCHOR + pd.Timedelta(hours=h)).strftime("%H:%M") for h in ticks])
    axes[-1].set_xlim(0, t_max / 3600)
    axes[-1].set_xlabel("wall clock")
    for a in axes:
        a.spines[["top"]].set_visible(False)
    fig.savefig(out / "night_2026-10-28.pdf", bbox_inches="tight")
    fig.savefig(out / "night_2026-10-28.png", dpi=130, bbox_inches="tight")

    # digest
    mins = S.state.value_counts().reindex(STATES).fillna(0) * 0.5
    runs = (S.state != S.state.shift()).cumsum()
    lines = [f"{g.clock.iloc[0]}  {g.state.iloc[0]:8s} {len(g) * 0.5:5.1f} min  rel delta {g.rel_delta.mean():.2f}"
             for _, g in S.groupby(runs)]
    summary = dict(minutes_per_state=mins.round(1).to_dict(), runs=lines)
    # sleep onset: first of three consecutive epochs at or above 0.45 after 03:50
    seq = S[(S.t_s >= 8 * 60)].reset_index(drop=True)
    on = None
    for k in range(len(seq) - 2):
        if all(seq.rel_delta.iloc[k:k + 3] >= 0.45):
            on = seq.clock.iloc[k]
            break
    deep = None
    for k in range(len(seq) - 2):
        if all(seq.rel_delta.iloc[k:k + 3] >= 0.70):
            deep = seq.clock.iloc[k]
            break
    summary["first_sustained_N2_like"] = on
    summary["first_sustained_N3_like"] = deep
    g = A[(A.hr_ppg_snr >= 10)] if "hr_ppg_snr" in A else A.iloc[:0]
    summary["hr_ppg_by_period"] = {
        "03:42-03:50 settling": float(g[g.t_s < 8 * 60].hr_ppg.median()) if len(g) else None,
        "03:50-03:58 lying awake": float(g[(g.t_s >= 8 * 60) & (g.t_s < 16 * 60)].hr_ppg.median()) if len(g) else None,
        "04:00-04:30 slow-wave sleep": float(g[(g.t_s >= 18 * 60) & (g.t_s < 48 * 60)].hr_ppg.median()) if len(g) else None,
        "04:32-05:02 after the turn": float(g[(g.t_s >= 50 * 60)].hr_ppg.median()) if len(g) else None}
    (out / "night_2026-10-28_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", out / "night_2026-10-28.pdf")


if __name__ == "__main__":
    main(sys.argv[1])
