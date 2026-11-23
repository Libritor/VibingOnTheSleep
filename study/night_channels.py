"""What each electrode of a night recording actually carries, 30 s at a time, with no classifier in
the way. For every channel (TP9, AF7, AF8, TP10 and the fifth electrode, each referenced to the
headband's Fpz) and every 30-s epoch on the wall clock:

  quality   raw sd, railed share, mains (58-62 Hz) power, HSI when the phone logged it, and a
            'usable' flag: no railing, and 0.8-60 uV sd after a 60 Hz notch and a 0.3 Hz high-pass
  spectrum  absolute and relative delta/theta/alpha/sigma/beta, the alpha peak (Hz, dB above 1/f)
  events    spindles (11-16 Hz bursts 0.5-2 s above 3x the channel's median envelope), slow
            waves (0.5-2 Hz half-waves, p-p above --sw-uv; also the 90th pct p-p), eye movements
            (steep 1-5 Hz deflections, the same rule as the detector's EOG count) on the single
            channel and, when AF7 and AF8 are both usable, on AF7-AF8
  body      head angle versus the first ten minutes, movement (sd of |acc|), breathing rate from
            the accelerometer when still (0.1-0.6 Hz peak), and on an Athena the heart rate,
            RMSSD and battery the phone logged

Timing: --device-clock trusts the sample count (t = clock0 + i / 256), which is right when the
phone buffered the Bluetooth stream (MuseLog's CLOCK_TIME then lags by minutes); otherwise the
phone's timestamps are used. Epochs are anchored to the whole minute of --anchor.

    python study/night_channels.py data/muselog/20261028/muse_session_20261028_034217_Muse1.npz \
        --label muse2 --device-clock --out study/participants/P-0001/nights/phone_20261028
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, iirnotch, sosfiltfilt, filtfilt, welch

FS = 256.0
NAMES = ["TP9", "AF7", "AF8", "TP10", "Oz"]       # Oz = AUX_RIGHT (the micro-USB electrode on a Muse 2)
COLS = [0, 1, 2, 3, 5]
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 15), "beta": (15, 30)}


def clean(x, fs=FS):
    """60 Hz (and 120 Hz) notch, 0.3 Hz high-pass."""
    for f0 in (60.0, 120.0):
        if f0 < fs / 2:
            b, a = iirnotch(f0, 30.0, fs)
            x = filtfilt(b, a, x)
    sos = butter(2, 0.3, btype="high", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def alpha_peak(f, P):
    """Peak in 7-13 Hz above a 1/f line fitted on 3-30 Hz outside that range: (Hz, dB)."""
    m = (f >= 3) & (f <= 30) & ~((f > 7) & (f < 13))
    co = np.polyfit(np.log10(f[m]), np.log10(P[m] + 1e-12), 1)
    a = (f >= 7) & (f <= 13)
    resid = 10 * (np.log10(P[a] + 1e-12) - np.polyval(co, np.log10(f[a])))
    k = int(np.argmax(resid))
    return float(f[a][k]), float(resid[k])


def spindles(x, fs, thr):
    """Bursts of the 11-16 Hz envelope above thr for 0.5-2 s: (count, mean peak amplitude uV)."""
    env = np.abs(hilbert(sosfiltfilt(butter(4, [11, 16], btype="band", fs=fs, output="sos"), x)))
    env = np.convolve(env, np.ones(int(0.1 * fs)) / int(0.1 * fs), mode="same")
    above = env > thr
    edges = np.flatnonzero(np.diff(above.astype(int)))
    starts = edges[::2] if above[0] == 0 else np.r_[0, edges[1::2]]
    ends = edges[1::2] if above[0] == 0 else edges[::2]
    n = min(len(starts), len(ends))
    dur = (ends[:n] - starts[:n]) / fs
    ok = (dur >= 0.5) & (dur <= 2.0)
    amps = [env[s:e].max() for s, e, o in zip(starts[:n], ends[:n], ok) if o]
    return int(ok.sum()), (float(np.mean(amps)) if amps else 0.0)


def sigma_env_median(x, fs):
    env = np.abs(hilbert(sosfiltfilt(butter(4, [11, 16], btype="band", fs=fs, output="sos"), x)))
    return float(np.median(env))


def slow_waves(x, fs, thr_uv):
    """0.3-4 Hz half-waves 0.25-1 s long (0.5-2 Hz): count with p-p above thr, 90th pct p-p."""
    y = sosfiltfilt(butter(2, [0.3, 4], btype="band", fs=fs, output="sos"), x)
    zc = np.flatnonzero(np.diff(np.signbit(y).astype(int)) != 0)
    pp = []
    for s, e in zip(zc[:-1], zc[1:]):
        d = (e - s) / fs
        if 0.25 <= d <= 1.0:
            seg = y[s:e]
            nxt = y[e:min(len(y), e + int(fs))]
            pp.append(abs(seg[np.argmax(np.abs(seg))]) + (abs(nxt[np.argmax(np.abs(nxt))]) if len(nxt) else 0))
    pp = np.array(pp)
    return int((pp > thr_uv).sum()), (float(np.percentile(pp, 90)) if len(pp) else 0.0)


def eye_moves(e, fs):
    """The detector's rapid-eye-movement count: steep 1-5 Hz deflections above 5 MAD of the slope."""
    rem = sosfiltfilt(butter(2, [1.0, 5.0], btype="band", fs=fs, output="sos"), e)
    d = np.diff(rem) * fs
    thr = 5 * np.median(np.abs(d)) / 0.6745 + 1e-9
    peaks = np.flatnonzero((np.abs(d[1:]) > thr) & (np.abs(d[:-1]) <= thr))
    sem = sosfiltfilt(butter(2, [0.2, 0.8], btype="band", fs=fs, output="sos"), e)
    return int(len(peaks)), float(np.log10(np.var(sem) + 1e-12))


def breathing(a, fs_a):
    """Dominant 0.1-0.6 Hz rhythm of the accelerometer axis with the most slow variance:
    (breaths per minute, peak-to-median ratio of its spectrum)."""
    a = a - a.mean(0)
    sos = butter(2, [0.08, 0.8], btype="band", fs=fs_a, output="sos")
    s = sosfiltfilt(sos, a, axis=0)
    ax = int(np.argmax(s.var(0)))
    f, P = welch(s[:, ax], fs=fs_a, nperseg=int(20 * fs_a), noverlap=int(10 * fs_a), nfft=int(160 * fs_a))
    m = (f >= 0.1) & (f <= 0.6)
    k = int(np.argmax(P[m]))
    return float(f[m][k] * 60), float(P[m][k] / (np.median(P[m]) + 1e-18))


def ppg_rate(O, fs_o):
    """Heart rate from the Athena's raw optical channels (730/850 nm, inner and outer, both sides):
    the 0.7-2.5 Hz spectral peak of each channel, the median across channels and its worst
    peak-to-median ratio. The SDK's own HR_BPM read 1.3x too high for most of the first night."""
    O = O - O.mean(0)
    sos = butter(2, [0.6, 3.0], btype="band", fs=fs_o, output="sos")
    bpm, snr = [], []
    for k in (0, 1, 2, 3, 4, 5, 6, 7):                   # the eight 730/850 nm channels
        if O[:, k].std() < 1e-6:
            continue
        y = sosfiltfilt(sos, O[:, k])
        f, P = welch(y, fs=fs_o, nperseg=int(20 * fs_o), noverlap=int(10 * fs_o), nfft=int(160 * fs_o))
        m = (f >= 0.7) & (f <= 2.5)
        bpm.append(f[m][int(np.argmax(P[m]))] * 60)
        snr.append(P[m].max() / (np.median(P[m]) + 1e-18))
    if not bpm:
        return None
    return float(np.median(bpm)), float(np.min(snr)), float(np.ptp(bpm))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device-clock", action="store_true", help="time by sample count, not phone time")
    ap.add_argument("--anchor", default=None, help="wall clock the epoch grid starts at (default: clock0's minute)")
    ap.add_argument("--sw-uv", type=float, default=20.0, help="slow-wave p-p threshold, uV (Fpz-referenced)")
    a = ap.parse_args(argv)
    z = np.load(a.npz, allow_pickle=True)
    eeg, hsi = z["eeg"].astype(float), z["hsi"]
    clock0 = pd.Timestamp(str(z["clock0"]))
    anchor = pd.Timestamp(a.anchor) if a.anchor else clock0.floor("min")
    t = (np.arange(len(eeg)) / FS if a.device_clock else z["t"]) + (clock0 - anchor).total_seconds()
    acc_rows = z["acc"]
    misc = z["misc"]
    if len(acc_rows):
        acc_t = (np.arange(len(acc_rows)) / FS if (a.device_clock and len(acc_rows) == len(eeg)) else acc_rows[:, 0]) \
            + (clock0 - anchor).total_seconds()
        acc = acc_rows[:, 1:]
        ref_up = np.nanmedian(acc, axis=0)            # the night's usual lying posture = 0 degrees
    else:
        acc_t, acc = None, None
    if len(misc):
        misc_t = (np.arange(len(misc)) / FS if (a.device_clock and len(misc) == len(eeg)) else misc[:, 0]) \
            + (clock0 - anchor).total_seconds()
    optics = z["optics"]
    if len(optics):
        opt_t = (np.arange(len(optics)) / FS if (a.device_clock and len(optics) == len(eeg)) else optics[:, 0]) \
            + (clock0 - anchor).total_seconds()

    # channel-wide spindle thresholds: 3x the median sigma envelope over the recording's quiet samples
    thr = {}
    for c, name in zip(COLS, NAMES):
        x = eeg[:, c]
        if not np.isfinite(x).all() or x.std() == 0:
            continue
        step = int(FS * 30)
        meds = []
        for s in range(0, len(x) - step, step * 4):
            seg = x[s:s + step]
            if ((seg < 5) | (seg > 1640)).mean() < 0.01 and 0.8 < clean(seg - seg.mean()).std() < 60:
                meds.append(sigma_env_median(clean(seg - seg.mean()), FS))
        thr[name] = 3 * float(np.median(meds)) if meds else np.inf

    rows = []
    e_end = int(np.ceil(t[-1] / 30)) * 30
    for e0 in np.arange(np.floor(t[0] / 30) * 30, e_end, 30.0):
        i, j = np.searchsorted(t, e0), np.searchsorted(t, e0 + 30)
        if j - i < 20 * FS:
            continue
        row = dict(t_s=float(e0), clock=(anchor + pd.Timedelta(seconds=e0)).strftime("%H:%M:%S"), n=int(j - i))
        usable = {}
        cleaned = {}
        for c, name in zip(COLS, NAMES):
            x = eeg[i:j, c]
            if not np.isfinite(x).all() or x.std() == 0:
                row[f"{name}_usable"] = 0
                continue
            rail = float(((x < 5) | (x > 1640)).mean())
            y = clean(x - x.mean())
            f, P = welch(y, fs=FS, nperseg=int(4 * FS), noverlap=int(2 * FS))
            df = f[1] - f[0]
            f0, P0 = welch(x - x.mean(), fs=FS, nperseg=int(4 * FS), noverlap=int(2 * FS))
            mains = float(P0[(f0 >= 58) & (f0 <= 62)].sum() * df)
            ok = rail < 0.01 and 0.8 < y.std() < 60
            usable[name] = ok
            cleaned[name] = y
            row[f"{name}_sd_raw"] = round(float(x.std()), 1)
            row[f"{name}_sd"] = round(float(y.std()), 2)
            row[f"{name}_rail"] = round(rail, 3)
            row[f"{name}_mains"] = round(mains, 1)
            row[f"{name}_usable"] = int(ok)
            ci = NAMES.index(name)
            if ci < 4 and np.isfinite(hsi[i:j, ci]).any():
                row[f"{name}_hsi"] = float(np.nanmedian(hsi[i:j, ci]))
            if not ok:
                continue
            tot = P[(f >= 0.5) & (f <= 30)].sum() * df + 1e-12
            for b, (lo, hi) in BANDS.items():
                bp = P[(f >= lo) & (f < hi)].sum() * df
                row[f"{name}_{b}"] = round(float(bp), 2)
                row[f"{name}_rel_{b}"] = round(float(bp / tot), 3)
            row[f"{name}_total"] = round(float(tot), 2)
            pk_hz, pk_db = alpha_peak(f, P)
            row[f"{name}_alpha_hz"], row[f"{name}_alpha_db"] = round(pk_hz, 2), round(pk_db, 1)
            n_sp, amp_sp = spindles(y, FS, thr.get(name, np.inf))
            row[f"{name}_spindles"], row[f"{name}_spindle_uv"] = n_sp, round(amp_sp, 1)
            n_sw, p90 = slow_waves(y, FS, a.sw_uv)
            row[f"{name}_sw"], row[f"{name}_sw_p90"] = n_sw, round(p90, 1)
            n_em, _ = eye_moves(y, FS)
            row[f"{name}_eye"] = n_em
        if usable.get("AF7") and usable.get("AF8"):
            n_em, log_sem = eye_moves(cleaned["AF7"] - cleaned["AF8"], FS)
            row["EOG_eye"], row["EOG_log_sem"] = n_em, round(log_sem, 2)
            row["EOG_sd"] = round(float((cleaned["AF7"] - cleaned["AF8"]).std()), 2)
        if acc is not None:
            ia, ja = np.searchsorted(acc_t, e0), np.searchsorted(acc_t, e0 + 30)
            A = acc[ia:ja]
            A = A[np.isfinite(A).all(1)]
            if len(A) > 100:
                mag = np.linalg.norm(A, axis=1)
                med = np.median(A, axis=0)
                cosang = med @ ref_up / (np.linalg.norm(med) * np.linalg.norm(ref_up) + 1e-9)
                row["head_angle"] = round(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), 1)
                row["move"] = round(float(mag.std() * 1000), 2)             # milli-g
                if row["move"] < 8:
                    # the IMU is forward-filled onto EEG samples; average to 8 Hz before the spectrum
                    nb = len(A) // 32
                    A8 = A[:nb * 32].reshape(nb, 32, 3).mean(1)
                    fs_a = len(A) / 30.0 / 32
                    bpm, ratio = breathing(A8, fs_a)
                    row["breath_bpm"], row["breath_snr"] = round(bpm, 1), round(ratio, 1)
        if len(misc):
            im, jm = np.searchsorted(misc_t, e0), np.searchsorted(misc_t, e0 + 30)
            M = misc[im:jm]
            if len(M):
                for k, nm in ((1, "battery"), (2, "hr"), (3, "rmssd")):
                    v = M[:, k]
                    if np.isfinite(v).any():
                        row[nm] = round(float(np.nanmedian(v)), 1)
        if len(optics):
            io, jo = np.searchsorted(opt_t, e0), np.searchsorted(opt_t, e0 + 30)
            O = optics[io:jo, 1:]
            if len(O) > 20 * FS and np.isfinite(O).all():
                nb = len(O) // 8                           # forward-filled onto 256 Hz rows: average to 32 Hz
                r = ppg_rate(O[:nb * 8].reshape(nb, 8, 16).mean(1), len(O) / 30.0 / 8)
                if r:
                    row["hr_ppg"], row["hr_ppg_snr"], row["hr_ppg_spread"] = round(r[0], 1), round(r[1], 1), round(r[2], 1)
        rows.append(row)
    T = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / f"channels_{a.label}.csv", index=False)
    meta = dict(npz=str(a.npz), label=a.label, clock0=clock0.isoformat(), anchor=anchor.isoformat(),
                device_clock=a.device_clock, spindle_thr_uv={k: round(v, 2) for k, v in thr.items()},
                sw_uv=a.sw_uv, epochs=len(T))
    (out / f"channels_{a.label}.json").write_text(json.dumps(meta, indent=2))
    print(f"{a.label}: {len(T)} epochs from {T.clock.iloc[0]} to {T.clock.iloc[-1]}; spindle thresholds {meta['spindle_thr_uv']}")
    print("usable epochs per channel:", {n: int(T[f'{n}_usable'].sum()) for n in NAMES if f"{n}_usable" in T})
    # ten-minute digest
    print(f"{'clock':>8} " + " ".join(f"{n:>4}" for n in NAMES) + f" | {'ch':>3} {'delta':>5} {'sigma':>5} {'a_dB':>5} {'spin':>4} {'sw':>4} {'eye':>4} {'EOGe':>4} | {'ang':>4} {'mov':>5} {'br':>5} {'HR':>4}")
    for k in range(0, len(T), 20):
        g = T.iloc[k:k + 20]
        us = " ".join(f"{g[f'{n}_usable'].mean():4.2f}" if f"{n}_usable" in g else "   -" for n in NAMES)
        best = None
        for n in ("AF8", "AF7", "Oz"):
            if f"{n}_rel_delta" in g and g[f"{n}_rel_delta"].notna().sum() >= 5:
                best = n
                break
        if best:
            dl, sg = g[f"{best}_rel_delta"].median(), g[f"{best}_rel_sigma"].median()
            adb, sp, sw, ey = g[f"{best}_alpha_db"].median(), g[f"{best}_spindles"].sum(), g[f"{best}_sw"].sum(), g[f"{best}_eye"].sum()
        else:
            dl = sg = adb = sp = sw = ey = np.nan
        eo = g["EOG_eye"].sum() if "EOG_eye" in g else np.nan
        ang = g["head_angle"].median() if "head_angle" in g else np.nan
        mv = g["move"].median() if "move" in g else np.nan
        br = g["breath_bpm"].median() if "breath_bpm" in g else np.nan
        hr = g["hr"].median() if "hr" in g else np.nan
        print(f"{g.clock.iloc[0]:>8} {us} | {best or '-':>3} {dl:5.2f} {sg:5.3f} {adb:5.1f} {sp:4.0f} {sw:4.0f} {ey:4.0f} {eo:4.0f} | {ang:4.0f} {mv:5.1f} {br:5.1f} {hr:4.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
