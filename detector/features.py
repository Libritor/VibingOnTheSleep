"""Muse-compatible sleep features.

Only two signals are used, both of which a Muse headband can approximate:
  eeg : one frontal EEG channel (Sleep-EDF Fpz-Cz; on a Muse, mean of AF7/AF8 against TP9/TP10)
  eog : horizontal EOG (Sleep-EDF 'EOG horizontal'; on a Muse, AF7 - AF8)
No chin EMG, no occipital channel: a Muse has neither. The same function runs offline on
Sleep-EDF and live on Muse windows, so the validation numbers apply to the live detector.
"""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12),
         "sigma": (12, 15), "beta": (15, 30)}


def band_feats(x, fs):
    f, p = welch(x, fs=fs, nperseg=int(4 * fs), noverlap=int(2 * fs))
    tot = p[(f >= 0.5) & (f <= 30)].sum() + 1e-12
    out = {}
    for name, (lo, hi) in BANDS.items():
        bp = p[(f >= lo) & (f < hi)].sum() + 1e-12
        out[f"log_{name}"] = np.log10(bp)
        out[f"rel_{name}"] = bp / tot
    out["log_total"] = np.log10(tot)
    out["alpha_theta"] = np.log10(out["rel_alpha"] / out["rel_theta"])
    out["delta_beta"] = np.log10(out["rel_delta"] / out["rel_beta"])
    c = np.cumsum(p[(f >= 0.5) & (f <= 30)]) / tot
    out["sef90"] = f[(f >= 0.5) & (f <= 30)][np.searchsorted(c, 0.9)]
    return out


def eog_feats(e, fs):
    sos_sem = butter(2, [0.2, 0.8], btype="band", fs=fs, output="sos")
    sos_rem = butter(2, [1.0, 5.0], btype="band", fs=fs, output="sos")
    sem = sosfiltfilt(sos_sem, e)
    rem = sosfiltfilt(sos_rem, e)
    d = np.diff(rem) * fs                      # slope, uV/s
    thr = 5 * np.median(np.abs(d)) / 0.6745 + 1e-9
    peaks = np.flatnonzero((np.abs(d[1:]) > thr) & (np.abs(d[:-1]) <= thr))
    return {"log_sem": np.log10(np.var(sem) + 1e-12),
            "log_rem": np.log10(np.var(rem) + 1e-12),
            "sem_rem": np.log10((np.var(sem) + 1e-12) / (np.var(rem) + 1e-12)),
            "rem_count": float(len(peaks)),
            "log_eog_absmax": np.log10(np.max(np.abs(e)) + 1e-9)}


def epoch_feats(eeg, eog, fs):
    eeg = np.asarray(eeg, float) - np.mean(eeg)   # Muse counts carry a DC offset
    eog = np.asarray(eog, float) - np.mean(eog)
    d = band_feats(eeg, fs)
    d.update(eog_feats(eog, fs))
    return d
