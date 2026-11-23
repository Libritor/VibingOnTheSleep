"""The live detector must reproduce the LOSO validation for a held-out night."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, joblib, mne
from model import LiveDetector, ROOT
mne.set_log_level("ERROR")
b = joblib.load(ROOT / "models" / "nap_base3min_minus_subject0.joblib")
loso = pd.read_csv(ROOT / "data" / "loso_predictions_base6.csv")
loso = loso[loso.night == "SC4001"].sort_values("epoch")
raw = mne.io.read_raw_edf(next((ROOT / "data" / "sc").glob("SC4001*PSG.edf")), include=["EEG Fpz-Cz", "EOG horizontal"], preload=True)
fs = raw.info["sfreq"]; eeg = raw.get_data(picks=0)[0] * 1e6; eog = raw.get_data(picks=1)[0] * 1e6
det = LiveDetector(b, step_s=30.0); n = int(30 * fs)
diffs = []; first_bad = None
for _, r in loso.iterrows():
    a = int(r.t * fs)
    p = det.push(eeg[a:a + n], eog[a:a + n], fs)
    if p is not None:
        d_ = max(abs(p[s] - r[f"p_{s}"]) for s in ["W", "N1", "N2", "N3", "REM"]); diffs.append(d_)
        if d_ > 1e-6 and first_bad is None: first_bad = (int(r.epoch), d_)
print("first_bad", first_bad, "n_bad", sum(d > 1e-6 for d in diffs), "median", np.median(diffs)); print("epochs compared", len(diffs), "max |dp|", max(diffs))
assert max(diffs) < 0.05 and sum(d > 1e-6 for d in diffs) <= 0.01 * len(diffs), "live detector diverges from validation"
print("PASS")
