"""Training and the live (incremental) detector. Shares features.py with the validation,
so the live detector computes exactly what was validated."""
from pathlib import Path
import collections
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from features import epoch_feats

ROOT = Path(__file__).resolve().parent.parent
STAGES = ["W", "N1", "N2", "N3", "REM"]
META = {"subject", "night", "epoch", "t", "stage"}


def context(F):
    """F: DataFrame of normalised per-epoch features in time order -> model inputs."""
    out = [F]
    for lag in (1, 2, 3):
        out.append(F.shift(lag).bfill().add_suffix(f"_l{lag}"))
    out.append(F.rolling(6, min_periods=1).mean().add_suffix("_m6"))
    return pd.concat(out, axis=1)


def train(base_epochs, exclude_subjects=()):
    df = pd.read_csv(ROOT / "data" / "epochs.csv")
    df = df[~df.subject.isin(exclude_subjects)]
    base = [c for c in df.columns if c not in META]
    Xs, ys = [], []
    for _, d in df.groupby("night"):
        d = d.sort_values("epoch")
        ref = d.iloc[:base_epochs][base]
        F = (d[base] - ref.median()) / (ref.std() + 1e-6)
        Xs.append(context(F)); ys.append(d.stage.map({s: i for i, s in enumerate(STAGES)}).values)
    X, y = pd.concat(Xs), np.concatenate(ys)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, random_state=0)
    clf.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return {"clf": clf, "base": base, "columns": list(X.columns), "base_epochs": base_epochs,
            "excluded": list(exclude_subjects)}


class LiveDetector:
    """Feed it 30-s windows of (eeg, eog) every `step_s` seconds; it returns stage probabilities.
    The first `base_epochs` epochs-worth of windows (eyes-closed wake) set the calibration."""

    def __init__(self, bundle, step_s=5.0):
        self.b = bundle
        self.k = int(round(30.0 / step_s))          # windows per 30-s epoch
        self.calib = []
        self.ref_med = self.ref_std = None
        self.hist = collections.deque(maxlen=6 * self.k + 1)

    @property
    def calibrated(self):
        return self.ref_med is not None

    def calibration_progress(self):
        need = self.b["base_epochs"] * self.k
        return min(1.0, len(self.calib) / need)

    def push(self, eeg, eog, fs):
        f = pd.Series(epoch_feats(eeg, eog, fs))[self.b["base"]]
        if not self.calibrated:
            self.calib.append(f)
            if len(self.calib) >= self.b["base_epochs"] * self.k:
                # use one window per epoch, like the validation (non-overlapping epochs)
                C = pd.DataFrame(self.calib[::self.k])
                self.ref_med, self.ref_std = C.median(), C.std() + 1e-6
                for c in self.calib:                      # calibration windows are context too
                    self.hist.append((c - self.ref_med) / self.ref_std)
            return None
        z = (f - self.ref_med) / self.ref_std
        self.hist.append(z)
        h = list(self.hist)
        cur = h[-1]
        lag = lambda n: h[-1 - n * self.k] if len(h) > n * self.k else h[(len(h) - 1) % self.k]
        row = [cur] + [lag(n).add_suffix(f"_l{n}") for n in (1, 2, 3)]
        avail = [h[-1 - n * self.k] for n in range(6) if len(h) > n * self.k]
        m6 = pd.concat(avail, axis=1).mean(axis=1).add_suffix("_m6")
        x = pd.concat(row + [m6])[self.b["columns"]].to_frame().T.astype(float)
        p = self.b["clf"].predict_proba(x)[0]
        return dict(zip(STAGES, map(float, p)))


if __name__ == "__main__":
    out = ROOT / "models"; out.mkdir(exist_ok=True)
    joblib.dump(train(6), out / "nap_base3min.joblib")
    joblib.dump(train(20), out / "night_base10min.joblib")
    joblib.dump(train(6, exclude_subjects=(0,)), out / "nap_base3min_minus_subject0.joblib")
    print("saved", [p.name for p in out.iterdir()])
