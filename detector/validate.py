"""Leave-one-subject-out validation of the Muse-montage sleep detector on Sleep-EDF.

Honesty constraints, so the numbers transfer to live use:
  * only Fpz-Cz EEG + horizontal EOG (what a Muse approximates), no EMG, no Pz-Oz
  * each night normalised by its own first 10 min (pre-sleep wake), the rig's calibration
  * context features use PAST epochs only (the live detector cannot see the future)
  * the subject being tested is never in the training set
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "epochs.csv")
STAGES = ["W", "N1", "N2", "N3", "REM"]
META = {"subject", "night", "epoch", "t", "stage"}
base = [c for c in df.columns if c not in META]


BASE_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 6   # 6 epochs = 3 min calibration


def prepare(d):
    d = d.sort_values("epoch").copy()
    ref = d.iloc[:BASE_EPOCHS][base]              # eyes-closed wake calibration
    d[base] = (d[base] - ref.median()) / (ref.std() + 1e-6)
    out = [d[base]]
    for lag in (1, 2, 3):
        out.append(d[base].shift(lag).bfill().add_suffix(f"_l{lag}"))
    out.append(d[base].rolling(6, min_periods=1).mean().add_suffix("_m6"))
    X = pd.concat(out, axis=1)
    return X, d


parts = [prepare(g) for _, g in df.groupby("night")]
X = pd.concat([p[0] for p in parts]); D = pd.concat([p[1] for p in parts])
y = D.stage.map({s: i for i, s in enumerate(STAGES)}).values
subj = D.subject.values
P = np.zeros((len(y), 5))
for s in np.unique(subj):
    tr, te = subj != s, subj == s
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                         random_state=0)
    clf.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
    P[te] = clf.predict_proba(X[te])
pred = P.argmax(1)
D = D.assign(**{f"p_{s}": P[:, i] for i, s in enumerate(STAGES)}, pred=[STAGES[i] for i in pred])
D.to_csv(ROOT / "data" / f"loso_predictions_base{BASE_EPOCHS}.csv", index=False)

res = {"calibration_epochs": BASE_EPOCHS, "n_subjects": int(len(np.unique(subj))), "n_epochs": int(len(y)),
       "kappa": float(cohen_kappa_score(y, pred)),
       "accuracy": float((y == pred).mean()),
       "f1": dict(zip(STAGES, map(float, f1_score(y, pred, average=None)))),
       "kappa_per_subject": [float(cohen_kappa_score(y[subj == s], pred[subj == s]))
                             for s in np.unique(subj)],
       "confusion": confusion_matrix(y, pred).tolist()}

# ---- REM cue detector: cue only when P(REM) >= threshold ------------------------------
rem = []
for th in (0.5, 0.6, 0.7, 0.8, 0.9):
    fire = P[:, 4] >= th
    tp = (fire & (y == 4)).sum()
    by = {s: int((fire & (y == i)).sum()) for i, s in enumerate(STAGES)}
    rem.append({"threshold": th, "fires": int(fire.sum()),
                "ppv": float(tp / max(fire.sum(), 1)),
                "sensitivity": float(tp / (y == 4).sum()),
                "fires_by_true_stage": by,
                "fires_in_N3_or_W_pct": float((by["N3"] + by["W"]) / max(fire.sum(), 1))})
res["rem_detector"] = rem

# ---- Nap detector: sleep onset alarm and N2 stop, per night, run causally ------------
def first(mask, k=1):
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= k:
            return i
    return None

nap = []
for th in (0.5, 0.6, 0.7):
    for k in (1, 2):
        lat, inwin, n2lat = [], 0, []
        nights = 0
        for _, g in D.groupby("night"):
            g = g.sort_values("epoch")
            st = g.stage.values
            onset = first(st != "W")                  # scored sleep onset
            n2 = first(np.isin(st, ["N2", "N3"]))    # scored N2 = stop condition
            psleep = 1 - g.p_W.values
            alarm = first(psleep >= th, k)
            pn2 = g.p_N2.values + g.p_N3.values
            stop = first(pn2 >= th, k)
            nights += 1
            if alarm is not None and onset is not None:
                lat.append((alarm - onset) * 0.5)
                if alarm < n2:
                    inwin += 1
            if stop is not None and n2 is not None:
                n2lat.append((stop - n2) * 0.5)
        nap.append({"p_sleep_threshold": th, "consecutive_epochs": k, "nights": nights,
                    "onset_alarm_minus_scored_onset_min_median": float(np.median(lat)),
                    "onset_alarm_minus_scored_onset_min_IQR": [float(np.percentile(lat, 25)), float(np.percentile(lat, 75))],
                    "alarm_before_scored_N2_pct": float(inwin / nights),
                    "n2_stop_minus_scored_N2_min_median": float(np.median(n2lat)),
                    "n2_stop_minus_scored_N2_min_IQR": [float(np.percentile(n2lat, 25)), float(np.percentile(n2lat, 75))]})
res["nap_detector"] = nap

# ---- Sleep-onset latency from N1 entry to N2, scored (how wide is the window?) --------
win = []
for _, g in D.groupby("night"):
    st = g.sort_values("epoch").stage.values
    win.append((first(np.isin(st, ["N2", "N3"])) - first(st != "W")) * 0.5)
res["scored_N1_window_min"] = {"median": float(np.median(win)), "min": float(min(win)), "max": float(max(win))}

(ROOT / "detector" / f"results_base{BASE_EPOCHS}.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("kappa", "accuracy", "f1", "scored_N1_window_min")}, indent=1))
for r in rem: print("REM", r["threshold"], "ppv %.2f sens %.2f N3/W %.2f" % (r["ppv"], r["sensitivity"], r["fires_in_N3_or_W_pct"]))
for r in nap: print("NAP", r["p_sleep_threshold"], r["consecutive_epochs"], "onset err %.1f" % r["onset_alarm_minus_scored_onset_min_median"], r["onset_alarm_minus_scored_onset_min_IQR"], "before N2 %.2f" % r["alarm_before_scored_N2_pct"], "N2 stop err %.1f" % r["n2_stop_minus_scored_N2_min_median"])
