"""What the night's raw EEG shows on its own, without the classifier: per 30-s epoch, the
amplitude of each electrode (sd, uV), whether the contact rule passes, the relative delta, theta,
alpha and sigma power of the frontal signal, the slow-eye-movement power of the EOG, and the AUX
(Oz) alpha and delta when a fifth electrode was recorded. A quiet stretch that is real sleep keeps
EEG-sized amplitudes (10-60 uV) with delta and sigma rising; an electrode that has come off is
flat (sd under 0.5), railed, or hundreds of uV of noise. Prints a 15-minute timeline and writes
physio.csv beside the night's other files.

    python rig/night_physio.py study/participants/P-0001/nights/<stamp>
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "detector"))
sys.path.insert(0, str(ROOT / "rig"))
from features import band_feats, eog_feats  # noqa: E402
from sources import contact_ok, derive  # noqa: E402

FS = 256


def main(night):
    night = Path(night)
    t, X, aux = [], [], []
    with open(night / "raw.csv", newline="") as f:
        r = csv.reader(f)
        head = next(r)
        ia = head.index("AUX") if "AUX" in head else None
        for row in r:
            try:
                X.append([float(v) for v in row[1:5]])
                t.append(float(row[0]))
                a = row[ia] if ia is not None else ""
                aux.append(float(a) if a not in ("", "nan") else np.nan)
            except ValueError:
                continue
    t, X, aux = np.array(t), np.array(X), np.array(aux)
    t -= t[0]
    rows = []
    for e0 in np.arange(0, t[-1] - 30, 30.0):
        i, j = np.searchsorted(t, e0), np.searchsorted(t, e0 + 30)
        x = X[i:j]
        if len(x) < 20 * FS:
            continue
        sd = x.std(0)
        ok = contact_ok(x, FS)
        eeg, eog = derive(x)
        b = band_feats(eeg - eeg.mean(), FS)
        o = eog_feats(eog - eog.mean(), FS)
        row = dict(t_min=round(e0 / 60, 1), contact_ok=int(ok),
                   sd_TP9=round(sd[0], 1), sd_AF7=round(sd[1], 1), sd_AF8=round(sd[2], 1), sd_TP10=round(sd[3], 1),
                   rel_delta=round(b["rel_delta"], 3), rel_theta=round(b["rel_theta"], 3),
                   rel_alpha=round(b["rel_alpha"], 3), rel_sigma=round(b["rel_sigma"], 3),
                   log_sem=round(o["log_sem"], 2), eog_sd=round(float(eog.std()), 1))
        a = aux[i:j]
        if np.isfinite(a).all() and a.std() > 0:
            ab = band_feats(a - a.mean(), FS)
            row.update(sd_AUX=round(float(a.std()), 1), aux_rel_alpha=round(ab["rel_alpha"], 3),
                       aux_rel_delta=round(ab["rel_delta"], 3))
        rows.append(row)
    keys = list(rows[0].keys()) | {k for r in rows for k in r} if False else sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1].keys()).index(k) if k in rows[-1] else 99)
    with open(night / "physio.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{night.name}: {len(rows)} epochs of 30 s ({len(rows) / 2:.0f} min)")
    print(f"{'from':>6} {'contact':>7} {'sd TP9':>7} {'AF7':>6} {'AF8':>6} {'TP10':>6} {'AUX':>7} "
          f"{'delta':>6} {'theta':>6} {'alpha':>6} {'sigma':>6} {'SEM':>6}")
    for k in range(0, len(rows), 30):
        g = rows[k:k + 30]
        med = lambda key: np.nanmedian([r.get(key, np.nan) for r in g])
        print(f"{g[0]['t_min']:6.0f} {np.mean([r['contact_ok'] for r in g]):7.2f} {med('sd_TP9'):7.1f} "
              f"{med('sd_AF7'):6.1f} {med('sd_AF8'):6.1f} {med('sd_TP10'):6.1f} {med('sd_AUX'):7.1f} "
              f"{med('rel_delta'):6.2f} {med('rel_theta'):6.2f} {med('rel_alpha'):6.2f} {med('rel_sigma'):6.2f} "
              f"{med('log_sem'):6.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
