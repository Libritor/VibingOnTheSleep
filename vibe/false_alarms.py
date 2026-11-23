"""How often does sleep itself make a pattern the decoder would read as YES or NO?

Nobody was asked anything on these recordings, so every verdict is a false alarm. Two sources:

  edf    the 20 Sleep-EDF nights (scored stages; Fpz-Cz and a horizontal eye channel at 100 Hz,
         laid out as the replay source does). Eye steps there come from electrodes beside the
         eyes, larger than the Muse's forehead pair sees, so the YES rate is an upper bound on
         the pattern, not a Muse number. Muscle is read from Fpz-Cz above 25 Hz only.
  muse   this user's own nights on the headband (no scored stages; clock hours instead).

Rates are per 10-s window, the length of a response window, with the uncalibrated default
thresholds (the evening calibration sets personal ones).

    python vibe/false_alarms.py edf            # -> vibe/false_alarms_sleepedf.json
    python vibe/false_alarms.py muse <raw.npz> [more.npz]
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vibe"))
from signals import Decoder  # noqa: E402

STAGE = {"Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2", "Sleep stage 3": "N3",
         "Sleep stage 4": "N3", "Sleep stage R": "REM"}
WIN = 10.0


def edf(hours=8.0):
    import mne
    mne.set_log_level("ERROR")
    dec = Decoder()
    tot = collections.defaultdict(collections.Counter)
    nights = 0
    for psg in sorted((ROOT / "data" / "sc").glob("*-PSG.edf")):
        hyp = next(iter(sorted(psg.parent.glob(psg.name[:6] + "*-Hypnogram.edf"))), None)
        if hyp is None:
            continue
        ann = mne.read_annotations(hyp)
        stages = []
        for on, du, de in zip(ann.onset, ann.duration, ann.description):
            stages += [STAGE.get(de, "?")] * int(du // 30)
        first = next((k for k, s in enumerate(stages) if s in ("N1", "N2", "N3", "REM")), None)
        if first is None:
            continue
        raw = mne.io.read_raw_edf(psg, include=["EEG Fpz-Cz", "EOG horizontal"], preload=True)
        fs = raw.info["sfreq"]
        e, o = raw.get_data(picks=0)[0] * 1e6, raw.get_data(picks=1)[0] * 1e6
        t0 = max(0.0, first * 30 - 1800)
        n = int(WIN * fs)
        for k in range(int(hours * 3600 / WIN)):
            a = int((t0 + k * WIN) * fs)
            if a + n > len(e):
                break
            st = stages[min(len(stages) - 1, int((t0 + k * WIN) // 30))]
            rows = np.stack([np.full(n, 800.0), 800 + e[a:a + n] + o[a:a + n] / 2,
                             800 + e[a:a + n] - o[a:a + n] / 2, np.full(n, 800.0)], axis=1)
            tot[st][str(dec.decode(rows, fs)["answer"])] += 1
        nights += 1
        print(psg.name[:6], {s: dict(c) for s, c in sorted(tot.items()) if s != "?"}, flush=True)
    out = dict(nights=nights, hours_per_night=hours, window_s=WIN, thresholds=dec.p, by_stage={})
    for st in ("W", "N1", "N2", "N3", "REM"):
        c = tot[st]
        n = sum(c.values())
        out["by_stage"][st] = dict(windows=n, yes=c["yes"], no=c["no"], ambiguous=c["ambiguous"],
                                   yes_pct=round(100 * c["yes"] / max(n, 1), 2),
                                   no_pct=round(100 * c["no"] / max(n, 1), 2))
    (ROOT / "vibe" / "false_alarms_sleepedf.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["by_stage"], indent=1))
    return out


def muse(paths):
    dec = Decoder()
    for path in paths:
        z = np.load(path, allow_pickle=True)
        eeg = z["eeg"][:, :4].astype(float)
        t = z["t"]
        fs = float(z["fs_est"]) if "fs_est" in z.files else 256.0
        fs = 256.0 if abs(fs - 256) < 12 else fs
        acc = z["acc"] if "acc" in z.files and len(z["acc"]) else None
        n = int(WIN * fs)
        per_hour = collections.defaultdict(collections.Counter)
        for a in range(0, len(eeg) - n, n):
            rows = eeg[a:a + n]
            ac = None
            if acc is not None:
                m = (acc[:, 0] >= t[a]) & (acc[:, 0] < t[a] + WIN)
                ac = acc[m, 1:4] if m.sum() > 30 else None
            r = dec.decode(rows, fs, ac)
            h = int(t[a] // 3600)
            key = "unusable" if r["n_forehead"] < 2 else "moved" if (r["why"].startswith("head moved")) else str(r["answer"])
            per_hour[h][key] += 1
        print(f"\n{path}  (fs {fs:.0f} Hz, clock0 {z['clock0'] if 'clock0' in z.files else '?'})")
        tot = collections.Counter()
        for h in sorted(per_hour):
            c = per_hour[h]
            tot.update(c)
            print(f"  hour {h}: quiet {c['None']:4d}  yes {c['yes']:3d}  no {c['no']:3d}  both {c['ambiguous']:3d}  "
                  f"moved {c['moved']:3d}  forehead off {c['unusable']:4d}")
        good = tot["None"] + tot["yes"] + tot["no"] + tot["ambiguous"]
        print(f"  all: {good} usable still windows ({good * WIN / 3600:.1f} h): yes {tot['yes']} "
              f"({100 * tot['yes'] / max(good, 1):.2f}%), no {tot['no']} ({100 * tot['no'] / max(good, 1):.2f}%), "
              f"both {tot['ambiguous']}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "edf":
        edf()
    elif len(sys.argv) >= 3 and sys.argv[1] == "muse":
        muse(sys.argv[2:])
    else:
        print(__doc__)
