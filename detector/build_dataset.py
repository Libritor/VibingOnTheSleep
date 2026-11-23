"""Turn the downloaded Sleep-EDF nights into a per-epoch feature table (data/epochs.csv)."""
import re
from pathlib import Path
import numpy as np, pandas as pd, mne
from features import epoch_feats

mne.set_log_level("ERROR")
ROOT = Path(__file__).resolve().parent.parent / "data"
MAP = {"Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
       "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM"}
EP = 30.0

rows = []
psgs = sorted(ROOT.rglob("SC4*PSG.edf"))
for psg in psgs:
    m = re.match(r"SC4(\d\d)(\d)", psg.name)
    subj = int(m.group(1))
    hyp = next(psg.parent.glob(psg.name[:6] + "*Hypnogram.edf"))
    raw = mne.io.read_raw_edf(psg, include=["EEG Fpz-Cz", "EOG horizontal"], preload=True)
    fs = raw.info["sfreq"]
    ann = mne.read_annotations(hyp)
    labels = []
    for on, dur, desc in zip(ann.onset, ann.duration, ann.description):
        if desc in MAP:
            for k in range(int(dur // EP)):
                labels.append((on + k * EP, MAP[desc]))
    st = [s for _, s in labels]
    sleep_idx = [i for i, s in enumerate(st) if s != "W"]
    if not sleep_idx:
        continue
    lo = max(0, sleep_idx[0] - 60)        # keep 30 min of wake before first sleep
    hi = min(len(st), sleep_idx[-1] + 61)  # and 30 min after the last
    eeg = raw.get_data(picks="EEG Fpz-Cz")[0] * 1e6
    eog = raw.get_data(picks="EOG horizontal")[0] * 1e6
    n = int(EP * fs)
    for i in range(lo, hi):
        t0, stage = labels[i]
        a = int(t0 * fs)
        if a + n > len(eeg):
            break
        f = epoch_feats(eeg[a:a + n], eog[a:a + n], fs)
        f.update(subject=subj, night=psg.name[:6], epoch=i - lo, t=t0, stage=stage)
        rows.append(f)
    print(psg.name, "epochs", hi - lo, flush=True)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "epochs.csv", index=False)
print(df.groupby("stage").size(), "\nsubjects", df.subject.nunique())
