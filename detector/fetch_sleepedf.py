"""Download Sleep-EDF Expanded (SC, night 1) for subjects 0..19 into ./data via MNE."""
import sys
from pathlib import Path
from mne.datasets.sleep_physionet.age import fetch_data

out = Path(__file__).resolve().parent.parent / "data"
out.mkdir(exist_ok=True)
subjects = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 20))
for s in subjects:
    try:
        paths = fetch_data(subjects=[s], recording=[1], path=str(out), on_missing="warn")
        print("ok", s, paths, flush=True)
    except Exception as e:
        print("fail", s, e, flush=True)
