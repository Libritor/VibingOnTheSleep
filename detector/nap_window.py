"""Where exactly does the sleep-onset alarm land, per night? (uses loso_predictions.csv)"""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
D = pd.read_csv(ROOT / "data" / "loso_predictions_base6.csv")

def first(mask, k):
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= k: return i
    return None

out = []
for th in (0.5, 0.6, 0.7, 0.8):
    for k in (1, 2, 3):
        cats = {"inside_N1_window": 0, "early_within_2min": 0, "early_over_2min": 0, "late_in_N2_or_after": 0, "never": 0}
        stage_at_alarm = []
        for _, g in D.groupby("night"):
            g = g.sort_values("epoch"); st = g.stage.values
            onset = first(st != "W", 1); n2 = first(np.isin(st, ["N2", "N3"]), 1)
            a = first((1 - g.p_W.values) >= th, k)
            if a is None: cats["never"] += 1; continue
            stage_at_alarm.append(st[a])
            if onset <= a < n2: cats["inside_N1_window"] += 1
            elif a < onset and onset - a <= 4: cats["early_within_2min"] += 1
            elif a < onset: cats["early_over_2min"] += 1
            else: cats["late_in_N2_or_after"] += 1
        out.append({"th": th, "k": k, **cats,
                    "scored_stage_at_alarm": pd.Series(stage_at_alarm).value_counts().to_dict()})
        print(th, k, cats)
(ROOT / "detector" / "nap_window_base6.json").write_text(json.dumps(out, indent=1))
base = (D.stage == "REM").mean(); print("REM base rate %.3f" % base)
