"""Cue versus sham, for one participant's naps.

The schedule puts one cue and one sham in each consecutive pair of naps, so the test respects the
design: within each pair, the difference cue minus sham; under the null, each pair's sign is a coin
flip. The exact p-value enumerates all 2^pairs sign patterns (5 pairs: 32 patterns, smallest
possible one-sided p = 1/32 = 0.031). The hypothesis is directional (cue > sham) and is stated
before the data, so the one-sided p is primary; the two-sided p is reported too.

Outcomes, in order of priority:
  1. incorporation (0-3), from human ratings if present, else the language model's, else a
     lexical score (does any report name the object), which needs no rater
  2. number of new post-nap uses
  3. mean originality of the new uses (rated)

    python study/analyze.py --participant P-0001
"""
import argparse
import csv
import itertools
import json
import re
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(STUDY))
from score import collect  # noqa: E402


def read_csv(p):
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8") as f:
        return {r["item"]: r for r in csv.DictReader(f)}


def lexical(obj, reports):
    words = obj.lower().split()
    pat = re.compile(r"\b(" + "|".join(re.escape(w) + r"s?" for w in words) + r")\b")
    return float(any(pat.search(r.lower()) for r in reports))


def paired_test(pairs):
    d = np.array([c - s for c, s in pairs], float)
    obs = d.mean()
    flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
    null = (flips * d).mean(axis=1)
    return {"pairs": len(d), "mean_diff_cue_minus_sham": float(obs),
            "p_one_sided": float((null >= obs - 1e-12).mean()),
            "p_two_sided": float((np.abs(null) >= abs(obs) - 1e-12).mean()),
            "pairs_favouring_cue": int((d > 0).sum()), "ties": int((d == 0).sum())}


def detector_summary(session_dir):
    m = json.loads((session_dir / "manifest.json").read_text())
    rows = list(csv.DictReader(open(session_dir / "detector.csv", newline="")))
    watched = [r for r in rows if r["state"] != "CALIB"]
    ok = np.mean([int(r["contact_ok"]) for r in watched]) if watched else float("nan")
    return {"triggers": len(m["triggers"]),
            "eeg_triggers": sum("eeg" in t["by"] for t in m["triggers"]),
            "nod_triggers": sum("nod" in t["by"] for t in m["triggers"]),
            "overshoots": m["overshoots"], "contact_ok_fraction": float(ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    a = ap.parse_args()
    pdir = STUDY / "participants" / a.participant
    sdir = pdir / "scoring"
    key = read_csv(sdir / "key.csv") or {}
    human = read_csv(sdir / "human_ratings.csv")
    llm_r = read_csv(sdir / "llm_ratings.csv")
    by_session = {r["session"]: item for item, r in key.items()}
    source = "human" if human else "llm" if llm_r else "lexical"
    ratings = human or llm_r or {}

    table = []
    for it in collect(a.participant):
        item = by_session.get(it["session"])
        r = ratings.get(item, {}) if item else {}
        if source == "human":
            inc = r.get("incorporation_0_3", "")
            orig = r.get("originality_1_5_per_new_use", "")
        else:
            inc, orig = r.get("incorporation", ""), r.get("originality", "")
        inc = float(inc) if str(inc).strip() != "" else lexical(it["object"], it["reports"]) \
            if source == "lexical" else float("nan")
        o = [float(x) for x in re.split(r"[;, ]+", str(orig)) if x.strip()]
        row = {"nap_index": it["nap_index"], "condition": it["condition"], "object": it["object"],
               "incorporation": inc, "new_uses": len(it["new_uses"]),
               "originality_mean": float(np.mean(o)) if o else float("nan"),
               **detector_summary(pdir / "sessions" / it["session"])}
        table.append(row)

    table.sort(key=lambda r: r["nap_index"])
    by_idx = {r["nap_index"]: r for r in table}
    sched = json.loads((pdir / "schedule.json").read_text())["naps"]
    results = {"participant": a.participant, "rating_source": source, "naps_done": len(table)}
    for outcome in ("incorporation", "new_uses", "originality_mean"):
        pairs = []
        for k in range(0, len(sched) - 1, 2):
            p = [by_idx.get(sched[k]["index"]), by_idx.get(sched[k + 1]["index"])]
            if None in p:
                continue
            c = next(x for x in p if x["condition"] == "cue")[outcome]
            s = next(x for x in p if x["condition"] == "sham")[outcome]
            if not (np.isnan(c) or np.isnan(s)):
                pairs.append((c, s))
        results[outcome] = paired_test(pairs) if pairs else {"pairs": 0}
    for cond in ("cue", "sham"):
        rows = [r for r in table if r["condition"] == cond]
        results[f"{cond}_means"] = {k: float(np.nanmean([r[k] for r in rows])) if rows else None
                                    for k in ("incorporation", "new_uses", "originality_mean",
                                              "triggers", "overshoots", "contact_ok_fraction")}
    sdir.mkdir(exist_ok=True)
    with open(sdir / "per_nap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    (sdir / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
