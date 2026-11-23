"""Make a participant's nap schedule: cue and sham in balanced, shuffled pairs, one object per nap.

Each consecutive pair of naps holds one cue and one sham in random order, so the two conditions
are balanced over time (practice, season, sleep debt). Objects are shuffled and never repeated.
The schedule is written once and never rewritten; the seed is stored so it can be re-derived.

    python study/make_schedule.py --participant P-0001 --naps 10
"""
import argparse
import json
import secrets
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent


def make(n_naps, seed, objects):
    if n_naps % 2:
        raise ValueError("use an even number of naps so cue and sham balance")
    if n_naps > len(objects):
        raise ValueError(f"only {len(objects)} objects in cues.json")
    rng = np.random.default_rng(seed)
    conds = []
    for _ in range(n_naps // 2):
        pair = ["cue", "sham"]
        rng.shuffle(pair)
        conds += pair
    objs = list(rng.permutation(objects)[:n_naps])
    return [{"index": i + 1, "condition": c, "object": str(o)} for i, (c, o) in enumerate(zip(conds, objs))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--naps", type=int, default=10)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    pdir = STUDY / "participants" / a.participant
    pdir.mkdir(parents=True, exist_ok=True)
    out = pdir / "schedule.json"
    if out.exists():
        raise SystemExit(f"{out} exists; schedules are never rewritten.")
    seed = a.seed if a.seed is not None else secrets.randbits(32)
    objects = json.loads((STUDY / "cues.json").read_text())["objects"]
    out.write_text(json.dumps({"participant": a.participant, "seed": seed,
                               "naps": make(a.naps, seed, objects)}, indent=1))
    print(f"Wrote {out}. Conditions are inside; the person napping should not open it.")


if __name__ == "__main__":
    main()
