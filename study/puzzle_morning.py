"""Morning half of the puzzle-cueing study: three questions about the night, then every puzzle left
unsolved last evening again, in a new order and without sounds. Only after the last one does it
check the split against the evening's commitment hash and show which puzzles were cued.

    python study/puzzle_morning.py --participant P-0001 --self       # the latest session without a morning
    python study/puzzle_morning.py --participant P-0001 --self --session study/participants/P-0001/puzzle_sessions/<stamp>

Writes morning.csv, morning_attempts.csv and result.json into the session folder.
"""
import argparse
import csv
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
import puzzle_bank as pb  # noqa: E402
from puzzle_common import commitment, participant_dir, sessions, timed_answer  # noqa: E402


def ask(q, allowed):
    while True:
        r = input(f"{q} [{'/'.join(allowed)}] ").strip().lower()
        if r in allowed:
            return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--self", action="store_true")
    ap.add_argument("--session")
    ap.add_argument("--simulate", nargs=2, type=float, metavar=("P_CUED", "P_UNCUED"),
                    help="test only: no typing; cued and uncued puzzles solved with these probabilities")
    a = ap.parse_args(argv)
    pdir = participant_dir(a.participant, a.self or a.simulate is not None)
    if a.session:
        sdir = Path(a.session)
    else:
        todo = [s for s in sessions(pdir) if (s / "split.json").exists() and not (s / "morning.csv").exists()]
        if not todo:
            sys.exit("No evening session is waiting for its morning.")
        sdir = todo[-1]
    man = json.loads((sdir / "manifest.json").read_text())
    if (sdir / "morning.csv").exists():
        sys.exit(f"{sdir.name} already has a morning.")
    split = json.loads((sdir / "split.json").read_text())
    if commitment(split["cued"], split["uncued"], split["salt"]) != man["split_commitment"]:
        sys.exit("split.json does not match the evening's commitment hash: it was changed after the evening. "
                 "This night cannot be scored.")
    bank = {p["id"]: p for p in pb.load()}
    todo = split["cued"] + split["uncued"]
    rng = random.Random(man["seed"] + 2)
    rng.shuffle(todo)
    sim = random.Random(man["seed"] + 3) if a.simulate else None
    cued = set(split["cued"])

    if sim:
        answers = dict(heard_sounds="n", sleep_quality="4", thought_about_puzzles="n")
    else:
        print(f"\nGOOD MORNING. Session {sdir.name}: {len(todo)} puzzles from last evening.\n")
        answers = dict(
            heard_sounds=ask("Did you notice any sounds during the night?", ["y", "n", "unsure"]),
            sleep_quality=ask("How well did you sleep? 1 badly ... 5 very well", list("12345")),
            thought_about_puzzles=ask("Since waking, have you thought about any of last night's puzzles?", ["y", "n"]))
        print(f"\nNow the puzzles again, up to {man['seconds']:.0f} s each, no sounds. 'skip' gives one up.")
        input("Press Enter to begin... ")
    rows, att = [], []
    for k, pid in enumerate(todo, 1):
        p = bank[pid]
        if sim:
            solved = sim.random() < (a.simulate[0] if pid in cued else a.simulate[1])
            took, tries = (round(sim.uniform(10, man["seconds"]), 1) if solved else man["seconds"]), []
        else:
            print(f"\n--- Puzzle {k} of {len(todo)} ---\n{pb.prompt(p)}\n({pb.HOW_TO_ANSWER[p['family']]})")
            solved, took, tries = timed_answer(man["seconds"], lambda s, p=p: pb.check(p, s))
        rows.append(dict(order=k, id=pid, family=p["family"], solved=int(solved), seconds=took, attempts=len(tries)))
        att += [dict(id=pid, t=t, text=x, ok=int(ok), message=m) for t, x, ok, m in tries]

    with open(sdir / "morning.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["order", "id", "family", "solved", "seconds", "attempts"])
        w.writeheader()
        w.writerows(rows)
    with open(sdir / "morning_attempts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "t", "text", "ok", "message"])
        w.writeheader()
        w.writerows(att)
    nc = json.loads((sdir / "night_cues.json").read_text()) if (sdir / "night_cues.json").exists() else {}
    by = {g: [r for r in rows if (r["id"] in cued) == (g == "cued")] for g in ("cued", "uncued")}
    res = dict(session=sdir.name, finished=dt.datetime.now().isoformat(timespec="seconds"), **answers,
               cued_solved=sum(r["solved"] for r in by["cued"]), cued_n=len(by["cued"]),
               uncued_solved=sum(r["solved"] for r in by["uncued"]), uncued_n=len(by["uncued"]),
               cues_total=nc.get("cues_total"), minutes_cueing=nc.get("minutes_cueing"),
               night_recorded=bool(nc), simulated=bool(sim))
    (sdir / "result.json").write_text(json.dumps(res, indent=1))
    pc = res["cued_solved"] / res["cued_n"] if res["cued_n"] else float("nan")
    pu = res["uncued_solved"] / res["uncued_n"] if res["uncued_n"] else float("nan")
    print(f"\nThe split matched the evening's commitment hash ({man['split_commitment'][:16]}...).")
    print(f"Solved this morning: cued {res['cued_solved']}/{res['cued_n']} ({pc:.0%}), "
          f"uncued {res['uncued_solved']}/{res['uncued_n']} ({pu:.0%}).")
    if nc:
        print(f"Last night: {nc['cues_total']} cues over {nc['minutes_cueing']} min of cueing.")
        if not nc["cues_total"]:
            print("No cue was played (no sustained slow-wave sleep was seen), so this night tests nothing; "
                  "it still counts in the intention-to-treat analysis.")
    else:
        print("No night_cues.json: the night rig was not run with --cues for this session.")
    print("One night is one small sample: the answer comes from study/puzzle_analysis.py over many nights.")
    return sdir


if __name__ == "__main__":
    main()
