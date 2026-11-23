"""Evening half of the puzzle-cueing study: try tonight's puzzles, each with its own sound, then
split the unsolved ones (hidden) into the half that will be cued in slow-wave sleep and the half
that will not.

    python study/puzzle_evening.py --participant P-0001 --self
    # then, at bedtime, with MuseLog streaming:
    python rig/night_watch.py --participant P-0001 --cues study/participants/P-0001/puzzle_sessions/<stamp>
    # and in the morning:
    python study/puzzle_morning.py --participant P-0001 --self

Tonight's set: --n puzzles (default 20), as even a mix of arithmetic, resistor networks and
anagrams as the number allows, none used on an earlier night. Each puzzle's sound plays when it
appears and every --sound-every seconds while it is shown. Up to --seconds per puzzle; type 'skip'
to give one up.

Written to study/participants/<P>/puzzle_sessions/<stamp>/: manifest.json (puzzles, sound
specifications, the split's commitment hash), evening.csv (one row per puzzle), attempts.csv
(every typed answer), split.json (which unsolved puzzles are cued: do not open it before the
morning; the commitment hash in the manifest shows it was not changed afterwards).
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
import puzzle_sounds as ps  # noqa: E402
from puzzle_common import SoundLoop, make_split, participant_dir, sessions, timed_answer  # noqa: E402


def used_ids(pdir):
    used = set()
    for s in sessions(pdir):
        used |= set(json.loads((s / "manifest.json").read_text()).get("puzzle_ids", []))
    return used


def choose(bank, used, n, k_session, rng):
    """n unused puzzles, families as even as possible; the family that gets one fewer rotates."""
    fams = list(pb.FAMILIES)
    base, extra = divmod(n, len(fams))
    quota = {f: base + (1 if (i - k_session) % len(fams) < extra else 0) for i, f in enumerate(fams)}
    out = []
    for f in fams:
        free = [p for p in bank if p["family"] == f and p["id"] not in used]
        if len(free) < quota[f]:
            sys.exit(f"The bank has only {len(free)} unused {f} puzzles left; extend study/puzzle_bank.py.")
        out += free[:quota[f]]
    rng.shuffle(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--self", action="store_true", help="investigator self-experiment (no REB code)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--sound-every", type=float, default=10.0)
    ap.add_argument("--gain", type=float, default=0.0, help="evening sound level, dB on the cue scale")
    ap.add_argument("--no-sound", action="store_true")
    ap.add_argument("--simulate", type=float, help="test only: no typing, each puzzle solved with this probability")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(argv)

    pdir = participant_dir(a.participant, a.self or a.simulate is not None)
    bank = pb.load()
    prior = sessions(pdir)
    seed = a.seed if a.seed is not None else int(time.time())
    rng = random.Random(seed)
    chosen = choose(bank, used_ids(pdir), a.n, len(prior), rng)
    specs = ps.specs_for_set(len(chosen), seed)
    sounds = {p["id"]: dict(spec) for p, spec in zip(chosen, specs)}

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sdir = pdir / "puzzle_sessions" / stamp
    k = 1
    while sdir.exists():                       # two sessions in one second (tests)
        k += 1
        sdir = pdir / "puzzle_sessions" / f"{stamp}_{k}"
    sdir.mkdir(parents=True)
    man = dict(participant=a.participant, session=stamp, started=dt.datetime.now().isoformat(timespec="seconds"),
               seed=seed, seconds=a.seconds, sound_every=a.sound_every, evening_gain_db=a.gain,
               simulated=a.simulate is not None, puzzle_ids=[p["id"] for p in chosen], sounds=sounds,
               bank_version=json.loads(pb.BANK.read_text())["version"])
    (sdir / "manifest.json").write_text(json.dumps(man, indent=1))

    sim = random.Random(seed + 1) if a.simulate is not None else None
    audio = not (a.no_sound or sim)
    if not sim:
        print(f"\nTONIGHT'S PUZZLES: {len(chosen)}, up to {a.seconds:.0f} s each. Each one has its own short "
              "sound, which plays when the puzzle appears and every few seconds while you work on it. "
              "Set the laptop volume so the sounds are comfortable.\n"
              "How to answer:\n  arithmetic: " + pb.HOW_TO_ANSWER["arith"] +
              "\n  resistors:  " + pb.HOW_TO_ANSWER["resistor"] +
              "\n  anagrams:   " + pb.HOW_TO_ANSWER["anagram"] +
              "\nA wrong answer just says why; keep trying until the time runs out. Type 'skip' to give up "
              "on a puzzle. Leaving many unsolved is expected: the morning retries them.")
        input("\nPress Enter to begin... ")

    rows, att = [], []
    for k, p in enumerate(chosen, 1):
        text = pb.prompt(p)
        if sim:
            solved = sim.random() < a.simulate
            took, tries = (round(sim.uniform(10, a.seconds), 1) if solved else a.seconds), []
        else:
            print(f"\n--- Puzzle {k} of {len(chosen)} ---\n{text}\n({pb.HOW_TO_ANSWER[p['family']]})")
            with SoundLoop(ps.make(sounds[p["id"]]), a.sound_every, a.gain, enabled=audio):
                solved, took, tries = timed_answer(a.seconds, lambda s, p=p: pb.check(p, s))
        rows.append(dict(order=k, id=p["id"], family=p["family"], solved=int(solved), seconds=took,
                         attempts=len(tries), sound=f"{sounds[p['id']]['timbre']}/{sounds[p['id']]['contour']}"))
        att += [dict(id=p["id"], t=t, text=x, ok=int(ok), message=m) for t, x, ok, m in tries]

    with open(sdir / "evening.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(sdir / "attempts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "t", "text", "ok", "message"])
        w.writeheader()
        w.writerows(att)
    unsolved = [(r["id"], r["family"]) for r in rows if not r["solved"]]
    split = make_split(unsolved, random.Random(secrets_seed()))
    (sdir / "split.json").write_text(json.dumps(split, indent=1))
    man.update(finished=dt.datetime.now().isoformat(timespec="seconds"), solved=sum(r["solved"] for r in rows),
               unsolved=len(unsolved), cued_n=len(split["cued"]), uncued_n=len(split["uncued"]),
               split_commitment=split["commitment"])
    (sdir / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"\nSolved {man['solved']} of {len(rows)}. {len(unsolved)} unsolved: {len(split['cued'])} will be cued "
          f"in deep sleep tonight, {len(split['uncued'])} will not (which is which stays hidden until the morning).")
    print(f"Commitment hash of the split: {split['commitment'][:16]}...")
    if len(unsolved) < 4:
        print("Fewer than four unsolved puzzles: tonight cannot test much. Next time use --seconds 60.")
    elif man["solved"] == 0 and not sim:
        print("None solved: if that keeps happening, use --seconds 120 so the morning is not hopeless too.")
    try:
        rel = sdir.relative_to(STUDY.parent)
    except ValueError:
        rel = sdir
    print(f"\nAt bedtime, with MuseLog streaming to this PC:\n  python rig/night_watch.py --participant "
          f"{a.participant} --cues {rel.as_posix()}")
    return sdir


def secrets_seed():
    import secrets
    return secrets.randbits(64)


if __name__ == "__main__":
    main()
