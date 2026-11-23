"""End-to-end test of the puzzle-cueing study without a sleeper.

  1. the bank: every puzzle solvable, every checker accepts a solution and refuses a wrong one,
     and the checkers' edge cases (unary minus, a number used twice, division by zero, too many
     resistors, an anagram that uses other letters)
  2. sounds: distinct (timbre, contour) pairs within a set, equal loudness
  3. the split: balanced by family, its commitment hash verifies, and a changed split is caught
  4. three simulated nights for a test participant: evening (simulated answers), the first night
     replayed from Sleep-EDF through rig/night_watch.py with --cues (cues must fire, land in
     scored N2/N3, and never be named on the console), morning (simulated answers), analysis

    python study/test_puzzles.py
"""
import contextlib
import io
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "study"))
sys.path.insert(0, str(ROOT / "rig"))
import puzzle_analysis  # noqa: E402
import puzzle_bank as pb  # noqa: E402
import puzzle_evening  # noqa: E402
import puzzle_morning  # noqa: E402
import puzzle_sounds as ps  # noqa: E402
from puzzle_common import commitment, make_split  # noqa: E402

PID = "P-PUZTEST"
PSG = ROOT / "data" / "sc" / "SC4001E0-PSG.edf"
HYP = ROOT / "data" / "sc" / "SC4001EC-Hypnogram.edf"


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


def test_bank():
    bank = pb.load()
    check(pb.selfcheck(bank), f"bank of {len(bank)} puzzles: every one solvable, checkers accept and refuse")
    a = dict(family="arith", numbers=[1, 3, 4, 6], target=24)
    check(pb.check(a, "6/(1-3/4)")[0] and pb.check(a, "6 ÷ (1 − 3/4)")[0] and pb.check(a, "6x4x1x(4-3)+0x6")[0] is False,
          "arithmetic: typed division and minus signs accepted")
    check(not pb.check(a, "-1+3*4*6/3")[0], "arithmetic: a number used twice or a unary minus is refused")
    check(not pb.check(a, "6/(3-3)+1+4")[0], "arithmetic: wrong numbers refused")
    check(not pb.check(dict(family="arith", numbers=[1, 1, 3, 4], target=24), "4/(1-1)*3")[0], "arithmetic: division by zero refused")
    r = dict(family="resistor", target="3/5", count=3)
    check(pb.check(r, "(1+1)|1")[0] is False and pb.check(dict(family="resistor", target="2/3", count=3), "(1+1)|1")[0],
          "resistors: value checked exactly")
    check(pb.check(dict(family="resistor", target="2", count=5), "((1+1)|(1+1))+1")[0],
          "resistors: parser handles nested brackets")
    check(not pb.check(dict(family="resistor", target="1/2", count=2), "(1|1)+(1|1)|(1|1)")[0],
          "resistors: right value with more resistors than allowed is refused")
    w = [p for p in bank if p["family"] == "anagram"][0]
    check(not pb.check(w, "X" * len(w["letters"]))[0], "anagram: other letters refused")
    txt = pb.BANK.read_text().upper()
    check(not any(f'"{x}"' in txt for x in pb.words()), "bank.json holds no anagram answer in plain text")


def test_sounds():
    specs = ps.specs_for_set(20, 5)
    pairs = {(s["timbre"], s["contour"]) for s in specs}
    check(len(pairs) == 20, "20 sounds in a set, all with different timbre and contour")
    rms = [float(np.sqrt(np.mean(ps.make(s) ** 2))) for s in specs]
    check(max(rms) / min(rms) < 1.05, f"equal loudness (RMS {min(rms):.4f}-{max(rms):.4f})")


def test_split():
    rng = random.Random(4)
    rows = [(f"A{i}", "arith") for i in range(5)] + [(f"R{i}", "resistor") for i in range(4)] + [(f"W{i}", "anagram") for i in range(5)]
    sp = make_split(rows, rng)
    fam = {i: f for i, f in rows}
    per = {f: (sum(fam[i] == f for i in sp["cued"]), sum(fam[i] == f for i in sp["uncued"])) for f in ("arith", "resistor", "anagram")}
    check(all(abs(c - u) <= 1 for c, u in per.values()) and abs(len(sp["cued"]) - len(sp["uncued"])) <= 1,
          f"split balanced by family {per}")
    check(commitment(sp["cued"], sp["uncued"], sp["salt"]) == sp["commitment"], "commitment verifies")
    swapped = sp["cued"][1:] + sp["uncued"][:1]
    check(commitment(swapped, sp["uncued"][1:] + sp["cued"][:1], sp["salt"]) != sp["commitment"], "a changed split is caught")


def test_end_to_end():
    pdir = ROOT / "study" / "participants" / PID
    if pdir.exists():
        shutil.rmtree(pdir)
    pdir.mkdir(parents=True)
    (pdir / "consent_puzzles.json").write_text(json.dumps(dict(
        participant=PID, reb_code="SELF-EXPERIMENT", study="puzzles", consented=True, withdrawn=False)))
    import night_watch
    sessions = []
    for k in range(3):
        with contextlib.redirect_stdout(io.StringIO()):
            sdir = puzzle_evening.main(["--participant", PID, "--simulate", "0.35", "--seed", str(100 + k)])
        sessions.append(sdir)
    m0 = [json.loads((s / "manifest.json").read_text()) for s in sessions]
    ids = [set(m["puzzle_ids"]) for m in m0]
    check(not (ids[0] & ids[1]) and not (ids[1] & ids[2]), "no puzzle repeats across nights")
    check(all(m["unsolved"] >= 4 for m in m0), f"evenings left {[m['unsolved'] for m in m0]} puzzles unsolved")

    split = json.loads((sessions[0] / "split.json").read_text())
    if not PSG.exists():
        print("SKIP night replay: Sleep-EDF SC4001 not downloaded")
    else:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            night_watch.main(["--participant", PID, "--replay", str(PSG), "--replay-start-hours", "8.0",
                              "--replay-hours", "2.5", "--cues", str(sessions[0])])
        console = out.getvalue()
        nc = json.loads((sessions[0] / "night_cues.json").read_text())
        check(nc["cues_total"] > 20, f"night replay: {nc['cues_total']} cues over {nc['minutes_cueing']} min")
        check(set(nc["cues_per_puzzle"]) <= set(split["cued"]), "only cued puzzles were played")
        check(not any(i in console for i in split["cued"] + split["uncued"]), "the console never names a puzzle")
        import csv
        import mne
        mne.set_log_level("ERROR")
        ann = mne.read_annotations(str(HYP))
        hyp = []
        for du, de in zip(ann.duration, ann.description):
            hyp += [de[-1]] * int(du // 30)
        ev = list(csv.DictReader(open(Path(nc["night_dir"]) / "events.csv")))
        stages = [hyp[int((8.0 * 3600 + float(e["t_s"])) // 30)] for e in ev if e["event"] == "cue"]
        share = sum(s in "234" for s in stages) / len(stages)
        check(share >= 0.9, f"cues in scored N2/N3: {share:.0%} ({len(stages)} cues)")
    for k, s in enumerate(sessions):
        with contextlib.redirect_stdout(io.StringIO()):
            puzzle_morning.main(["--participant", PID, "--session", str(s), "--simulate", "0.6", "0.2"])
        check((s / "result.json").exists(), f"morning {k + 1} scored")
    tampered = json.loads((sessions[2] / "split.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        rep = puzzle_analysis.main(["--participant", PID, "--include-simulated", "--draws", "2000"])
    itt = rep["intention_to_treat"]
    check(itt["nights"] == 3 and itt["difference"] > 0, f"analysis: planted effect seen ({itt['cued']} vs {itt['uncued']}, "
                                                       f"p one-sided {itt['p_one_sided']})")
    check((pdir / "puzzle_results" / "puzzle_cueing.pdf").exists(), "figure written (vector PDF)")
    # a split changed after the evening must stop the morning
    s = sessions[2]
    (s / "morning.csv").unlink()
    tampered["cued"], tampered["uncued"] = tampered["uncued"], tampered["cued"]
    (s / "split.json").write_text(json.dumps(tampered))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            puzzle_morning.main(["--participant", PID, "--session", str(s), "--simulate", "0.6", "0.2"])
        caught = False
    except SystemExit as e:
        caught = "commitment" in str(e)
    check(caught, "a split.json edited after the evening stops the morning")
    shutil.rmtree(pdir)


if __name__ == "__main__":
    test_bank()
    test_sounds()
    test_split()
    test_end_to_end()
    print("ALL PASS")
