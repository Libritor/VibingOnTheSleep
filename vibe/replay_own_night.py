"""Play one of this rig's own headband recordings through tonight's night logic, in simulated
time, with a scripted stand-in for the agent and no sound: what would the rig have done on that
night? Sleep onset by which rule and when, exchanges at awakenings (the recording's real
accelerometer decides), cues, and how many signals the decoder saw that nobody made.

    python vibe/replay_own_night.py data/muselog/20261028/muse_session_20261028_034217_Muse2.npz
    python vibe/replay_own_night.py study/participants/P-0001/nights/20261029_031006/raw.npz --start-minutes 0

The recording is an .npz from study/muselog_load.py (a phone CSV) or study/rawcsv_to_npz.py (the
rig's raw.csv). Nothing was asked on those nights, so no answer here is a real answer.
"""
import argparse
import csv
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vibe"))
import common  # noqa: E402,F401
import evening  # noqa: E402
import night as night_mod  # noqa: E402
from agent import StubAgent  # noqa: E402
from common import Ears  # noqa: E402

PID = "P-VIBEREPLAY"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--start-minutes", type=float, default=0.0)
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--wakes", type=int, default=1)
    ap.add_argument("--keep", action="store_true", help="keep the scratch session folder")
    a, rest = ap.parse_known_args()
    z = np.load(a.npz, allow_pickle=True)
    clock0 = dt.datetime.fromisoformat(str(z["clock0"])) if "clock0" in z.files else None
    pdir = ROOT / "study" / "participants" / PID
    shutil.rmtree(pdir, ignore_errors=True)
    pdir.mkdir(parents=True)
    (pdir / "consent_vibe.json").write_text(json.dumps(dict(participant=PID, reb_code="SELF-EXPERIMENT",
                                                            consented=True, withdrawn=False, study="vibe")))
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        sdir = evening.main(["--participant", PID, "--simulate", "0.0", "--brief", "A replay of a recorded night."])
    (sdir / "signals.json").unlink()                     # made-signal thresholds are not this face's: use the defaults
    agent = StubAgent.load(sdir / "agent_state.json", sdir / "agent_turns.jsonl")
    args = night_mod.parser().parse_args([
        "--participant", PID, "--session", str(sdir), "--source", "replay", "--replay", a.npz,
        "--replay-start-hours", str(a.start_minutes / 60), "--replay-hours", str(a.hours), "--quiet",
        "--agent", "stub", "--wakes", str(a.wakes), *rest])
    n = night_mod.Night(args, agent=agent, ears=Ears(enabled=False, scripted=[]))
    n.run()
    s = n.summary

    def wall(t_s):
        if clock0 is None:
            return f"{t_s / 60:.0f} min"
        return (clock0 + dt.timedelta(minutes=a.start_minutes, seconds=t_s)).strftime("%H:%M:%S")
    with open(n.out / "events.csv", newline="", encoding="utf-8") as f:
        ev = list(csv.DictReader(f))
    print(f"\n{Path(a.npz).name}: {s['hours']} h replayed from {wall(0)}")
    for e in ev:
        if e["event"] in ("calibrated", "sleep_onset", "deeper_than_planned", "wake", "no_sleep_onset_seen", "awakening",
                          "cueing_on", "cues_ready", "answer", "no_answer", "question", "stream_ended"):
            print(f"  {wall(float(e['t_s']))}  {e['event']}: {e['detail'][:120]}")
    cues = [float(e["t_s"]) for e in ev if e["event"] == "cue"]
    if cues:
        print(f"  cues: {len(cues)}, {wall(cues[0])} to {wall(cues[-1])}")
    u = s["unasked_windows"]
    print(f"  unasked 10-s windows: {u['usable']} usable ({u['forehead_off']} with the forehead off): the decoder read "
          f"yes in {u['yes']} ({u['accidental_yes_pct']}%), no in {u['no']} ({u['accidental_no_pct']}%)")
    print(f"  questions in sleep: {[(q['context'], q['stage'], q['read']) for q in s['questions_in_sleep']]}")
    if not a.keep:
        shutil.rmtree(pdir, ignore_errors=True)
        shutil.rmtree(Path(agent.ws), ignore_errors=True)


if __name__ == "__main__":
    main()
