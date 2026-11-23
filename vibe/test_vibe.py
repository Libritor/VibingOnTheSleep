"""End-to-end test of Vibing on the Sleep without a sleeper, a headband, a microphone or the agent's
account (a scripted stand-in replies; `python vibe/agent.py --smoke` tests the real one).

  1. the decoder on made signals at 256 and 100 Hz: rest, eyes, frowns, both, too weak, one frown
  2. the night voice: a sentence rendered by Kokoro and transcribed back by faster-whisper
  3. the evening, simulated: plan, signal practice on made signals, open questions, hidden split
  4. the night: a Sleep-EDF night replayed through vibe/night.py from before sleep onset for 3 h.
     Made signals are added to the response windows (eyes after the sleep-onset question, frowns
     after the awakening question) and the 'spoken' words are scripted. Checked: sleep onset is
     found and lands in scored sleep; the question is asked and its YES reaches the agent; the
     dictation reaches the agent; a stir later in the night gets one whispered exchange and its NO
     reaches the agent; cues play only for the cued questions and land in scored N2/N3; the
     unasked windows are counted
  5. the morning, simulated: the reveal verifies the split's commitment

What this does not test: a real face making the signals in real sleep, the forehead rule on a
Muse (Sleep-EDF is Fpz-Cz), and whether any of it helps.

    python vibe/test_vibe.py            # about two minutes
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vibe"))
import common  # noqa: E402,F401
import evening  # noqa: E402
import morning  # noqa: E402
import night as night_mod  # noqa: E402
from agent import StubAgent  # noqa: E402
from common import Ears  # noqa: E402
from false_alarms import STAGE  # noqa: E402
from signals import Decoder, synth_eyes, synth_muscle, synth_rest  # noqa: E402

PID = "P-VIBETEST"
PSG = ROOT / "data" / "sc" / "SC4001E0-PSG.edf"
HYP = ROOT / "data" / "sc" / "SC4001EC-Hypnogram.edf"
START_H, HOURS = 8.0, 3.0


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def test_decoder():
    for fs in (256.0, 100.0):
        n = int(10 * fs)
        d = Decoder()
        got = [d.decode(synth_rest(n, fs, 1), fs)["answer"],
               d.decode(synth_rest(n, fs, 2) + synth_eyes(n, fs), fs)["answer"],
               d.decode(synth_rest(n, fs, 3) + synth_muscle(n, fs), fs)["answer"],
               d.decode(synth_rest(n, fs, 4) + synth_eyes(n, fs) + synth_muscle(n, fs, starts=(5.0, 6.2)), fs)["answer"],
               d.decode(synth_rest(n, fs, 5) + synth_eyes(n, fs, amp=30), fs)["answer"],
               d.decode(synth_rest(n, fs, 6) + synth_muscle(n, fs, starts=(1.0,)), fs)["answer"]]
        check(got == [None, "yes", "no", "ambiguous", None, None],
              f"decoder at {fs:.0f} Hz: rest, eyes, two frowns, both, weak eyes, one frown -> {got}")
    rows = synth_rest(2560, 256.0, 7) + synth_eyes(2560, 256.0)
    acc = np.tile([0.0, 0.0, 1.0], (2560, 1)) + np.random.default_rng(0).normal(0, 0.05, (2560, 3))
    check(Decoder().decode(rows, 256.0, acc)["answer"] is None, "a window with head movement gives no answer")
    rows[:, 1] = 0.0
    check(Decoder().decode(rows, 256.0)["answer"] is None, "eyes cannot be read with one forehead electrode off")


def test_voice():
    from voice import MODEL_DIR, Voice
    if not (MODEL_DIR / "kokoro-v1.0.onnx").exists():
        print("SKIP voice: the Kokoro model is not downloaded")
        return
    import soundfile as sf
    text = "Should I add the dark theme? Eyes for yes. Frown for no."
    v = Voice("kokoro", enabled=False)
    y, fs = v.wav(text, "night")
    tmp = ROOT / "study" / "audio" / "vibe" / "_test.wav"
    sf.write(str(tmp), y, fs)
    heard = Ears(enabled=True).transcribe(tmp).lower()
    tmp.unlink(missing_ok=True)
    tmp.with_suffix(".txt").unlink(missing_ok=True)
    want = [w.strip(".?,").lower() for w in text.split()]
    hit = sum(w in heard for w in want) / len(want)
    check(hit >= 0.9, f"night voice understood by the transcriber: {hit:.0%} of words ({len(y) / fs:.1f} s)")
    check(abs(float(np.sqrt(np.mean(y ** 2))) - 0.05) < 0.02, "night voice is at the cue sounds' level")


def stage_lookup():
    import mne
    ann = mne.read_annotations(HYP)
    stages = []
    for on, du, de in zip(ann.onset, ann.duration, ann.description):
        stages += [STAGE.get(de, "?")] * int(du // 30)
    return lambda t_s: stages[min(len(stages) - 1, int((START_H * 3600 + t_s) // 30))]


class StirringNight(night_mod.Night):
    """The replay has no accelerometer, so the sleeper's stir is scripted: 40 s of waking signs
    with movement, 100 minutes into the replay."""
    stir = (6000.0, 6040.0)

    def waking(self, t, seconds):
        if self.stir[0] <= t <= self.stir[1]:
            return 15, 3
        return super().waking(t, seconds)


def test_night(sdir):
    pdir = sdir.parent.parent
    agent = StubAgent.load(sdir / "agent_state.json", sdir / "agent_turns.jsonl")
    agent.replies = [
        dict(summary="The first version runs and its test passes.", question="Should I add a second example?",
             if_yes="Add a second example.", done=True),
        dict(summary="The second example is in.", question=None, done=True),
        dict(summary="Lines are counted too now.", question="Should it also read from the keyboard?", done=True),
        dict(summary="Left as it is.", question=None, done=True)]
    said = ["make it count the lines as well, lines and words, like a river", "yes I think so", ""]

    def inject(context, t, rows, fs):
        n = len(rows)
        if context == "asked:sleep onset":
            return rows + synth_eyes(n, fs, amp=250.0, start=2.0)
        if context == "asked:awakening":
            return rows + synth_muscle(n, fs, amp=40.0, starts=(2.0, 3.3))
        return rows
    a = night_mod.parser().parse_args([
        "--participant", PID, "--session", str(sdir), "--source", "replay", "--replay", str(PSG),
        "--replay-start-hours", str(START_H), "--replay-hours", str(HOURS), "--quiet", "--agent", "stub"])
    t0 = time.time()
    n = StirringNight(a, agent=agent, ears=Ears(enabled=False, scripted=said), inject=inject)
    n.run()
    s = n.summary
    stage = stage_lookup()
    print(f"  ({HOURS:.0f} h replayed in {time.time() - t0:.0f} s)")

    check(s["wakes"] == 1 and len(s["sleep_onsets"]) == 1, "one wake at sleep onset, as set")
    on = s["sleep_onsets"][0]
    st = stage(on["t_s"])
    check(st in ("N1", "N2"), f"sleep onset found {on['t_s'] / 60:.0f} min in by the {on['by']} rule, scored {st} there")
    q = s["questions_in_sleep"]
    check(len(q) == 1 and q[0]["question"] == "Should I add a second example?" and q[0]["answer"] == "yes",
          f"the agent's question was asked at sleep onset and the eye signal read as yes ({q[0]['why']})")
    seen = "\n".join(agent.seen)
    check("YES to your question: Should I add a second example?" in seen, "the yes reached the agent as an ANSWER")
    check("DICTATION" in seen and "like a river" in seen, "the spoken report reached the agent as a DICTATION")
    check(on.get("said_asleep") == "yes", "the 'were you asleep' answer was kept with the trigger")
    w = s["awakenings"]
    check(len(w) == 1 and w[0]["answer"] == "no" and w[0]["signal"] == "no",
          f"one whispered exchange at the stir ({w[0]['clock']}), two frowns read as no")
    check("NO to your question: Should it also read from the keyboard?" in seen, "the no reached the agent")
    split = json.loads((sdir / "questions" / "split.json").read_text())
    per = s["cues"]["per_question"]
    check(s["cues"]["total"] > 0 and set(per) <= set(split["cued"]) and not set(per) & set(split["uncued"]),
          f"{s['cues']['total']} cues in the night, only for the {len(split['cued'])} cued questions")
    import csv
    with open(n.out / "events.csv", newline="", encoding="utf-8") as f:
        ev = list(csv.DictReader(f))
    at = [stage(float(e["t_s"])) for e in ev if e["event"] == "cue"]
    deep = sum(x in ("N2", "N3") for x in at) / max(len(at), 1)
    check(deep >= 0.9, f"{deep:.0%} of cues in scored N2 or N3 ({sum(x == 'N3' for x in at)} of {len(at)} in N3)")
    said_ev = [e for e in ev if e["event"] == "say"]
    cue_t = [float(e["t_s"]) for e in ev if e["event"] == "cue"]
    check(all(abs(float(e["t_s"]) - c) > 1.0 for e in said_ev for c in cue_t), "no cue played over a spoken sentence")
    u = s["unasked_windows"]
    check(u["usable"] > 500, f"{u['usable']} unasked 10-s windows decoded: accidental yes {u['accidental_yes_pct']}%, "
                             f"no {u['accidental_no_pct']}%")
    check((n.out / "night.json").exists() and (sdir / "agent_state.json").exists(), "night.json and the agent's state saved")
    return s


def main():
    for p in (PSG, HYP):
        if not p.exists():
            sys.exit(f"{p} is missing: fetch the Sleep-EDF nights first (README, 'Rebuild the validation').")
    pdir = ROOT / "study" / "participants" / PID
    shutil.rmtree(pdir, ignore_errors=True)
    pdir.mkdir(parents=True)
    (pdir / "consent_vibe.json").write_text(json.dumps(dict(participant=PID, reb_code="SELF-EXPERIMENT",
                                                            consented=True, withdrawn=False, study="vibe")))
    test_decoder()
    test_voice()
    sdir = evening.main(["--participant", PID, "--simulate", "0.3", "--brief", "A tiny word counting tool with one test."])
    ev = json.loads((sdir / "evening.json").read_text())
    sig = json.loads((sdir / "signals.json").read_text())
    man = json.loads((sdir / "questions" / "manifest.json").read_text())
    split = json.loads((sdir / "questions" / "split.json").read_text())
    check(sig["verify_right"] == 6 and sig["calib_right"] == 14, "evening: signal practice on made signals, 14/14 and 6/6")
    check(sorted(split["cued"] + split["uncued"]) == sorted(man["open"]) and abs(len(split["cued"]) - len(split["uncued"])) <= 1
          and len(man["open"]) >= 2, f"evening: {len(man['open'])} open questions split {len(split['cued'])} cued / "
                                     f"{len(split['uncued'])} uncued")
    check(Path(ev["workspace"]).exists() and ev["plan"]["one_liner"], "evening: workspace made, plan saved")
    test_night(sdir)
    morning.main(["--participant", PID, "--session", str(sdir), "--simulate", "0.6"])
    m = json.loads((sdir / "morning.json").read_text())
    check(m["reveal"]["split_unchanged"] and m["reveal"]["cued"]["n"] == len(split["cued"]),
          "morning: the reveal verifies the split's evening commitment")
    split["cued"], split["uncued"] = split["uncued"], split["cued"]
    from puzzle_common import commitment
    check(commitment(split["cued"], split["uncued"], split["salt"]) != man["split_commitment"] or split["cued"] == split["uncued"],
          "a changed split would be caught")
    shutil.rmtree(Path(ev["workspace"]), ignore_errors=True)
    print("\nALL PASS")


if __name__ == "__main__":
    main()
