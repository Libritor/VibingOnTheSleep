"""Vibing on the Sleep: the evening, awake, about 15 minutes before bed.

  1  THE BRIEF. Type (or --dictate) what you want built tonight. With --project <repo> the agent
     works in a git worktree of that repository on a new branch; otherwise in a new folder under
     ~/VibingOnTheSleep.
  2  THE PLAN. The agent's first turn: its plan in a sentence, the few words the rig will whisper
     at sleep onset ("Think about ..."), a first yes/no question, and up to eight open questions
     only you can settle.
  3  SIGNAL PRACTICE, headband on. Eyes left-right-left-right = yes, two frowns = no. Fourteen
     prompted trials set the decoder's thresholds for your face; six more check it (those six were
     not used to set anything). Under 4 of 6 right, the night asks no yes/no questions.
  4  THE OPEN QUESTIONS. Each is read to you with its own short sound. Answer now (typed) or
     leave it for the night: half of the open ones (hidden) will have their sounds replayed in
     deep sleep, all are asked again in the morning.

    python vibe/evening.py --participant P-0001 --self
    python vibe/evening.py --participant P-0001 --self --project C:/path/to/repo --source ble
    python vibe/evening.py --participant P-0001 --self --source none      # no headband yet: skip step 3

Then at bedtime: python vibe/night.py --participant P-0001 --self
"""
import argparse
import datetime as dt
import json
import random
import secrets
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vibe"))
import common  # noqa: E402,F401
from agent import Agent, StubAgent, make_workspace  # noqa: E402
from common import Ears, new_session, participant_dir  # noqa: E402
from signals import Decoder, synth_eyes, synth_muscle, synth_rest  # noqa: E402
from voice import Voice  # noqa: E402

import puzzle_sounds as ps  # noqa: E402
from puzzle_common import SoundLoop, make_split  # noqa: E402
from sources import Buffer, MuseBLE, MuseOSC  # noqa: E402

WORD = {"rest": "Rest.", "eyes": "Eyes.", "muscle": "Frown."}
WIN = 6.0


def get_brief(a, ears, sdir):
    if a.brief:
        return a.brief
    if a.brief_file:
        return Path(a.brief_file).read_text(encoding="utf-8").strip()
    if a.dictate:
        input("\nTHE BRIEF. Press Enter, then say what you want built tonight. Stop talking when done... ")
        wav, secs = ears.record(sdir / "brief.wav", 180, min_seconds=10, quiet_seconds=5)
        text = ears.transcribe(wav)
        print(f"\nHeard ({secs:.0f} s):\n  {text}\n")
        fix = input("Enter to accept, or type the brief as it should read: ").strip()
        return fix or text
    print("\nTHE BRIEF. What should be built tonight? Type it; finish with an empty line.")
    lines = []
    while True:
        ln = input("> ")
        if not ln.strip() and lines:
            break
        if ln.strip():
            lines.append(ln.strip())
    return " ".join(lines)


def connect(a):
    buf = Buffer(seconds=60)
    if a.source == "ble":
        src = MuseBLE(buf, a.muse_name, log=lambda m: print("  " + m, flush=True))
        print("\nLooking for the Muse on this computer's Bluetooth...", flush=True)
    else:
        src = MuseOSC(buf, a.osc_port, log=lambda m: print("  " + m, flush=True))
        print(f"\nWaiting for MuseLog on UDP {a.osc_port} (SnowballArcade format, Full-rate raw EEG on)...", flush=True)
    src.start()
    t0 = time.time()
    while buf.last is None or time.time() - buf.last > 2:
        time.sleep(0.5)
        if time.time() - t0 > a.headband_wait:
            print(f"No headband after {a.headband_wait:.0f} s: skipping the signal practice. The night will use "
                  "default thresholds.")
            return None, None
    n0 = buf.count
    time.sleep(4)
    print(f"  headband streaming at {(buf.count - n0) / 4:.0f} Hz")
    return buf, src


def practice(a, voice, sdir, buf):
    """Step 3. Returns the saved record (also in <session>/signals.json)."""
    rng = random.Random(secrets.randbits(32) if a.simulate is None else 1)
    dec = Decoder()

    def trial(lab, k):
        if buf is None:                                   # simulated face
            n = int(WIN * 256)
            rows = synth_rest(n, 256.0, seed=1000 + k)
            if lab == "eyes":
                rows = rows + synth_eyes(n, 256.0, amp=rng.uniform(80, 140), start=rng.uniform(0.6, 1.5))
            elif lab == "muscle":
                rows = rows + synth_muscle(n, 256.0, amp=rng.uniform(15, 30), starts=(1.0, rng.uniform(1.9, 2.6)), seed=k)
            return rows, 256.0
        voice.say(WORD[lab], "day")
        time.sleep(WIN + 0.3)
        r = buf.raw(WIN)
        if r is None or buf.last is None or time.time() - buf.last > 3:
            raise RuntimeError("the headband stream stopped during the practice")
        return r[0], buf.fs

    if buf is not None:
        print("\nSIGNAL PRACTICE. Lie back or sit still, eyes CLOSED, and listen.\n"
              "  \"Eyes\"  -> look far left, far right, far left, far right (eyes still closed), two looks a second\n"
              "  \"Frown\" -> frown hard, relax, frown hard again (or clench your jaw twice)\n"
              "  \"Rest\"  -> do nothing\nAbout three minutes.")
        input("Press Enter, then close your eyes... ")
        voice.say("Signal practice. Keep your eyes closed. When I say eyes, look far left, far right, far left, "
                  "far right. When I say frown, frown hard twice. When I say rest, do nothing.", "day")
    labels = ["rest"] * 4 + ["eyes"] * 5 + ["muscle"] * 5
    rng.shuffle(labels)
    trials, fs = [], 256.0
    for k, lab in enumerate(labels):
        rows, fs = trial(lab, k)
        trials.append((lab, rows))
    fit = dec.fit(trials, fs)
    table, right, n = dec.score(trials, fs)
    vlabels = ["rest", "eyes", "muscle"] * 2
    rng.shuffle(vlabels)
    if buf is not None:
        voice.say("Good. Six more, the same way.", "day")
    want = {"rest": None, "eyes": "yes", "muscle": "no"}
    verify = []
    for k, lab in enumerate(vlabels):
        rows, fs = trial(lab, 100 + k)
        m = dec.decode(rows, fs)
        verify.append(dict(asked=lab, read=m["answer"], why=m["why"], right=m["answer"] == want[lab]))
    vr = sum(v["right"] for v in verify)
    if buf is not None:
        voice.say("Done. Open your eyes.", "day")
    rec = dict(fit=fit, calib_table=table, calib_right=right, calib_n=n, verify=verify, verify_right=vr,
               verify_n=len(verify), simulated=buf is None, recorded_at=dt.datetime.now().isoformat(timespec="seconds"))
    dec.save(sdir / "signals.json", **rec)
    np.savez_compressed(sdir / "signals_trials.npz", fs=fs, labels=np.array([lab for lab, _ in trials]),
                        rows=np.stack([r[:int(WIN * fs) - 2] for _, r in trials]))
    print(f"\n  eye step: yours {fit['eye_amp']:.0f} uV against {fit['eye_noise']:.0f} uV at rest -> threshold {dec.p['eye_thr']:.0f} uV")
    print(f"  frown:    yours {fit['emg_amp']:.1f} uV against {fit['emg_base']:.1f} uV at rest -> threshold {dec.p['emg_thr']:.1f} uV")
    print(f"  the 14 practice trials read back: {right} right   {table}")
    print(f"  the 6 check trials (not used for the thresholds): {vr} of 6 right")
    for v in verify:
        print(f"     asked {v['asked']:6s} read {str(v['read']):9s} {'' if v['right'] else '<- ' + v['why']}")
    if vr < 4:
        print("  Under 4 of 6: the night will ask no yes/no questions (run the evening again to retry, or pass "
              "--force-probes to the night). Usual causes: forehead electrodes not on skin, or small eye movements.")
    return rec


def open_questions(a, voice, sdir, plan):
    """Step 4. Writes <session>/questions/{manifest.json, split.json}; returns the evening's answers."""
    qs = [q.strip() for q in plan.get("open_questions", []) if q.strip()][:a.questions]
    qdir = sdir / "questions"
    qdir.mkdir()
    ids = [f"Q{k + 1}" for k in range(len(qs))]
    seed = secrets.randbits(30)
    sounds = dict(zip(ids, ps.specs_for_set(len(qs), seed))) if qs else {}
    answered = {}
    sim = random.Random(2) if a.simulate is not None else None
    if qs and not sim:
        print(f"\nTHE OPEN QUESTIONS: {len(qs)}. Each has its own short sound. Hold each one in mind while its sound "
              "repeats. Type an answer if you already have one, or press Enter alone to sleep on it.")
        input("Press Enter to begin... ")
    for qid, q in zip(ids, qs):
        if sim:
            if sim.random() < a.simulate:
                answered[qid] = "simulated evening answer"
            continue
        y = ps.make(sounds[qid])
        print(f"\n--- {qid} of {len(qs)} ---\n{q}")
        if voice.enabled:
            voice.play(y, ps.FS, 0.0)
            voice.say(q, "day")
        t0 = time.time()
        with SoundLoop(y, a.sound_every, 0.0, enabled=voice.enabled):
            text = input("answer now, or Enter to sleep on it > ").strip()
            if not text and time.time() - t0 < a.think_seconds:
                left = a.think_seconds - (time.time() - t0)
                print(f"  (keep it in mind for {left:.0f} s more, so the sound is tied to it)")
                time.sleep(left)
        if text:
            answered[qid] = text
    open_ids = [q for q in ids if q not in answered]
    split = make_split([(q, "question") for q in open_ids], random.Random(secrets.randbits(64)))
    (qdir / "split.json").write_text(json.dumps(split, indent=1))
    man = dict(session=sdir.name, questions=dict(zip(ids, qs)), sounds=sounds, answered_evening=answered,
               open=open_ids, cued_n=len(split["cued"]), uncued_n=len(split["uncued"]),
               split_commitment=split["commitment"], sound_seed=seed)
    (qdir / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    if qs:
        print(f"\n{len(answered)} answered now, {len(open_ids)} left open: {len(split['cued'])} will be cued in deep "
              f"sleep, {len(split['uncued'])} will not (which is which stays hidden until the morning).")
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--self", action="store_true", help="investigator self-experiment (no REB code)")
    ap.add_argument("--project", help="a git repository: the agent works in a worktree of it on a new branch")
    ap.add_argument("--brief", help="tonight's brief as text")
    ap.add_argument("--brief-file")
    ap.add_argument("--dictate", action="store_true", help="say the brief instead of typing it")
    ap.add_argument("--source", choices=("osc", "ble", "none"), default="osc", help="headband for the signal practice")
    ap.add_argument("--osc-port", type=int, default=5000)
    ap.add_argument("--muse-name", default="")
    ap.add_argument("--headband-wait", type=float, default=180.0)
    ap.add_argument("--questions", type=int, default=6, help="most open questions to take into the night")
    ap.add_argument("--think-seconds", type=float, default=25.0, help="least time with each open question")
    ap.add_argument("--sound-every", type=float, default=8.0)
    ap.add_argument("--voice", default="kokoro", choices=("kokoro", "edge", "sapi"))
    ap.add_argument("--stt-model", default="base")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--agent", choices=("claude", "stub"), default="claude")
    ap.add_argument("--agent-model")
    ap.add_argument("--agent-budget", type=float, default=3.0, help="dollars per agent turn")
    ap.add_argument("--agent-turns", type=int, default=8, help="most agent turns from now to the morning")
    ap.add_argument("--simulate", type=float, help="test only: no typing, no headband; each open question answered "
                                                  "in the evening with this probability")
    a = ap.parse_args(argv)
    sim = a.simulate is not None

    pdir = participant_dir(a.participant, a.self or sim)
    sdir, stamp = new_session(pdir)
    audio = not (a.no_audio or sim)
    voice = Voice(a.voice, enabled=audio)
    ears = Ears(enabled=audio, stt_model=a.stt_model)
    if audio:
        ears.check()
        voice.prerender(list(WORD.values()), "day")
    brief = (a.brief or "A tiny test project.") if sim else get_brief(a, ears, sdir)
    if len(brief.split()) < 3:
        sys.exit("The brief is empty. Nothing to build tonight.")

    ws, branch, project = make_workspace(stamp, a.project)
    cls = StubAgent if (a.agent == "stub" or sim) else Agent
    agent = cls(ws, sdir / "agent_turns.jsonl", model=a.agent_model, budget_usd=a.agent_budget, max_turns=a.agent_turns)
    print(f"\nWorkspace: {ws}  (branch {branch}" + (f", a worktree of {project})" if project else ")"))
    print("The agent is reading the brief and planning (a minute or two)...", flush=True)
    plan = agent.plan_turn(brief)
    print(f"\nTHE PLAN\n  {plan['summary']}\n  at sleep onset you will hear: \"Think about {plan['one_liner']}.\"")
    if plan.get("question"):
        print(f"  its first yes/no question, for the night: {plan['question']}")
    if audio:
        voice.say(plan["summary"], "day")

    sig = None
    if sim:
        sig = practice(a, voice, sdir, None)
    elif a.source != "none":
        buf, _ = connect(a)
        if buf is not None:
            try:
                sig = practice(a, voice, sdir, buf)
            except RuntimeError as e:                     # the plan is already made: do not lose the evening
                print(f"\nSignal practice abandoned ({e}). The night will use default thresholds.")
    man = open_questions(a, voice, sdir, plan)
    if man["answered_evening"]:
        agent.submit("brief", "Answers given in the evening to your open questions:\n" + "\n".join(
            f"- {man['questions'][q]}\n  ANSWER: {t}" for q, t in man["answered_evening"].items()))

    agent.save(sdir / "agent_state.json")
    (sdir / "evening.json").write_text(json.dumps(dict(
        participant=a.participant, session=stamp, recorded_at=dt.datetime.now().isoformat(timespec="seconds"),
        brief=brief, workspace=str(ws), branch=branch, project=project, plan=plan,
        signals=None if sig is None else dict(verify_right=sig["verify_right"], verify_n=sig["verify_n"]),
        open_questions=len(man["open"]), simulated=sim), indent=1), encoding="utf-8")
    if not sim:
        print(f"\nEvening saved: {sdir}\nAt bedtime, with the headband streaming:\n"
              f"  python vibe/night.py --participant {a.participant} --self" + (" --source ble" if a.source == "ble" else ""))
    return sdir


if __name__ == "__main__":
    main()
