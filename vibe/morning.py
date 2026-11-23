"""Vibing on the Sleep: the morning, awake, about 10 minutes.

  1  Say anything you remember: dreams, ideas, what you want changed (recorded, transcribed here).
  2  The open questions again, without their sounds. Answer aloud; then a confidence from 1 to 5.
  3  The agent gets all of it, finishes, and writes MORNING.md in the workspace.
  4  The reveal: which questions were cued in deep sleep (and the proof the split was not changed),
     what the night's exchanges were, how often the decoder saw a signal nobody made, and the
     branch the agent built. Nothing is merged: the last lines say how to keep or discard it.

    python vibe/morning.py --participant P-0001 --self
"""
import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vibe"))
import common  # noqa: E402,F401
from agent import Agent, StubAgent, git  # noqa: E402
from common import Ears, find_session, participant_dir  # noqa: E402
from voice import Voice  # noqa: E402

from puzzle_common import commitment  # noqa: E402


def latest_night(sdir):
    nights = sorted(p for p in sdir.glob("night*") if (p / "night.json").exists())
    return nights[-1] if nights else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--self", action="store_true")
    ap.add_argument("--session", default="latest")
    ap.add_argument("--voice", default="kokoro", choices=("kokoro", "edge", "sapi"))
    ap.add_argument("--stt-model", default="base")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--agent", choices=("claude", "stub"), default="claude")
    ap.add_argument("--simulate", type=float, help="test only: each open question answered with this probability")
    a = ap.parse_args(argv)
    sim = a.simulate is not None
    pdir = participant_dir(a.participant, a.self or sim)
    sdir = find_session(pdir, a.session)
    ev = json.loads((sdir / "evening.json").read_text(encoding="utf-8"))
    ndir = latest_night(sdir)
    night = json.loads((ndir / "night.json").read_text(encoding="utf-8")) if ndir else None
    audio = not (a.no_audio or sim)
    voice, ears = Voice(a.voice, enabled=audio), Ears(enabled=audio, stt_model=a.stt_model)
    mdir = sdir / "morning"
    mdir.mkdir(exist_ok=True)
    rng = random.Random(3 if sim else None)

    # 1 the waking report
    report = ""
    if not sim:
        input("\nGOOD MORNING. Press Enter, then say anything you remember: dreams, ideas, what you want changed... ")
        voice.say("Good morning. Tell me anything you remember.", "day")
        wav, secs = ears.record(mdir / "report.wav", 120, min_seconds=8, quiet_seconds=6)
        report = ears.transcribe(wav)
        print(f"\nHeard ({secs:.0f} s): {report or '(nothing)'}")

    # 2 the open questions, all of them, without sounds
    qdir = sdir / "questions"
    man = json.loads((qdir / "manifest.json").read_text(encoding="utf-8")) if (qdir / "manifest.json").exists() else None
    answers = {}
    if man and man["open"]:
        order = list(man["open"])
        rng.shuffle(order)
        if not sim:
            print(f"\nTHE OPEN QUESTIONS: {len(order)}. Answer each aloud in a sentence or two, or say 'no answer'.")
        for qid in order:
            q = man["questions"][qid]
            if sim:
                ok = rng.random() < a.simulate
                answers[qid] = dict(text="simulated answer" if ok else "", confidence=rng.randint(2, 5) if ok else None)
                continue
            print(f"\n{qid}: {q}")
            voice.say(q, "day")
            wav, _ = ears.record(mdir / f"{qid}.wav", 40, min_seconds=5, quiet_seconds=4)
            text = ears.transcribe(wav)
            print(f"  heard: {text or '(nothing)'}")
            fix = input("  Enter to keep, or type the answer as it should read: ").strip()
            text = fix or text
            none = not text.strip() or text.lower().strip(" .") in ("no answer", "i don't know", "skip")
            conf = None
            if not none:
                c = input("  how sure, 1 (a guess) to 5 (settled)? ").strip()
                conf = int(c) if c in "12345" and c else None
            answers[qid] = dict(text="" if none else text, confidence=conf)

    # 3 the agent's last turn
    cls = StubAgent if (a.agent == "stub" or sim) else Agent
    agent = cls.load(sdir / "agent_state.json", sdir / "agent_turns.jsonl",
                     event=lambda n, d="", **k: print(f"  [{n}] {d}"), autonomous=False)
    agent.max_turns = max(agent.max_turns, agent.turns + 1)          # the morning always gets its turn
    parts = [f"The operator's waking report: {report or '(none)'}"]
    if answers:
        parts.append("Their answers to your open questions:\n" + "\n".join(
            f"- {man['questions'][q]}\n  ANSWER: {v['text'] or '(no answer)'}"
            + (f"  (confidence {v['confidence']}/5)" if v["confidence"] else "") for q, v in answers.items()))
    parts.append("Apply what follows from these, run the checks, commit, and write MORNING.md at the top of the "
                 "folder: what now exists and how to run it, what each of the night's inputs led to, what you are "
                 "unsure of or left undone. Then stop.")
    agent.submit("morning", "\n\n".join(parts))
    print("\nThe agent is finishing (a few minutes)...", flush=True)
    agent.poll()
    out = agent.wait(1500)
    agent.save(sdir / "agent_state.json")

    # 4 the reveal
    reveal = None
    if man and (qdir / "split.json").exists():
        split = json.loads((qdir / "split.json").read_text())
        honest = commitment(split["cued"], split["uncued"], split["salt"]) == man["split_commitment"]
        per = (night or {}).get("cues", {}).get("per_question", {})

        def rate(ids):
            got = [q for q in ids if answers.get(q, {}).get("text")]
            conf = [answers[q]["confidence"] for q in got if answers[q]["confidence"]]
            return dict(n=len(ids), answered=len(got), mean_confidence=round(sum(conf) / len(conf), 2) if conf else None)
        reveal = dict(split_unchanged=honest, cued=rate(split["cued"]), uncued=rate(split["uncued"]),
                      cues_played=per)
        print("\nTHE REVEAL" + ("" if honest else "  (WARNING: the split file does not match its evening commitment)"))
        for qid in man["open"]:
            tag = "CUED  " if qid in split["cued"] else "uncued"
            v = answers.get(qid, {})
            print(f"  {qid} {tag} played {per.get(qid, 0):3d}x  answered {'yes' if v.get('text') else 'no '}"
                  f"  confidence {v.get('confidence') or '-'}  {man['questions'][qid][:60]}")
        print(f"  cued: {reveal['cued']}   uncued: {reveal['uncued']}")
        print("  One night settles nothing: with a handful of questions this is a record, not a test.")
    if night:
        k, u = night["known_answer_questions"], night["unasked_windows"]
        print(f"\nTHE NIGHT ({night['hours']} h)")
        print(f"  wakes at sleep onset: {night['wakes']}  " + "; ".join(
            f"{t['clock']} by {t['by']}, said asleep: {t.get('said_asleep')}" for t in night["sleep_onsets"]))
        print(f"  spoken reports: {[r['words'] for r in night['reports']]} words")
        print(f"  yes/no questions in sleep: {len(night['questions_in_sleep'])}, answered "
              f"{sum(p['answer'] is not None for p in night['questions_in_sleep'])}; with a known answer: "
              f"{k['asked']} asked, {k['answered']} answered, {k['right']} right")
        print(f"  signals nobody asked for: yes in {u['accidental_yes_pct']}% and no in {u['accidental_no_pct']}% of "
              f"{u['usable']} quiet 10-s windows (forehead off in {u['forehead_off']})")
        print(f"  exchanges at awakenings: {len(night['awakenings'])};  cues in deep sleep: {night['cues']['total']}")
    ws = Path(ev["workspace"])
    print(f"\nTHE WORK  ({agent.turns} agent turns, about ${agent.cost:.2f})")
    if out:
        print(f"  {out['summary']}")
    log = git(ws, "log", "--oneline", "-15").stdout.strip()
    print("  " + (log.replace("\n", "\n  ") or "(no commits)"))
    print(f"  folder: {ws}\n  read first: {ws / 'MORNING.md'}  and  {ws / 'NIGHT_LOG.md'}")
    if ev.get("project"):
        print(f"  to keep it:    git -C \"{ev['project']}\" merge {ev['branch']}\n"
              f"  to discard it: git -C \"{ev['project']}\" worktree remove \"{ws}\"  then  git -C \"{ev['project']}\" branch -D {ev['branch']}")
    else:
        print("  it is its own repository; nothing else was touched. Delete the folder to discard it.")
    (sdir / "morning.json").write_text(json.dumps(dict(
        recorded_at=dt.datetime.now().isoformat(timespec="seconds"), report=report, answers=answers,
        reveal=reveal, agent=dict(turns=agent.turns, cost_usd=round(agent.cost, 2), last=out and out.get("summary")),
        simulated=sim), indent=1), encoding="utf-8")
    return sdir


if __name__ == "__main__":
    main()
