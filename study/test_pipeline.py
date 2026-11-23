"""End-to-end test of rig -> scoring -> analysis without a sleeper.

Runs the nap rig in replay mode for ten scheduled naps (held-out Sleep-EDF nights), then writes
synthetic reports and use lists with a planted effect (cue naps mention the object) and checks:
  * with the effect, incorporation favours cue in every pair and one-sided p = 1/32
  * with no effect (shuffled), the test does not reach p < 0.05
Also checks a spoken report survives TTS -> wav -> local transcription.

    python study/test_pipeline.py [--skip-whisper]
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "study"
PY = sys.executable


def run(*args):
    r = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"failed: {args}")
    return r.stdout


def setup(pid):
    shutil.rmtree(STUDY / "participants" / pid, ignore_errors=True)
    p = subprocess.run([PY, "study/consent.py", "--participant", pid, "--self"], cwd=ROOT,
                       input="y\n" * 8, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    run("study/make_schedule.py", "--participant", pid, "--naps", "10", "--seed", "7")


def replay_naps(pid):
    import pandas as pd
    ep = pd.read_csv(ROOT / "data" / "epochs.csv")
    nights = sorted(ep.night.unique())[:10]
    sched = json.loads((STUDY / "participants" / pid / "schedule.json").read_text())["naps"]
    for nap, night in zip(sched, nights):
        g = ep[ep.night == night].sort_values("epoch")
        start = g[g.stage != "W"].iloc[0].t - 600
        psg = next((ROOT / "data" / "sc").glob(f"{night}*PSG.edf"))
        run("rig/nap_rig.py", "--participant", pid, "--source", "replay", "--replay", str(psg),
            "--replay-start", str(start), "--nap", str(nap["index"]), "--quiet", "--max-minutes", "30")


def plant(pid, effect):
    import random
    rnd = random.Random(3)
    for man in (STUDY / "participants" / pid / "sessions").glob("*/manifest.json"):
        m = json.loads(man.read_text())
        d, obj = man.parent, m["object"]
        cued = m["condition"] == "cue" if effect else rnd.random() < 0.5
        for k in range(1, m["cycles"] + 1):
            text = (f"I was holding a {obj} and turning it over" if cued and k == 1
                    else "I was thinking about the lecture and a train platform")
            (d / f"report_{k}.txt").write_text(text, encoding="utf-8")
        for phase, uses in (("pre", ["doorstop", "paperweight"]),
                            ("post", ["doorstop", "paperweight", "a hinge"] + (["a whistle"] if cued else []))):
            with open(d / f"aut_{phase}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "use"])
                w.writerows([[0, u] for u in uses])


def analyze(pid):
    sdir = STUDY / "participants" / pid / "scoring"
    shutil.rmtree(sdir, ignore_errors=True)
    run("study/score.py", "--participant", pid, "make-sheet")
    run("study/analyze.py", "--participant", pid)
    return json.loads((sdir / "results.json").read_text())


def whisper_check():
    sys.path.insert(0, str(ROOT / "rig"))
    from nap_rig import tts_wav
    sys.path.insert(0, str(STUDY))
    from transcribe import transcribe_file
    from faster_whisper import WhisperModel
    wav = tts_wav("I was standing on a bridge holding a brick, and the brick was warm.",
                  ROOT / "data" / "tts_test.wav")
    text = transcribe_file(WhisperModel("base", device="cpu", compute_type="int8"), wav)
    print("whisper:", text)
    assert "brick" in text.lower(), "transcription lost the object word"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-whisper", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="reuse existing replay sessions")
    a = ap.parse_args()
    pid = "P-SYNTH"
    if not a.reuse:
        setup(pid)
        replay_naps(pid)
    mans = [json.loads(p.read_text()) for p in (STUDY / "participants" / pid / "sessions").glob("*/manifest.json")]
    print(f"{sum(m['completed'] for m in mans)}/10 replay naps completed; "
          f"triggers {sum(len(m['triggers']) for m in mans)}, overshoots {sum(m['overshoots'] for m in mans)}")
    assert sum(m["completed"] for m in mans) == 10

    plant(pid, effect=True)
    r = analyze(pid)
    print("effect:", r["rating_source"], r["incorporation"], r["new_uses"])
    assert r["rating_source"] == "lexical"
    assert r["incorporation"]["pairs_favouring_cue"] == 5 and abs(r["incorporation"]["p_one_sided"] - 1 / 32) < 1e-9
    assert r["new_uses"]["p_one_sided"] < 0.05

    plant(pid, effect=False)
    r = analyze(pid)
    print("null:", r["incorporation"], r["new_uses"])
    assert r["incorporation"]["p_one_sided"] >= 0.05 or r["incorporation"]["mean_diff_cue_minus_sham"] <= 0

    if not a.skip_whisper:
        whisper_check()
    print("PASS")


if __name__ == "__main__":
    main()
