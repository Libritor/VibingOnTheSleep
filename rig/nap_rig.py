"""Edison nap rig: sleep-onset dream incubation with a Muse headband, cue versus sham.

One run is one nap of one participant. The order of events:

  consent check -> pre-nap task (alternative uses of today's object, 3 min)
  -> calibration: 3 min sitting back, eyes closed, awake (sets the detector's baseline and
     the head's upright posture)
  -> watch: the trigger fires on a head nod (Muse accelerometer) or the EEG detector's
     P(asleep), whichever comes first (--trigger)
  -> the condition's audio plays: CUE ("think of a <object>") or SHAM ("notice your breathing")
  -> hold for --hold seconds, then the wake tone and "what were you just thinking about?"
  -> a spoken report is recorded; back to watch, up to --cycles times
  -> N2 guard throughout: if P(N2 or N3) passes --p-n2 the wake comes at once (overshoot)
  -> post-nap task (alternative uses again, 3 min)

The condition comes from the participant's pre-made schedule and is never printed. Everything is
logged under study/participants/<P>/sessions/. A replay mode feeds a Sleep-EDF night through the
same code so the rig can be tested without a sleeper.

    python rig/nap_rig.py --participant P-0001 --source ble
    python rig/nap_rig.py --participant P-0001 --source osc          # MuseLog on the phone
    python rig/nap_rig.py --participant P-TEST --source replay --replay data/sc/SC4001E0-PSG.edf
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "detector"))
sys.path.insert(0, str(ROOT / "rig"))
from model import LiveDetector  # noqa: E402
from sources import Buffer, MuseBLE, MuseOSC, Replay  # noqa: E402

try:
    import msvcrt
except ImportError:
    msvcrt = None

STUDY = ROOT / "study"
AUDIO = STUDY / "audio"
VERSION = "0.1.0"


# ------------------------------------------------------------------ helpers
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def git_rev():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "uncommitted"
    except Exception:
        return "unknown"


def tts_wav(text, path):
    """Windows' built-in speech synthesiser to a wav file, cached by text."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = text.replace("'", "''")
    ps = ("Add-Type -AssemblyName System.Speech; "
          "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = -2; "
          f"$s.SetOutputToWaveFile('{path}'); $s.Speak('{safe}'); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)
    return path


def tone(freq=880.0, seconds=0.25, gap=0.15, n=3, fs=44100, level=0.3):
    t = np.arange(int(seconds * fs)) / fs
    beep = level * np.sin(2 * np.pi * freq * t) * np.hanning(len(t))
    silence = np.zeros(int(gap * fs))
    return np.concatenate([np.concatenate([beep, silence]) for _ in range(n)]).astype(np.float32), fs


class Audio:
    def __init__(self, enabled):
        self.enabled = enabled
        if enabled:
            import sounddevice as sd
            import soundfile as sf
            self.sd, self.sf = sd, sf

    def play_wav(self, path, block=False):
        if not self.enabled:
            return 0.0
        data, fs = self.sf.read(str(path), dtype="float32")
        self.sd.play(data, fs)
        if block:
            self.sd.wait()
        return len(data) / fs

    def play_tone(self, **kw):
        if not self.enabled:
            return
        data, fs = tone(**kw)
        self.sd.play(data, fs)
        self.sd.wait()

    def check_mic(self):
        """Open the microphone for a moment so a blocked mic stops the session before the nap."""
        if not self.enabled:
            return
        try:
            self.sd.rec(1600, samplerate=16000, channels=1, dtype="float32")
            self.sd.wait()
        except Exception as e:
            raise SystemExit(f"The microphone cannot be opened ({e}). In Windows Settings > Privacy & "
                             "security > Microphone, turn on access for Python, then start again.")

    def record(self, path, seconds, fs=16000):
        if not self.enabled:
            return None
        rec = self.sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="float32")
        self.sd.wait()
        self.sf.write(str(path), rec, fs)
        return path


def timed_lines(seconds, prompt):
    """Collect typed lines until the time runs out (Windows console). Returns [(t, text)]."""
    print(prompt)
    if msvcrt is None:
        end = time.time() + seconds
        out = []
        while time.time() < end:
            out.append((time.time(), input("> ")))
        return out
    out, buf, start = [], "", time.time()
    end = start + seconds
    last_shown = -1
    sys.stdout.write("> ")
    sys.stdout.flush()
    while time.time() < end:
        left = int(end - time.time())
        if left % 30 == 0 and left != last_shown:
            last_shown = left
            sys.stdout.write(f"\r[{left // 60}:{left % 60:02d} left] > {buf}")
            sys.stdout.flush()
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                if buf.strip():
                    out.append((time.time(), buf.strip()))
                buf = ""
                sys.stdout.write("\n> ")
            elif ch == "\b":
                buf = buf[:-1]
                sys.stdout.write("\b \b")
            else:
                buf += ch
                sys.stdout.write(ch)
            sys.stdout.flush()
        else:
            time.sleep(0.02)
    if buf.strip():
        out.append((time.time(), buf.strip()))
    print("\nTime.")
    return out


# ------------------------------------------------------------------ participant records
def load_participant(pid, allow_self):
    pdir = STUDY / "participants" / pid
    consent = pdir / "consent.json"
    sched = pdir / "schedule.json"
    if not consent.exists():
        sys.exit(f"No consent record for {pid}. Run: python study/consent.py --participant {pid}")
    c = json.loads(consent.read_text())
    if not c.get("consented"):
        sys.exit(f"{pid} has not consented. The rig will not run.")
    if c.get("withdrawn"):
        sys.exit(f"{pid} has withdrawn. The rig will not run.")
    if c.get("reb_code", "") in ("", "SELF-EXPERIMENT") and not allow_self:
        sys.exit("No REB approval code on file. Self-experiments by an investigator need --self.")
    if not sched.exists():
        sys.exit(f"No schedule for {pid}. Run: python study/make_schedule.py --participant {pid}")
    s = json.loads(sched.read_text())
    return pdir, c, s


def next_nap(pdir, schedule):
    done = {json.loads(p.read_text())["nap_index"]
            for p in (pdir / "sessions").glob("*/manifest.json")
            if json.loads(p.read_text()).get("completed")}
    for nap in schedule["naps"]:
        if nap["index"] not in done:
            return nap
    sys.exit("Every nap in the schedule is done.")


# ------------------------------------------------------------------ the session
class Session:
    def __init__(self, args):
        self.a = args
        self.replay = args.source == "replay"
        allow_self = args.self or self.replay
        self.pdir, self.consent, self.schedule = load_participant(args.participant, allow_self)
        self.nap = next_nap(self.pdir, self.schedule) if args.nap is None else \
            next(n for n in self.schedule["naps"] if n["index"] == args.nap)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = self.pdir / "sessions" / f"nap{self.nap['index']:02d}_{stamp}"
        self.dir.mkdir(parents=True)
        cues = json.loads((STUDY / "cues.json").read_text())
        self.obj = self.nap["object"]
        self.cue_text = cues["cue"].format(object=self.obj)
        self.sham_text = cues["sham"]
        self.wake_text = cues["wake"]
        self.audio = Audio(enabled=not (self.replay or args.no_audio))
        model_path = Path(args.model)
        import joblib
        self.det = LiveDetector(joblib.load(model_path), step_s=5.0)
        self.manifest = {
            "rig_version": VERSION, "git": git_rev(), "participant": args.participant,
            "reb_code": self.consent.get("reb_code"), "consent_record": "consent.json",
            "consent_sha": sha(self.pdir / "consent.json"),
            "nap_index": self.nap["index"], "condition": self.nap["condition"], "object": self.obj,
            "source": args.source, "model": model_path.name, "model_sha": sha(model_path),
            "calibration_epochs": self.det.b["base_epochs"],
            "params": {k: v for k, v in vars(args).items() if k not in ("participant",)},
            "started": dt.datetime.now().isoformat(timespec="seconds"), "completed": False,
            "cycles": 0, "triggers": [], "overshoots": 0,
            "note": "condition is stored here and in schedule.json; the rig never prints it",
        }
        self.write_manifest()
        self.ev = open(self.dir / "events.csv", "w", newline="")
        self.evw = csv.writer(self.ev)
        self.evw.writerow(["t", "wall", "event", "detail"])
        self.dl = open(self.dir / "detector.csv", "w", newline="")
        self.dlw = csv.writer(self.dl)
        self.dlw.writerow(["t", "state", "contact_ok", "nod_deg", "p_W", "p_N1", "p_N2", "p_N3",
                           "p_REM", "p_sleep", "p_deep"])
        self.raw = None
        if not self.replay:
            self.raw = open(self.dir / "raw.csv", "w", newline="")
            self.raww = csv.writer(self.raw)
            self.raww.writerow(["t", "TP9", "AF7", "AF8", "TP10", "ax", "ay", "az"])
        self.t0 = None

    # -- logging
    def now(self):
        return self.sim_t if self.replay else time.time() - self.t0

    def event(self, name, detail=""):
        t = self.now()
        self.evw.writerow([f"{t:.2f}", dt.datetime.now().isoformat(timespec="seconds"), name, detail])
        self.ev.flush()
        if not self.a.quiet:
            print(f"[{t / 60:6.2f} min] {name} {detail}")

    def write_manifest(self):
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest, indent=1))

    def say(self, text, key, block=True):
        if not self.audio.enabled:
            return 0.0
        return self.audio.play_wav(tts_wav(text, AUDIO / f"{key}.wav"), block=block)

    # -- tasks
    def task(self, phase):
        if self.replay or self.a.task_minutes <= 0:
            return
        secs = self.a.task_minutes * 60
        lines = timed_lines(secs, f"\n{phase.upper()} TASK. Object: a {self.obj.upper()}.\n"
                                  f"Type as many different uses for a {self.obj} as you can, "
                                  f"one per line, Enter after each. {self.a.task_minutes} minutes.")
        with open(self.dir / f"aut_{phase}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "use"])
            for t, u in lines:
                w.writerow([f"{t - self.t0:.1f}", u])
        self.event(f"aut_{phase}_done", f"{len(lines)} uses")

    # -- nod
    def nod_angle(self, upright):
        acc = self.buf.accel
        if acc is None or upright is None:
            return float("nan")
        a, b = np.asarray(acc, float), np.asarray(upright, float)
        c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    # -- the run
    def run(self):
        self.buf = Buffer()
        if self.raw is not None:
            def sink(t, row, acc):
                acc = acc or ["", "", ""]
                self.raww.writerow([f"{t:.4f}", *[f"{v:.2f}" for v in row], *acc])
            self.buf.sink = sink
        if not self.replay:
            self.audio.check_mic()
        self.t0 = time.time()
        self.sim_t = 0.0
        if self.replay:
            self.src = Replay(self.buf, self.a.replay, start_s=self.a.replay_start)
        elif self.a.source == "ble":
            self.src = MuseBLE(self.buf, self.a.muse_name, log=lambda m: self.event("muse", m))
            self.src.start()
        else:
            self.src = MuseOSC(self.buf, self.a.osc_port, log=lambda m: self.event("muse", m))
            self.src.start()
        self.event("session_start", f"nap {self.nap['index']} object {self.obj}")

        if not self.replay:
            print("Waiting for the headband...")
            while self.buf.last is None or time.time() - self.buf.last > 2:
                time.sleep(0.5)
            self.event("headband_streaming")
            for _ in range(3):      # Wi-Fi can stall for a moment after connecting
                n0 = self.buf.count
                time.sleep(5)
                rate = (self.buf.count - n0) / 5
                self.event("eeg_rate", f"{rate:.0f} Hz")
                if rate >= 200:
                    break
            if rate < 200:
                raise SystemExit(f"EEG arrives at {rate:.0f} Hz, not 256. In MuseLog turn on "
                                 "Full-rate raw EEG (OSC settings), or use --source ble.")
            if self.buf.accel is None:
                print("No accelerometer data: the nod trigger is off and only the EEG can trigger. "
                      "Update MuseLog to a build that streams /muse/acc, or use --source ble.")
                self.event("no_accelerometer")
            self.task("pre")
            print("\nSit back, hold still, close your eyes and stay awake. Calibrating for "
                  f"{self.det.b['base_epochs'] * 0.5:.0f} min.")
            self.say("Sit back, close your eyes, and stay awake for a few minutes. "
                     "Then you may fall asleep.", "calib_start")
        self.event("calibration_start")
        loop0 = time.time()     # the loop clock starts here, not at session start

        state, upright, acc_hist = "CALIB", None, []
        acc_roll = collections.deque(maxlen=120)    # (t, accel) once a second, for the nod
        cycles, last_wake, hold_until, tick = 0, -1e9, None, 0
        p = None
        step = 1.0
        end_t = self.a.max_minutes * 60
        while True:
            # advance time
            if self.replay:
                self.sim_t += step
                if not self.src.advance_to(self.sim_t):
                    self.event("replay_end")
                    break
            else:
                time.sleep(max(0.0, loop0 + tick + step - time.time()))
            tick += step
            t = self.now()
            if t > end_t:
                self.event("max_time")
                break
            # live: the first 30 s of calibration still hold the pre-nap task, so skip them
            settling = state == "CALIB" and not self.replay and tick <= 30
            if self.buf.accel is not None:
                acc_roll.append((t, list(self.buf.accel)))
            if self.buf.accel is not None and state == "CALIB" and not settling:
                acc_hist.append(list(self.buf.accel))

            # detector every 5 s on the last 30 s
            if int(tick) % 5 == 0 and not settling:
                w = self.buf.window(30)
                ok = bool(w and w[3])
                if w and ok:
                    p = self.det.push(w[0], w[1], w[2])
                elif w and not ok:
                    p = None
                # a nod is a head drop against where the head was 10-40 s ago, not against the
                # calibration posture (people lean back further once told to sleep)
                ref = [a for ta, a in acc_roll if t - 40 <= ta <= t - 10]
                nod = self.nod_angle(list(np.median(np.array(ref), axis=0))
                                     if upright is not None and len(ref) >= 20 else None)
                ps = None if p is None else 1 - p["W"]
                pdp = None if p is None else p["N2"] + p["N3"]
                self.dlw.writerow([f"{t:.1f}", state, int(ok), f"{nod:.1f}"] +
                                  ([f"{p[s]:.4f}" for s in ("W", "N1", "N2", "N3", "REM")] +
                                   [f"{ps:.4f}", f"{pdp:.4f}"] if p else [""] * 7))
                self.dl.flush()

                if state == "CALIB" and self.det.calibrated:
                    if acc_hist:
                        upright = list(np.median(np.array(acc_hist), axis=0))
                    state = "WATCH"
                    self.event("calibrated", "nod trigger " + ("on" if upright else "off"))
                    if not self.replay:
                        self.say("Calibrated. You may fall asleep now.", "calib_done")

                # N2 guard
                if state in ("WATCH", "HOLD") and pdp is not None and pdp >= self.a.p_n2:
                    self.manifest["overshoots"] += 1
                    self.event("overshoot_N2", f"p_deep={pdp:.2f}")
                    state = "WAKE"

                if state == "WATCH" and t - last_wake >= self.a.refractory:
                    eeg_hit = ps is not None and ps >= self.a.p_sleep
                    nod_hit = (not math.isnan(nod) and nod >= self.a.nod_deg
                               and ps is not None and ps >= self.a.nod_p_sleep)
                    fire = {"either": eeg_hit or nod_hit, "eeg": eeg_hit, "nod": nod_hit,
                            "both": eeg_hit and nod_hit}[self.a.trigger]
                    if fire:
                        why = "+".join(n for n, h in (("eeg", eeg_hit), ("nod", nod_hit)) if h)
                        self.manifest["triggers"].append({"t": round(t, 1), "by": why,
                                                          "p_sleep": ps, "nod_deg": nod})
                        self.event("trigger", f"{why} p_sleep={ps if ps is None else round(ps, 2)} "
                                              f"nod={nod:.0f}")
                        cond = self.nap["condition"]
                        text = self.cue_text if cond == "cue" else self.sham_text
                        key = f"cue_{self.obj}" if cond == "cue" else "sham"
                        dur = self.say(text, key, block=False)
                        self.event("prompt_played", f"{dur:.1f}s")   # condition deliberately not shown
                        hold_until = t + dur + self.a.hold
                        state = "HOLD"

            if state == "HOLD" and t >= hold_until:
                state = "WAKE"

            if state == "WAKE":
                cycles += 1
                self.event("wake", f"cycle {cycles}")
                self.audio.play_tone()
                self.say(self.wake_text, "wake")
                try:
                    path = self.audio.record(self.dir / f"report_{cycles}.wav", self.a.report_seconds)
                    self.event("report_saved", str(path.name) if path else "none (no audio)")
                except Exception as e:      # never lose the report to an audio error: type it
                    self.event("report_audio_failed", str(e)[:120])
                    lines = timed_lines(self.a.report_seconds + 60, "\nThe microphone failed. Type "
                                        "everything you were just thinking about, then wait.")
                    (self.dir / f"report_{cycles}.txt").write_text(
                        " ".join(u for _, u in lines), encoding="utf-8")
                    self.event("report_saved", f"report_{cycles}.txt (typed)")
                if not self.replay:
                    # the only label of the sleeper's state the rig has: ask for it every wake
                    self.say("Were you asleep? Type a number from one to four.", "rating")
                    lines = timed_lines(20, "\nWere you asleep? 1 awake, 2 drowsy, 3 light sleep, "
                                            "4 deeply asleep. Type the number and Enter.")
                    first = next((u.strip() for _, u in lines if u.strip()), "")
                    self.event("sleep_rating", first[:1] if first[:1] in "1234" and first else "none")
                if self.replay:
                    tick += self.a.report_seconds          # the report takes time in the replay too
                    self.sim_t += self.a.report_seconds
                last_wake = self.now()
                self.manifest["cycles"] = cycles
                self.write_manifest()
                if cycles >= self.a.cycles:
                    break
                if not self.replay:
                    self.say("Thank you. Close your eyes again. You may fall asleep.", "again")
                state = "WATCH"

        self.event("nap_end", f"{cycles} cycles")
        if not self.replay:
            self.say("The nap is over. Open your eyes.", "nap_over")
        self.task("post")
        self.manifest["completed"] = cycles > 0
        self.manifest["ended"] = dt.datetime.now().isoformat(timespec="seconds")
        self.write_manifest()
        for f in (self.ev, self.dl, self.raw):
            if f:
                f.close()
        print(f"\nSaved to {self.dir}")
        return self.dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--source", choices=("ble", "osc", "replay"), default="ble")
    ap.add_argument("--muse-name", default="")
    ap.add_argument("--osc-port", type=int, default=5000)
    ap.add_argument("--replay", help="Sleep-EDF PSG file for --source replay")
    ap.add_argument("--replay-start", type=float, default=0.0, help="seconds into the night")
    ap.add_argument("--model", default=str(ROOT / "models" / "nap_base3min.joblib"))
    ap.add_argument("--nap", type=int, help="run this schedule index instead of the next one")
    ap.add_argument("--trigger", choices=("either", "eeg", "nod", "both"), default="either")
    ap.add_argument("--p-sleep", type=float, default=0.7, help="EEG trigger: P(not wake)")
    ap.add_argument("--p-n2", type=float, default=0.7, help="N2 guard: P(N2 or N3)")
    ap.add_argument("--nod-deg", type=float, default=20.0)
    ap.add_argument("--nod-p-sleep", type=float, default=0.3,
                    help="a nod counts only when the EEG already gives P(not wake) at least this")
    ap.add_argument("--hold", type=float, default=60.0, help="seconds between prompt and wake")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--refractory", type=float, default=120.0, help="seconds after a wake")
    ap.add_argument("--report-seconds", type=float, default=45.0)
    ap.add_argument("--task-minutes", type=float, default=3.0)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--self", action="store_true", help="investigator self-experiment, no REB code")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.source == "replay" and not a.replay:
        ap.error("--source replay needs --replay <PSG.edf>")
    if sys.platform == "win32" and a.source != "replay":
        # keep Windows from sleeping mid-nap (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    return Session(a).run()


if __name__ == "__main__":
    main()
