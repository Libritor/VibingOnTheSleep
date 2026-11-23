"""Vibing on the Sleep: the night. The sleeper directs, the agent codes.

Run after vibe/evening.py (the brief, the signal calibration, the open questions). Four ways in:

  1  SLEEP-ONSET DICTATION  --wakes N (default 1). When sleep onset is detected the rig whispers
     "Think about <the project>", lets --hold seconds pass, wakes the sleeper with a soft chime and
     asks what they were thinking. The spoken answer is transcribed here and becomes the agent's
     next instruction. Sleep onset = the validated detector's P(asleep), or, when the ear electrodes
     give it nothing (every night so far), a forehead rule that is NOT validated: slow activity
     risen 0.15 over the eyes-closed waking level for 20 of 30 s with body and eyes still. The log
     says which fired, and the sleeper is asked "were you asleep?" after each wake.
  2  YES/NO IN SLEEP  At each sleep-onset trigger, before the cue, the agent's pending yes/no
     question is whispered and the next --answer-seconds are decoded: eyes left-right-left-right
     = yes, two frowns = no (vibe/signals.py). No question pending -> a question with a known
     answer, so the channel's accuracy is measured anyway. --probe-rem also asks in detected REM.
     Every 10 s of the whole night is decoded too, so the rate of accidental "answers" is known.
  3  NATURAL AWAKENINGS  When the sleeper has surfaced on their own (waking signs in 12 of 30 s,
     including movement) at least --wake-gap minutes after the last exchange, or makes the eye
     signal while stirring, the rig whispers where things stand and the pending question, then
     listens --awake-seconds for a signal or a few words. At most --night-questions a night. It
     never wakes anyone for this.
  4  CUES IN DEEP SLEEP  The evening's unanswered design questions each have a sound; half
     (hidden) are replayed in slow-wave sleep by the same gate as the puzzle study
     (rig/sws_cue.py). The morning asks them all again.

The agent (vibe/agent.py) works in its own folder all night under fixed limits, starts a turn
whenever the sleeper gives it something, and otherwise carries on with the plan.

    python vibe/night.py --participant P-0001 --self                     # MuseLog streaming OSC to port 5000
    python vibe/night.py --participant P-0001 --self --source ble        # Muse 2 / Muse S on the laptop's Bluetooth
    python vibe/night.py --participant P-0001 --self --wakes 0           # nothing wakes you; ways 2-4 only
    Ctrl+C ends the night. Then: python vibe/morning.py --participant P-0001 --self

Output: study/participants/<P>/vibe/<session>/night/ (raw.csv, detector.csv, events.csv,
signals.csv, reports/, night.json).
"""
import argparse
import collections
import csv
import datetime as dt
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vibe"))
import common  # noqa: E402  (puts study/, rig/ and detector/ on the path)
from agent import Agent, StubAgent  # noqa: E402
from common import Ears, find_session, flush_keys, hold_awake, key_pressed, participant_dir, yes_no_words  # noqa: E402
from signals import Decoder  # noqa: E402
from voice import Voice  # noqa: E402

from model import LiveDetector  # noqa: E402
from sources import Buffer, MuseBLE, MuseOSC, Replay  # noqa: E402
from sws_cue import Cuer, DeepGate, level_check, split_channels  # noqa: E402

STAGES = ("W", "N1", "N2", "N3", "REM")
CATCH = [("Is two plus two four?", "yes"), ("Does a week have ten days?", "no"), ("Is ice cold?", "yes"),
         ("Is the sun cold?", "no"), ("Is water wet?", "yes"), ("Is three more than five?", "no")]
HOW = "Look left and right for yes. Frown for no."
HOW_SHORT = "Eyes for yes. Frown for no."
# the forehead rule is armed only if slow activity was this low while awake with the eyes closed: above it the
# forehead cannot tell waking from sleep that night (the same limit rig/sws_cue.py warns at)
FOREHEAD_MAX_AWAKE = 0.50


class MuseReplay:
    """One of this rig's own recordings (.npz from study/muselog_load.py or rawcsv_to_npz.py) played
    in simulated time, with its accelerometer: the night's logic on a real Muse night, real movement
    included. Sample times are the recording's own clock."""

    def __init__(self, buf, path, start_s=0.0):
        z = np.load(path, allow_pickle=True)
        self.t = np.asarray(z["t"], float)
        e = np.asarray(z["eeg"], float)
        self.eeg = e[:, :4]
        self.aux = e[:, 5] if e.shape[1] > 5 and np.isfinite(e[:, 5]).mean() > 0.5 and np.nanstd(e[:, 5]) > 0 else None
        acc = np.asarray(z["acc"], float) if "acc" in z.files and len(z["acc"]) else None
        self.acc_t, self.acc = (acc[:, 0], acc[:, 1:4]) if acc is not None and acc.shape[1] >= 4 else (None, None)
        self.start = self.t[0] + start_s
        self.i = int(np.searchsorted(self.t, self.start))
        self.buf = buf
        self.connected = True

    def advance_to(self, sim_t):
        j = int(np.searchsorted(self.t, self.start + sim_t))
        for a in range(self.i, j, 64):                   # the accelerometer four times a second
            b = min(a + 64, j)
            if self.acc is not None:
                k = min(len(self.acc) - 1, int(np.searchsorted(self.acc_t, self.t[a])))
                self.buf.accel = [float(v) for v in self.acc[k]]
            for n in range(a, b):
                if self.aux is not None:
                    self.buf.aux = float(self.aux[n])
                self.buf.add(self.t[n] - self.start, [float(v) for v in self.eeg[n]])
        self.i = j
        return j < len(self.t)


def forehead_onset(recent, wake_delta, rise, still):
    """The forehead rule for sleep onset (NOT validated against scored sleep): slow activity at least
    `rise` over the eyes-closed waking level in 4 of the last 6 five-second windows, the eyes still
    in those windows, and no body movement for a minute. recent: [(deep score, eye movements)]."""
    up = sum(1 for s, e in recent if np.isfinite(s) and s >= wake_delta + rise and (e is None or e <= 5))
    return len(recent) >= 5 and up >= 4 and still


def sign_kind(reason):
    if not reason:
        return None
    for key, kind in (("movement", "movement"), ("alpha", "alpha"), ("eye movements", "eyes"), ("muscle", "muscle"),
                      ("no usable", "off"), ("delta fell", "delta")):
        if reason.startswith(key):
            return kind
    return "other"


class Night:
    def __init__(self, a, agent=None, ears=None, inject=None):
        self.a = a
        self.replay = a.source == "replay"
        pdir = participant_dir(a.participant, a.self or self.replay)
        self.sdir = find_session(pdir, a.session)
        self.evening = json.loads((self.sdir / "evening.json").read_text(encoding="utf-8"))
        self.out = self.sdir / "night"
        k = 1
        while self.out.exists():                       # a second start the same night keeps the first
            k += 1
            self.out = self.sdir / f"night_{k}"
        (self.out / "reports").mkdir(parents=True)
        self.audio = not (self.replay or a.no_audio)
        self.inject = inject                            # tests: add a made signal to a response window
        self.t_start = time.time()
        self.sim_t = 0.0
        self.next_tick = None
        self.k = 0

        self.ev_f = open(self.out / "events.csv", "w", newline="", encoding="utf-8")
        self.ev_w = csv.writer(self.ev_f)
        self.ev_w.writerow(["t_s", "clock", "event", "detail"])
        self.det_f = open(self.out / "detector.csv", "w", newline="")
        self.det_w = csv.writer(self.det_f)
        self.det_w.writerow(["t_s", "clock", "phase", "contact_ok", *[f"p_{s}" for s in STAGES], "p_sleep", "p_deep",
                             "stage", "deep_score", "deep_thr", "gate", "eye", "n_forehead", "cues"])
        self.sig_f = open(self.out / "signals.csv", "w", newline="")
        self.sig_w = csv.writer(self.sig_f)
        self.sig_w.writerow(["t_s", "clock", "phase", "context", "stage", "answer", "why", "eyes", "muscle",
                             "eye_amp", "emg_amp", "move_mg", "n_forehead"])
        self.raw_f = None

        self.voice = Voice(a.voice, enabled=self.audio, log=self.event)
        self.ears = ears or Ears(enabled=self.audio, stt_model=a.stt_model)
        self.dec = Decoder.load(self.sdir / "signals.json")
        sig = json.loads((self.sdir / "signals.json").read_text()) if (self.sdir / "signals.json").exists() else {}
        self.probes_on = not a.no_probes
        if self.probes_on and not self.replay and not a.force_probes:
            if not sig:
                self.event("probes_note", "no signal calibration tonight: default thresholds")
            elif sig.get("verify_n") and sig["verify_right"] / sig["verify_n"] < 0.6:
                self.event("probes_off", f"the calibration check read {sig['verify_right']} of {sig['verify_n']}: "
                                         "yes/no questions are off tonight (--force-probes overrides)")
                self.probes_on = False
        if agent is not None:
            self.agent = agent
        else:
            cls = StubAgent if a.agent == "stub" else Agent
            self.agent = cls.load(self.sdir / "agent_state.json", self.sdir / "agent_turns.jsonl", event=self.event,
                                  **{k: v for k, v in dict(model=a.agent_model, budget_usd=a.agent_budget,
                                                           max_turns=a.agent_turns).items() if v is not None})
        self.agent.event = self.event
        self.one_liner = (self.evening.get("plan") or {}).get("one_liner") or "your project"

        import joblib
        self.det = LiveDetector(joblib.load(a.model), step_s=5.0)
        self.buf = Buffer(seconds=120)
        self.p = None
        self.stage = ""
        self.g = None
        self.fh = collections.deque(maxlen=24)           # forehead (t, score, eye) per 5-s window
        self.signs = collections.deque(maxlen=120)       # (t, kind of waking sign or None) per second
        self.rem_hold = 0
        self.quiet_until = 0.0
        self.last_interaction = 0.0
        self.last_probe = -1e9
        self.phase = "start"
        self.rounds = 0
        self.n_probes = 0
        self.n_awake = 0
        self.n_rem_probes = 0
        self.probes, self.triggers, self.awakenings, self.reports = [], [], [], []
        self.spont = collections.Counter()
        self.stream_ok = True
        self.last_stream = time.time()
        self.gain_cue = self.gain_night = self.gain_wake = 0.0
        self.threshold = None
        self.cuer = None
        self.gate = None
        self.cue_ids = []
        self.pending_call = False                        # the sleeper made the eye signal while stirring
        self.summary = None

    # ------------------------------------------------------------------ time and logs
    def now(self):
        return self.sim_t if self.replay else time.time() - self.t_start

    def clock(self, t=None):
        return dt.datetime.fromtimestamp(self.t_start + (self.now() if t is None else t)).strftime("%H:%M:%S")

    def event(self, name, detail="", show=None):
        t = self.now()
        self.ev_w.writerow([f"{t:.1f}", self.clock(t), name, detail])
        self.ev_f.flush()
        if not self.a.quiet:
            txt = detail if show is None else show
            print(f"[{self.clock(t)}] {name}" + (f": {txt}" if txt else ""), flush=True)

    def advance(self):
        if self.replay:
            self.sim_t += 1.0
            if not self.src.advance_to(self.sim_t):
                raise EOFError
            return
        if self.next_tick is None:
            self.next_tick = time.time()
        self.next_tick += 1.0
        delay = self.next_tick - time.time()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_tick = time.time()

    # ------------------------------------------------------------------ sensing, once a second
    def _chans(self, seconds):
        r = self.buf.raw(seconds)
        if r is None:
            return None, None
        rows, aux, acc = r
        mag = np.linalg.norm(acc, axis=1)
        move = float(np.nanstd(mag) * 1000) if np.isfinite(mag).sum() > 10 else None
        return split_channels(rows, aux if np.isfinite(aux).all() else None), move

    def sense(self):
        """One second: read the stream, look for waking signs, cue if the gate says so, and every
        5 s stage the last 30 s; every 10 s decode the last 10 s for accidental signals."""
        self.advance()
        t = self.now()
        self.k += 1
        if not self.replay:
            alive = self.buf.last is not None and time.time() - self.buf.last < 5
            if alive:
                self.last_stream = time.time()
            if alive != self.stream_ok:
                self.event("stream_regained" if alive else "stream_lost")
                self.stream_ok = alive
            if not alive:
                self.agent.poll()
                if time.time() - self.last_stream > self.a.silence_minutes * 60:
                    self.event("end_after_silence")
                    raise EOFError
                return None
        reason = None
        chans, move = self._chans(10)
        if chans is not None:
            was = self.gate.state
            reason = self.gate.fast(t, chans, move)
            if reason and was == "cueing":
                self.event("cueing_paused", reason)
            if self.cuer and t >= self.quiet_until and not self.voice.speaking():
                self.cuer.tick(t, self.gate.state == "cueing", reason)
        self.signs.append((t, sign_kind(reason)))
        if self.k % 5 == 0:
            self.epoch(t)
        if self.k % 10 == 0 and t >= self.quiet_until:
            self.monitor(t)
        self.agent.poll()
        return reason

    def epoch(self, t):
        w = self.buf.window(30)
        ok = bool(w and w[3])
        p = self.det.push(w[0], w[1], w[2]) if (w and ok) else None
        self.p = p
        ps = pdp = None
        if p:
            ps, pdp = 1 - p["W"], p["N2"] + p["N3"]
            self.stage = max(STAGES, key=lambda s: p[s])
            self.rem_hold = self.rem_hold + 1 if p["REM"] >= self.a.rem_p else 0
        else:
            self.stage, self.rem_hold = "", 0
        chans, _ = self._chans(30)
        g = None
        if chans is not None:
            was = self.gate.state
            g = self.gate.epoch(t, chans)
            if g["state"] != was and (g["state"] == "cueing" or was == "cueing") and self.cuer:
                self.event("cueing_on" if g["state"] == "cueing" else "cueing_paused",
                           f"deep score {g['score']:.2f} (threshold {g['thr']:.2f})")
            self.fh.append((t, g["score"], g["eye"]))
        self.g = g
        self.det_w.writerow([f"{t:.1f}", self.clock(t), self.phase, int(ok)]
                            + ([f"{p[s]:.4f}" for s in STAGES] + [f"{ps:.4f}", f"{pdp:.4f}"] if p else [""] * 7)
                            + [self.stage]
                            + ([("" if not np.isfinite(g["score"]) else f"{g['score']:.3f}"), f"{g['thr']:.2f}",
                                g["state"], "" if g["eye"] is None else g["eye"], g["n_good"]] if g else [""] * 5)
                            + [self.cuer.n if self.cuer else 0])
        if int(t) % 60 < 5:
            self.det_f.flush()
            self.sig_f.flush()
            if not self.replay and not self.a.quiet:
                bits = [f"{self.phase}"]
                if p:
                    bits.append(f"{self.stage} P(sleep) {ps:.2f}")
                if g and np.isfinite(g["score"]):
                    bits.append(f"forehead {g['score']:.2f}/{g['thr']:.2f} {g['state']}")
                bits.append(f"agent turn {self.agent.turns}" + (" (working)" if self.agent.busy else ""))
                print(f"[{self.clock(t)}] " + " | ".join(bits), flush=True)

    def _decode(self, seconds, context):
        r = self.buf.raw(seconds)
        if r is None:
            return dict(answer=None, why="no data", eyes=0, muscle=0, eye_amp=0, emg_amp=0, move_mg=None, n_forehead=0)
        rows, _, acc = r
        if self.inject is not None:
            rows = self.inject(context, self.now(), rows, self.buf.fs)
        m = self.dec.decode(rows, self.buf.fs, acc if np.isfinite(acc).sum() > 30 else None)
        t = self.now()
        self.sig_w.writerow([f"{t:.1f}", self.clock(t), self.phase, context, self.stage, m["answer"] or "", m["why"],
                             m["eyes"], m["muscle"], f"{m['eye_amp']:.1f}", f"{m['emg_amp']:.2f}",
                             "" if m["move_mg"] is None else f"{m['move_mg']:.1f}", m["n_forehead"]])
        return m

    def monitor(self, t):
        """Nobody was asked anything: whatever the decoder reads here is an accidental signal,
        unless the sleeper is stirring and made the eye signal to be spoken to (way 3)."""
        m = self._decode(10, "unasked")
        self.spont[m["answer"] or ("off" if m["n_forehead"] < 2 else "quiet")] += 1
        if (m["answer"] == "yes" and self.phase == "night" and self.n_awake < self.a.night_questions
                and t - self.last_interaction >= 300 and self.waking(t, 30)[0] >= 3):
            self.pending_call = True

    def waking(self, t, seconds):
        recent = [k for ts, k in self.signs if t - ts <= seconds]
        return (sum(k in ("movement", "alpha", "eyes", "muscle") for k in recent), sum(k == "movement" for k in recent))

    def safely(self, what, fn, *args):
        """An exchange that fails (a sound device, the transcriber) is logged and the night goes on."""
        try:
            return fn(*args)
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as e:
            self.event("exchange_failed", f"{what}: {type(e).__name__}: {str(e)[:160]}")
            self.last_interaction = self.now()
            return None

    def run_for(self, seconds, guard=None):
        for _ in range(int(math.ceil(seconds))):
            self.sense()
            if guard and guard():
                return True
        return False

    # ------------------------------------------------------------------ speaking and listening
    def say(self, text, gain, style="night"):
        self.event("say", text)
        self.quiet_until = self.now() + 0.4 + 0.38 * len(text.split()) + 1.5
        dur = self.voice.say(text, style, gain, block=True)
        if self.replay:
            self.run_for(dur)
        self.quiet_until = self.now() + 1.5
        return dur

    def listen(self, seconds, context, record=False):
        rec = self.ears.start(seconds) if record else None
        self.quiet_until = self.now() + seconds + 2
        self.run_for(seconds)
        return self._decode(seconds, context), rec

    # ------------------------------------------------------------------ way 2: a yes/no question in sleep
    def probe(self, context):
        q, catch = self.agent.question, None
        if not q:
            catch = CATCH[self.n_probes % len(CATCH)]
            q = catch[0]
        self.n_probes += 1
        self.last_probe = self.now()
        self.event("question", f"{context}: {q}")
        self.say(f"{q} {HOW if self.n_probes == 1 else HOW_SHORT}", self.gain_night)
        m, _ = self.listen(self.a.answer_seconds, f"asked:{context}")
        ans = m["answer"] if m["answer"] in ("yes", "no") else None
        rec = dict(t_s=round(self.now(), 1), clock=self.clock(), context=context, stage=self.stage, question=q,
                   known_answer=catch[1] if catch else None, answer=ans, read=m["answer"], why=m["why"],
                   eyes=m["eyes"], muscle=m["muscle"])
        self.probes.append(rec)
        if ans:
            self.event("answer", f"{ans} ({m['why']})" + (f", known answer {catch[1]}" if catch else ""))
            if not catch:
                self.agent.answer(ans == "yes")
            self.say("Yes." if ans == "yes" else "No.", self.gain_night)       # what was read, so a slip is heard
        else:
            self.event("no_answer", m["why"])
            if not catch:
                self._unanswered(q)
        return rec

    def _unanswered(self, q):
        self.agent.question_asked += 1
        if self.agent.question_asked >= self.a.ask_times:
            self.agent.submit("answer", f"NO ANSWER after {self.agent.question_asked} asks, so the answer is no: {q}")
            self.agent.question = None

    # ------------------------------------------------------------------ way 1: sleep-onset dictation
    def onset_trigger(self, t):
        p = self.p
        if p is not None and 1 - p["W"] >= self.a.p_sleep:
            return "eeg", f"validated detector, P(asleep) {1 - p['W']:.2f}"
        g = self.gate
        if self.a.no_forehead_onset or len(g.base) < 6 or not np.isfinite(g.wake_delta) or g.wake_delta > FOREHEAD_MAX_AWAKE:
            return None
        recent = [(s, e) for ts, s, e in self.fh if t - ts <= 30]
        if forehead_onset(recent, g.wake_delta, self.a.onset_rise, self.waking(t, 60)[1] == 0):
            return "forehead", (f"forehead rule (not validated), slow activity {recent[-1][0]:.2f} against "
                                f"{g.wake_delta:.2f} awake")
        return None

    def n2_guard(self):
        p = self.p
        if p is not None and p["N2"] + p["N3"] >= self.a.p_n2:
            return True
        g, gate = self.g, self.gate
        return bool(g and np.isfinite(g["score"]) and np.isfinite(gate.wake_delta) and gate.wake_delta <= FOREHEAD_MAX_AWAKE
                    and g["score"] >= max(0.60, gate.wake_delta + 0.35))

    def onset_round(self, kind, why):
        self.rounds += 1
        self.event("sleep_onset", why)
        trig = dict(round=self.rounds, t_s=round(self.now(), 1), clock=self.clock(), by=kind, detail=why)
        self.triggers.append(trig)
        if self.probes_on:
            self.probe("sleep onset")
        self.say(f"Think about {self.one_liner}.", self.gain_night)
        if self.run_for(self.a.hold, guard=self.n2_guard):
            self.event("deeper_than_planned", "N2 signs before the hold ended: waking now")
            trig["overshoot"] = True
        self.wake_and_report(trig)

    def wake_and_report(self, trig):
        n = self.rounds
        self.event("wake", f"round {n}")
        if self.audio:
            self.voice.chime(self.gain_wake)
        if self.agent.summary and not self.agent.summary_spoken:
            self.say(self.agent.summary, self.gain_wake)
            self.agent.summary_spoken = True
        self.say("What were you just thinking? Tell me what to build next.", self.gain_wake)
        wav, secs = self.ears.record(self.out / "reports" / f"report_{n}.wav", self.a.report_seconds)
        if self.replay:
            self.run_for(20)
        self.say("Were you asleep just now? Say yes or no.", self.gain_wake)
        rec = self.ears.start(4.0)
        if self.audio:
            time.sleep(4.2)
        wav2 = self.ears.finish(rec, self.out / "reports" / f"asleep_{n}.wav")
        last = n >= self.a.wakes
        self.say("Thank you. Good night. I will keep working." if last else
                 "Thank you. Close your eyes. I will work on that.", self.gain_night)
        # the sleeper is on the way back to sleep; the words are turned into text now
        text = self.ears.transcribe(wav)
        if wav is None:
            (self.out / "reports" / f"report_{n}.txt").write_text(text, encoding="utf-8")
        words = len(text.split())
        self.event("report", f"{words} words in {secs:.0f} s")
        self.reports.append(dict(round=n, clock=self.clock(), words=words, text=text))
        if words >= 2:
            self.agent.submit("dictation", text)
        said = self.ears.transcribe(wav2)
        trig["said_asleep"] = yes_no_words(said)
        self.event("asleep_rating", trig["said_asleep"] or f"unclear ({said[:30]})")
        self.last_interaction = self.now()

    # ------------------------------------------------------------------ way 3: a natural awakening
    def awakening(self, why):
        self.n_awake += 1
        self.pending_call = False
        self.event("awakening", why)
        parts = []
        if self.agent.summary and not self.agent.summary_spoken:
            parts.append(self.agent.summary)
            self.agent.summary_spoken = True
        q = self.agent.question
        parts.append(f"{q} {HOW_SHORT} Or say it." if q else "Anything to change? Say it, or sleep on.")
        self.say(" ".join(parts), self.gain_night)
        m, rec = self.listen(self.a.awake_seconds, "asked:awakening", record=True)
        text = self.ears.transcribe(self.ears.finish(rec, self.out / "reports" / f"awake_{self.n_awake}.wav"))
        ans = m["answer"] if m["answer"] in ("yes", "no") else yes_no_words(text)
        rec = dict(t_s=round(self.now(), 1), clock=self.clock(), why=why, question=q, answer=ans if q else None,
                   signal=m["answer"], words=len(text.split()), text=text)
        self.awakenings.append(rec)
        if q and ans:
            self.event("answer", f"{ans} at an awakening ({'signal: ' + m['why'] if m['answer'] in ('yes', 'no') else 'spoken'})")
            self.agent.answer(ans == "yes")
        elif len(text.split()) >= 3:
            self.event("spoken_instruction", f"{len(text.split())} words")
            self.agent.submit("spoken", text)
        elif q:
            self.event("no_answer", "nothing at the awakening")
            self._unanswered(q)
        self.say("Good night.", self.gain_night)
        self.last_interaction = self.now()

    # ------------------------------------------------------------------ setting up
    def connect(self):
        a = self.a
        own = self.replay and str(a.replay).lower().endswith(".npz")
        if self.replay:
            self.src = (MuseReplay if own else Replay)(self.buf, a.replay, start_s=a.replay_start_hours * 3600.0)
            self.event("replay", Path(a.replay).name)
        else:
            self.raw_f = open(self.out / "raw.csv", "w", newline="")
            raw_w = csv.writer(self.raw_f)
            raw_w.writerow(["t", "TP9", "AF7", "AF8", "TP10", "AUX", "ax", "ay", "az"])
            n = [0]

            def sink(t, row, acc):
                acc = acc or ["", "", ""]
                aux = self.buf.aux
                raw_w.writerow([f"{t:.4f}", *[f"{v:.2f}" for v in row], "" if aux is None else f"{aux:.2f}", *acc])
                n[0] += 1
                if n[0] % 2560 == 0:
                    self.raw_f.flush()
            self.buf.sink = sink
            if a.source == "ble":
                self.src = MuseBLE(self.buf, a.muse_name, log=lambda m: self.event("muse", m))
                print("Looking for the Muse on this computer's Bluetooth...", flush=True)
            else:
                self.src = MuseOSC(self.buf, a.osc_port, log=lambda m: self.event("muse", m))
                print(f"Waiting for MuseLog on UDP {a.osc_port} (SnowballArcade format, Full-rate raw EEG on)...", flush=True)
            self.src.start()
            while self.buf.last is None or time.time() - self.buf.last > 2:
                time.sleep(0.5)
            self.event("headband_streaming")
            while True:
                n0 = self.buf.count
                time.sleep(5)
                rate = (self.buf.count - n0) / 5
                self.event("eeg_rate", f"{rate:.0f} Hz")
                if rate >= 200:
                    break
                print("EEG under 200 Hz: turn on Full-rate raw EEG in MuseLog's OSC settings; waiting...", flush=True)
            self.event("accelerometer", "present" if self.buf.accel is not None else "absent (movement is read from the electrodes only)")
        gate_kw = dict(min_minutes=a.cue_min_minutes, max_sd=200.0 if self.replay and not own else 60.0)
        self.gate = DeepGate(self.buf.fs, **gate_kw)

    def levels(self):
        """The bedtime hearing check sets every level of the night; then one sentence in the night
        voice, louder on request, so that the whisper can be understood and no louder."""
        a = self.a
        if not self.audio:
            return
        import sounddevice as sd

        def play(y):
            sd.play(y, 44100)
            sd.wait()
        if a.cue_gain is not None:
            self.gain_cue = a.cue_gain
        else:
            print("\nHEARING CHECK. Lie as you will sleep, with the speaker or earbuds where they will stay all "
                  "night, and the keyboard in reach. Do not touch the Windows volume after this. A soft hiss "
                  "plays every two seconds, getting louder: press any key as soon as you hear it. It runs twice.",
                  flush=True)
            input("Press Enter to start the hearing check... ")
            self.threshold = level_check(play, key_pressed, flush_keys, log=lambda m: print("  " + m, flush=True))
            if self.threshold is None:
                raise SystemExit("Nothing was heard up to full volume: check the speaker and the Windows volume.")
            self.gain_cue = min(0.0, self.threshold + a.cue_above)
        base = self.threshold if self.threshold is not None else self.gain_cue - a.cue_above
        self.gain_night = min(0.0, base + a.voice_above)
        self.gain_wake = min(0.0, base + a.wake_above)
        if a.cue_gain is None:
            while True:
                self.voice.say("This is how I will sound in the night. Can you make out every word?", "night", self.gain_night)
                flush_keys()
                r = input("Enter if every word was clear, L for louder, Q for quieter: ").strip().lower()
                if r == "l":
                    self.gain_night = min(0.0, self.gain_night + 3)
                elif r == "q":
                    self.gain_night -= 3
                else:
                    break
            self.gain_wake = max(self.gain_wake, min(0.0, self.gain_night + 6))
        self.event("levels", f"hearing threshold {self.threshold}, cues {self.gain_cue:.0f} dB, night voice "
                             f"{self.gain_night:.0f} dB, wake voice {self.gain_wake:.0f} dB")

    def load_cues(self):
        qdir = self.sdir / "questions"
        if self.a.no_cues or not (qdir / "split.json").exists():
            return
        from puzzle_sounds import make
        man = json.loads((qdir / "manifest.json").read_text(encoding="utf-8"))
        split = json.loads((qdir / "split.json").read_text())
        self.cue_ids = list(split["cued"])
        if not self.cue_ids:
            return
        sounds = {q: make(man["sounds"][q]) for q in self.cue_ids}
        if self.audio:
            import sounddevice as sd
            play = lambda y: sd.play(y, 44100)                 # noqa: E731
        else:
            play = lambda y: None                              # noqa: E731
        floor = self.threshold if self.threshold is not None else self.gain_cue - 12.0
        self.cuer = Cuer(sounds, self.gain_cue, floor, play,
                         lambda name, detail, show=None: self.event(name, detail, show=show),
                         self.a.cue_interval, 2.0, seed=0 if self.replay else int(time.time()))
        self.event("cues_ready", f"{len(sounds)} of tonight's open questions will be cued in deep sleep (which ones is hidden)")

    # ------------------------------------------------------------------ the night
    def run(self):
        a = self.a
        self.connect()
        if self.audio:
            print("Getting the night voice ready...", flush=True)
            self.voice.prerender(["Yes.", "No.", "Good night.", "What were you just thinking? Tell me what to build next.",
                                  "Were you asleep just now? Say yes or no.", "Thank you. Good night. I will keep working.",
                                  "Thank you. Close your eyes. I will work on that.", f"Think about {self.one_liner}.",
                                  "Anything to change? Say it, or sleep on.",
                                  "This is how I will sound in the night. Can you make out every word?",
                                  "Lie back and close your eyes. Stay awake and still for three minutes. "
                                  "Then sleep whenever you like."], "night")
        self.levels()
        self.load_cues()
        if not self.replay:
            hold_awake(True)
            self.t_start = time.time()
            print(f"\nThe night starts now. {a.wakes} wake(s) at sleep onset, then none. Ctrl+C ends it.\n", flush=True)
            self.voice.say("Lie back and close your eyes. Stay awake and still for three minutes. "
                           "Then sleep whenever you like.", "night", self.gain_wake)
        self.event("night_start", f"session {self.sdir.name}, workspace {self.agent.ws}")
        self.phase = "calib"
        onset_from = None
        try:
            while self.now() < (a.replay_hours if self.replay else a.max_hours) * 3600.0:
                self.sense()
                t = self.now()
                if self.phase == "calib":
                    if t >= a.calib_seconds and (self.det.calibrated or t >= a.calib_seconds + 90):
                        self.phase = "onset" if a.wakes > 0 else "night"
                        onset_from = self.last_interaction = t
                        fore = self.gate.wake_delta
                        self.event("calibrated", f"validated detector {'ready' if self.det.calibrated else 'NOT ready (it needs the ear electrodes on skin)'}; "
                                                 f"forehead awake level {fore:.2f}" + (
                                       "" if np.isfinite(fore) and fore <= FOREHEAD_MAX_AWAKE else " (too high to use the forehead rule tonight)"))
                elif self.phase == "onset":
                    if self.k % 5 == 0 and t - self.last_interaction >= a.refractory:
                        hit = self.onset_trigger(t)
                        if hit:
                            self.safely("sleep-onset round", self.onset_round, *hit)
                            if self.rounds >= a.wakes:
                                self.phase = "night"
                        elif t - onset_from > a.onset_max_minutes * 60:
                            self.event("no_sleep_onset_seen", f"nothing in {a.onset_max_minutes:.0f} min: no wake tonight")
                            self.phase = "night"
                            self.last_interaction = t
                else:
                    if self.n_awake < a.night_questions and (
                            self.pending_call or (t - self.last_interaction >= a.wake_gap * 60
                                                  and self.waking(t, 30)[0] >= 12 and self.waking(t, 30)[1] >= 2)):
                        self.safely("awakening", self.awakening,
                                    "eye signal while stirring" if self.pending_call else "awake on their own")
                    elif (a.probe_rem and self.probes_on and self.rem_hold >= 6 and self.n_rem_probes < a.rem_probes
                          and t - self.last_probe >= 1200 and t - self.last_interaction >= 600):
                        self.n_rem_probes += 1
                        self.safely("REM question", self.probe, "REM")
        except KeyboardInterrupt:
            self.event("stopped_by_hand")
        except EOFError:
            self.event("stream_ended")
        finally:
            self.close()
        return self.out

    def close(self):
        if not self.replay and self.agent.busy:
            print("Waiting for the agent to finish the step it is on (Ctrl+C to stop waiting)...", flush=True)
            try:
                self.agent.wait(900)
            except KeyboardInterrupt:
                pass
        self.agent.autonomous = False
        asked = [p for p in self.probes if p["known_answer"]]
        quiet = self.spont["quiet"] + self.spont["yes"] + self.spont["no"] + self.spont["ambiguous"]
        summary = dict(
            session=self.sdir.name, ended=dt.datetime.now().isoformat(timespec="seconds"),
            hours=round(self.now() / 3600, 2), wakes=self.rounds, sleep_onsets=self.triggers,
            reports=[dict(round=r["round"], clock=r["clock"], words=r["words"]) for r in self.reports],
            questions_in_sleep=self.probes,
            known_answer_questions=dict(asked=len(asked), answered=sum(p["answer"] is not None for p in asked),
                                        right=sum(p["answer"] == p["known_answer"] for p in asked)),
            awakenings=[{k: v for k, v in w.items() if k != "text"} for w in self.awakenings],
            unasked_windows=dict(usable=quiet, yes=self.spont["yes"], no=self.spont["no"],
                                 both=self.spont["ambiguous"], forehead_off=self.spont["off"],
                                 accidental_yes_pct=round(100 * self.spont["yes"] / max(quiet, 1), 2),
                                 accidental_no_pct=round(100 * self.spont["no"] / max(quiet, 1), 2)),
            cues=dict(total=self.cuer.n if self.cuer else 0, per_question=dict(self.cuer.counts) if self.cuer else {},
                      final_gain_db=self.cuer.gain_db if self.cuer else None),
            levels=dict(hearing_threshold_db=self.threshold, cue_db=self.gain_cue, night_voice_db=self.gain_night,
                        wake_voice_db=self.gain_wake),
            agent=dict(turns=self.agent.turns, cost_usd=round(self.agent.cost, 2), done=self.agent.done,
                       workspace=str(self.agent.ws)),
            decoder=self.dec.p)
        (self.out / "night.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        self.agent.save(self.sdir / "agent_state.json")
        self.event("night_end", f"{self.rounds} wake(s), {len(self.probes)} question(s) in sleep, "
                                f"{len(self.awakenings)} at awakenings, {summary['cues']['total']} cues, "
                                f"agent {self.agent.turns} turns")
        for f in (self.ev_f, self.det_f, self.sig_f, self.raw_f):
            if f:
                f.close()
        if not self.replay:
            hold_awake(False)
            print(f"\nwritten {self.out}\nIn the morning: python vibe/morning.py --participant {self.a.participant} --self")
        self.summary = summary


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--self", action="store_true", help="investigator self-experiment (no REB code)")
    ap.add_argument("--session", default="latest", help="evening session folder (default: the newest)")
    ap.add_argument("--source", choices=("osc", "ble", "replay"), default="osc")
    ap.add_argument("--osc-port", type=int, default=5000)
    ap.add_argument("--muse-name", default="")
    ap.add_argument("--replay", help="for --source replay: a Sleep-EDF PSG file, or one of this rig's own "
                                     "recordings as .npz (a real Muse night, with its accelerometer)")
    ap.add_argument("--replay-start-hours", type=float, default=0.0)
    ap.add_argument("--replay-hours", type=float, default=1.0)
    ap.add_argument("--model", default=str(ROOT / "models" / "nap_base3min.joblib"))
    # way 1
    ap.add_argument("--wakes", type=int, default=1, help="sleep-onset wakes for dictation (0 = never wake)")
    ap.add_argument("--hold", type=float, default=60.0, help="seconds between the cue and the wake")
    ap.add_argument("--report-seconds", type=float, default=45.0, help="longest spoken report")
    ap.add_argument("--refractory", type=float, default=120.0, help="seconds after an exchange before the next trigger")
    ap.add_argument("--onset-max-minutes", type=float, default=45.0, help="give up on a wake after this long without sleep onset")
    ap.add_argument("--p-sleep", type=float, default=0.7)
    ap.add_argument("--p-n2", type=float, default=0.7)
    ap.add_argument("--onset-rise", type=float, default=0.15, help="forehead rule: rise of slow activity over the awake level")
    ap.add_argument("--no-forehead-onset", action="store_true", help="only the validated detector may trigger")
    ap.add_argument("--calib-seconds", type=float, default=180.0)
    # way 2
    ap.add_argument("--no-probes", action="store_true", help="no yes/no questions in sleep")
    ap.add_argument("--force-probes", action="store_true", help="ask even if the evening signal check went badly")
    ap.add_argument("--answer-seconds", type=float, default=10.0)
    ap.add_argument("--ask-times", type=int, default=2, help="asks without an answer before the answer is taken as no")
    ap.add_argument("--probe-rem", action="store_true", help="also ask in detected REM")
    ap.add_argument("--rem-p", type=float, default=0.8)
    ap.add_argument("--rem-probes", type=int, default=3)
    # way 3
    ap.add_argument("--night-questions", type=int, default=2, help="most whispered exchanges at awakenings")
    ap.add_argument("--wake-gap", type=float, default=45.0, help="minutes since the last exchange before another")
    ap.add_argument("--awake-seconds", type=float, default=12.0)
    # way 4
    ap.add_argument("--no-cues", action="store_true")
    ap.add_argument("--cue-gain", type=float, help="cue gain in dB, skipping the hearing check")
    ap.add_argument("--cue-above", type=float, default=8.0, help="cues: dB above the hearing threshold")
    ap.add_argument("--cue-interval", type=float, default=12.0)
    ap.add_argument("--cue-min-minutes", type=float, default=20.0)
    # voice
    ap.add_argument("--voice", default="kokoro", choices=("kokoro", "edge", "sapi"))
    ap.add_argument("--voice-above", type=float, default=15.0, help="night voice: dB above the hearing threshold")
    ap.add_argument("--wake-above", type=float, default=24.0, help="wake voice and chime: dB above the threshold")
    ap.add_argument("--stt-model", default="base")
    ap.add_argument("--no-audio", action="store_true")
    # agent
    ap.add_argument("--agent", choices=("claude", "stub"), default="claude")
    ap.add_argument("--agent-model")
    ap.add_argument("--agent-budget", type=float, help="dollars per turn")
    ap.add_argument("--agent-turns", type=int, help="most turns in the night")
    ap.add_argument("--max-hours", type=float, default=10.0)
    ap.add_argument("--silence-minutes", type=float, default=15.0)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if a.source == "replay" and not a.replay:
        sys.exit("--source replay needs --replay <PSG file>")
    Night(a).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
