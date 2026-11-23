"""Cue puzzles in slow-wave sleep: the night half of the puzzle-cueing study.

The validated stage detector needs the mastoid (ear) electrodes, which carried mains noise all
of the first night. Slow-wave sleep showed clearly on the forehead electrodes against the
headband's own Fpz reference instead (both headbands agreed, r 0.64), so this gate reads only
AF7, AF8 and the fifth electrode (an Oz cup), each on its own:

  every 5 s, on the last 30 s:  per channel, usable = not railed, 0.8-60 uV sd after a 60 Hz
      notch and a 0.3 Hz high-pass (the night_channels.py rule), and broadband 20-45 Hz power under
      100 uV^2 (an electrode that has come off but not railed, like the Oz cup after the first
      night's turn, carries ~1500); the deep score is the median relative delta (0.5-4 Hz over
      0.5-30 Hz) of the usable channels. Narrow lines above 4 Hz (the Muse 2's 21.33 Hz packet line
      on AF7 and its 10.67 Hz subharmonic, mains residue, the 24-48 Hz comb) are removed with a
      2-Hz running median across frequency before any band power. A window counts as deep
      only if the score is at or above the threshold AND AF7-AF8 (the horizontal eye channel) shows
      at most 5 eye movements: waking with the eyes moving also puts power below 4 Hz on the
      forehead. On Sleep-EDF the count is 0 in N3 and 7-8 in waking; on this user's Muse 2 its
      95th percentile in slow-wave sleep was 5. And the slow waves must be large: per channel, the
      sd of the 0.5-2 Hz band against the 20th percentile of the same over the night's still
      windows so far (eyes and body still, after the calibration minutes; light sleep and quiet
      waking make up that lower end), median ratio over channels at least 1.8 to start, 1.6 to
      continue. Until a channel has 2 minutes of still windows it gives no ratio, and with no ratio
      nothing is cued. Relative delta alone let REM, N1 and N2 through on Sleep-EDF (60% of cues in
      N3 over 20 nights); N3 slow waves are about twice the amplitude of N2's. The reference is a
      percentile, not a delta band, because on some montages and people every window has high
      relative delta.
  cueing starts when, after --min-minutes of the night, 80% of the windows of the last two minutes
      were deep, the current one is, and nothing in those two minutes looked like waking; it
      continues while the score stays above threshold - 0.05 and the eyes stay still. After a pause
      for a dip (no waking sign), it resumes once the last 60 s are deep again; after a waking sign
      it needs the full two minutes
  every 1 s, on the last 10 s, a waking sign pauses cueing at once: movement (sd of |acc| above
      --move-mg milli-g), no usable forehead channel, alpha (8-12 Hz) above the eyes-closed wake
      level, 6 or more eye movements, muscle (each channel's 20-45 Hz power against its own median
      over the last 10 minutes of deep windows, so a turn that changes an electrode's contact
      re-baselines it; the smallest ratio over 4, so one noisy electrode cannot trip it), or,
      while cueing, delta falling below 0.40; a
      waking sign within 10 s of a cue also lowers the volume 3 dB
  the threshold is max(--deep-thr, wake delta + 0.30), capped at 0.85, the wake level coming from
      the first three minutes of still windows (no movement, at most 5 eye movements) within the
      first 15 minutes of the night (eyes closed, awake); a wake delta above 0.5 means the
      forehead cannot tell waking from deep sleep tonight (eye movements, loose electrodes) and is
      logged as a warning

Cues cycle through the cued puzzles in a fresh random order each round, one every 10-14 s. The
uncued puzzles' sounds are never played. Nothing printed says which puzzle a cue belongs to.
"""
import collections
import csv
import datetime as dt
import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch

CHANNELS = ("AF7", "AF8", "Oz")


def clean(x, fs):
    """60 Hz (and 120 Hz) notch, 0.3 Hz high-pass: the same filter as study/night_channels.py."""
    for f0 in (60.0, 120.0):
        if f0 < fs / 2:
            b, a = iirnotch(f0, 30.0, fs)
            x = filtfilt(b, a, x)
    sos = butter(2, 0.3, btype="high", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def eye_count(e, fs):
    """Steep 1-5 Hz deflections above 5 MAD of the slope: the rapid-eye-movement count of the
    detector and of study/night_channels.py."""
    rem = sosfiltfilt(butter(2, [1.0, 5.0], btype="band", fs=fs, output="sos"), e)
    d = np.diff(rem) * fs
    thr = 5 * np.median(np.abs(d)) / 0.6745 + 1e-9
    return int(np.sum((np.abs(d[1:]) > thr) & (np.abs(d[:-1]) <= thr)))


_SW = {}


def slow_wave_sd(y, fs):
    """sd of the 0.5-2 Hz band: the size of the slow waves."""
    if fs not in _SW:
        _SW[fs] = butter(2, [0.5, 2.0], btype="band", fs=fs, output="sos")
    return float(sosfiltfilt(_SW[fs], y).std())


def channel_stats(x, fs, max_sd=60.0, hf_max=100.0):
    """Usability and band powers of one raw channel (uV, with the Muse's ~800 uV offset)."""
    if not np.isfinite(x).all() or ((x < 5) | (x > 1640)).mean() >= 0.01:
        return None
    y = clean(x - x.mean(), fs)
    if not 0.8 < y.std() < max_sd:
        return None
    nper = int(min(4 * fs, len(y)))
    f, P = welch(y, fs=fs, nperseg=nper, noverlap=nper // 2)
    df = f[1] - f[0]
    above = f >= 4
    P[above] = np.minimum(P[above], median_filter(P, size=max(3, int(round(2.0 / df)) | 1), mode="nearest")[above])
    hfb = (f >= 20) & (f <= min(45, fs / 2 - 1))
    hf = float(np.median(P[hfb]) * (min(45, fs / 2 - 1) - 20))       # broadband muscle/contact noise, uV^2
    if hf > hf_max:
        return None
    tot = P[(f >= 0.5) & (f <= 30)].sum() * df + 1e-12
    return dict(rel_delta=P[(f >= 0.5) & (f < 4)].sum() * df / tot,
                rel_alpha=P[(f >= 8) & (f < 12)].sum() * df / tot,
                hf=hf, sw=slow_wave_sd(y, fs), y=y)


def split_channels(rows, aux):
    return {"AF7": rows[:, 1], "AF8": rows[:, 2], "Oz": aux}


class DeepGate:
    def __init__(self, fs, thr=0.70, hyst=0.05, hold_s=120.0, min_minutes=20.0, calib_s=180.0,
                 arousal_delta=0.40, alpha_floor=0.10, move_mg=6.0, hf_mult=4.0, max_sd=60.0, step_s=5.0,
                 cap=0.85, eye_max=5, eye_fast=6, hf_max=100.0, resume_s=60.0, amp_ratio=1.8, amp_q=20.0):
        self.fs, self.thr, self.hyst, self.hold_s, self.cap = fs, thr, hyst, hold_s, cap
        self.min_s, self.calib_s, self.arousal_delta = min_minutes * 60, calib_s, arousal_delta
        self.alpha_floor, self.move_mg, self.hf_mult, self.max_sd, self.step = alpha_floor, move_mg, hf_mult, max_sd, step_s
        self.eye_max, self.eye_fast, self.hf_max, self.resume_s = eye_max, eye_fast, hf_max, resume_s
        self.amp_ratio, self.amp_q = amp_ratio, amp_q
        self.still_sw = collections.defaultdict(lambda: collections.deque(maxlen=1440))   # per channel, 2 h
        self.state = "settling"
        self.hist = collections.deque()
        self.base = []
        self.hf_deep = collections.defaultdict(lambda: collections.deque(maxlen=120))   # per channel, 10 min
        self.last_arousal = -1e9
        self.last_move = None
        self.t0 = None

    @property
    def thr_eff(self):
        if len(self.base) >= 6:
            return min(self.cap, max(self.thr, self.wake_delta + 0.30))
        return self.thr

    @property
    def wake_delta(self):
        return float(np.median([b[0] for b in self.base])) if self.base else float("nan")

    @property
    def alpha_thr(self):
        if len(self.base) >= 6:
            return max(self.alpha_floor, 0.8 * float(np.median([b[1] for b in self.base])))
        return self.alpha_floor

    def _stats(self, chans):
        """(delta, alpha, {channel: hf}, number of usable channels, eye movements on AF7-AF8 or None,
        {channel: slow-wave sd})."""
        st = {k: channel_stats(v, self.fs, self.max_sd, self.hf_max) for k, v in chans.items() if v is not None}
        good = {k: s for k, s in st.items() if s}
        if not good:
            return None, None, {}, 0, None, {}
        eye = eye_count(good["AF7"]["y"] - good["AF8"]["y"], self.fs) if "AF7" in good and "AF8" in good else None
        return (float(np.median([s["rel_delta"] for s in good.values()])),
                float(np.median([s["rel_alpha"] for s in good.values()])),
                {k: s["hf"] for k, s in good.items()}, len(good), eye, {k: s["sw"] for k, s in good.items()})

    def amplitude(self, sw):
        """Median over channels of slow-wave sd against the 20th percentile of the channel's still
        windows; None until some channel has 2 minutes of them."""
        need = int(120 / self.step)
        r = [v / float(np.percentile(self.still_sw[k], self.amp_q)) for k, v in sw.items()
             if len(self.still_sw.get(k, ())) >= need]
        return float(np.median(r)) if r else None

    def epoch(self, t, chans):
        """Every 5 s with the last 30 s. Returns this window's numbers and the state."""
        if self.t0 is None:
            self.t0 = t
        delta, alpha, hf, n, eye, sw = self._stats(chans)
        score = delta if delta is not None else np.nan
        still = eye is None or eye <= self.eye_max
        body_still = self.last_move is None or not np.isfinite(self.last_move) or self.last_move <= self.move_mg
        if (delta is not None and still and body_still and len(self.base) < int(self.calib_s / self.step)
                and t - self.t0 <= 900):
            self.base.append((delta, alpha))
        thr = self.thr_eff
        if delta is not None and still and body_still and t - self.t0 > self.calib_s:
            for k, v in sw.items():
                self.still_sw[k].append(v)
        amp = self.amplitude(sw)
        big = amp is not None and amp >= self.amp_ratio
        is_deep = bool(np.isfinite(score) and score >= thr and still and big)
        self.hist.append((t, is_deep))
        while self.hist and self.hist[0][0] <= t - self.hold_s:
            self.hist.popleft()
        if is_deep and t - self.t0 >= self.min_s:
            for k, v in hf.items():
                self.hf_deep[k].append(v)
        if t - self.t0 < self.min_s:
            self.state = "settling"
        elif self.state == "cueing":
            if (not np.isfinite(score) or score < thr - self.hyst or not still
                    or amp is None or amp < self.amp_ratio - 0.2):          # 1.6 to continue
                self.state = "paused"
        else:
            win = [d for _, d in self.hist]
            enough = len(win) >= int(0.9 * self.hold_s / self.step)
            calm = t - self.last_arousal >= self.hold_s
            k = int(self.resume_s / self.step)
            dip_over = self.state == "paused" and calm and len(win) >= k and all(win[-k:])
            if (enough and np.mean(win) >= 0.8 and win[-1] and calm) or dip_over:
                self.state = "cueing"
            elif self.state == "settling":
                self.state = "watching"
        return dict(score=score, alpha=alpha, hf=float(np.median(list(hf.values()))) if hf else None, n_good=n,
                    eye=eye, amp=amp, thr=thr, state=self.state)

    def fast(self, t, chans, move_mg):
        """Every second with the last 10 s: the reason it looks like waking, or None."""
        self.last_move = move_mg
        if self.t0 is None or t - self.t0 < self.calib_s:
            return None
        delta, alpha, hf, n, eye, _ = self._stats(chans)
        reason = None
        if move_mg is not None and np.isfinite(move_mg) and move_mg > self.move_mg:
            reason = f"movement {move_mg:.1f} milli-g"
        elif n == 0:
            reason = "no usable forehead channel"
        elif alpha > self.alpha_thr:
            reason = f"alpha {alpha:.2f}"
        elif eye is not None and eye >= self.eye_fast:
            reason = f"eye movements {eye}"
        elif self._muscle(hf) > self.hf_mult:
            reason = "muscle"
        elif self.state == "cueing" and delta < self.arousal_delta:
            reason = f"delta fell to {delta:.2f}"
        if reason:
            self.last_arousal = t
            if self.state == "cueing":
                self.state = "paused"
        return reason

    def _muscle(self, hf):
        """Smallest over channels of 20-45 Hz power against that channel's own deep-sleep median."""
        r = [v / float(np.median(self.hf_deep[k])) for k, v in hf.items() if len(self.hf_deep.get(k, ())) >= 12]
        return float(min(r)) if r else 0.0


class Cuer:
    """Plays the cued puzzles' sounds while the gate says so."""

    def __init__(self, sounds, gain_db, floor_db, play, log, interval=12.0, jitter=2.0, seed=0):
        self.sounds, self.gain_db, self.floor_db = sounds, gain_db, floor_db
        self.play, self.log = play, log
        self.interval, self.jitter = interval, jitter
        self.rng = random.Random(seed)
        self.order = []
        self.next_t = None
        self.last_cue = None
        self.lowered_for = None
        self.counts = collections.Counter()
        self.n = 0

    def tick(self, t, cueing, arousal):
        if arousal and self.last_cue is not None and t - self.last_cue <= 10 and self.lowered_for != self.last_cue:
            self.lowered_for = self.last_cue
            new = max(self.floor_db, self.gain_db - 3.0)
            if new < self.gain_db:
                self.gain_db = new
                self.log("cue_gain_lowered", f"{new:.1f} dB after: {arousal}")
        if not cueing:
            self.next_t = None
            return
        if self.next_t is None:
            self.next_t = t + 5.0
        if t < self.next_t:
            return
        if not self.order:
            self.order = list(self.sounds)
            self.rng.shuffle(self.order)
        pid = self.order.pop()
        self.play(self.sounds[pid] * 10 ** (self.gain_db / 20))
        self.n += 1
        self.counts[pid] += 1
        self.last_cue = t
        self.log("cue", f"{pid} {self.gain_db:.1f} dB", show=f"cue {self.n} ({self.gain_db:.1f} dB)")
        self.next_t = t + self.interval + self.rng.uniform(-self.jitter, self.jitter)


def level_check(play, key_pressed, flush_keys, start_db=-70.0, step_db=3.0, max_db=0.0, runs=2, log=print):
    """Ascending method of limits with pink noise: the gain (dB on the cue scale) at which the
    sleeper first hears it, averaged over `runs`. None if nothing was heard up to max_db."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "study"))
    from puzzle_sounds import level_noise
    noise = level_noise()
    found = []
    for r in range(runs):
        db = start_db - (6.0 if r else 0.0)
        flush_keys()
        heard = None
        while db <= max_db:
            play(noise * 10 ** (db / 20))
            end = time.time() + 2.0
            while time.time() < end:
                if key_pressed():
                    heard = db
                    break
                time.sleep(0.02)
            if heard is not None:
                break
            db += step_db
        if heard is None:
            return None
        log(f"heard at {heard:.0f} dB")
        found.append(heard)
        time.sleep(1.5)
    return float(np.mean(found))


class NightCueing:
    """The gate and the cue player wired to a night_watch session."""

    def __init__(self, session_dir, buf, event, night_dir, audio=True, gain_db=None, above_db=8.0,
                 interval=12.0, jitter=2.0, gate_kw=None, level_check_fn=None, seed=None):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "study"))
        from puzzle_sounds import make
        self.sdir = Path(session_dir)
        self.buf, self.event, self.night_dir = buf, event, Path(night_dir)
        man = json.loads((self.sdir / "manifest.json").read_text())
        split = json.loads((self.sdir / "split.json").read_text())
        specs = man["sounds"]
        self.sounds = {pid: make(specs[pid]) for pid in split["cued"]}
        self.audio = audio
        if audio:
            import sounddevice as sd
            self.sd = sd
        threshold = None
        if gain_db is None and audio and level_check_fn is not None:
            threshold = level_check_fn(self._play_blocking)
            if threshold is None:
                raise SystemExit("Nothing was heard up to full volume: check the speaker or earbuds and "
                                 "the Windows volume, then start again.")
            gain_db = min(0.0, threshold + above_db)
        if gain_db is None:
            gain_db = -40.0
        floor_db = threshold if threshold is not None else gain_db - 12.0
        self.event("cue_level", f"hearing threshold {threshold if threshold is None else round(threshold, 1)} dB, "
                                f"cue gain {gain_db:.1f} dB, floor {floor_db:.1f} dB")
        self.gate = DeepGate(buf.fs, **(gate_kw or {}))
        self.cuer = Cuer(self.sounds, gain_db, floor_db, self._play, self._log, interval, jitter,
                         seed=seed if seed is not None else int(time.time()))
        self.f = open(self.night_dir / "cue_gate.csv", "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t_s", "score", "alpha", "eye", "amp", "n_good", "thr", "state", "gain_db", "cues"])
        self.state = None
        self.cueing_s = 0.0
        self.last_t = None
        self.threshold = threshold
        man["night_dir"] = str(self.night_dir)
        (self.sdir / "manifest.json").write_text(json.dumps(man, indent=1))

    def _log(self, name, detail, show=None):
        self.event(name, detail, show=show)

    def _play(self, y):
        if self.audio:
            self.sd.play(y, 44100)

    def _play_blocking(self, y):
        if self.audio:
            self.sd.play(y, 44100)
            self.sd.wait()

    def _chans(self, seconds):
        r = self.buf.raw(seconds)
        if r is None:
            return None, None
        rows, aux, acc = r
        mag = np.linalg.norm(acc, axis=1)
        move = float(np.nanstd(mag) * 1000) if np.isfinite(mag).sum() > 10 else None
        return split_channels(rows, aux if np.isfinite(aux).all() else None), move

    def tick(self, t):
        chans, move = self._chans(10)
        if chans is None:
            return
        was = self.gate.state
        reason = self.gate.fast(t, chans, move)
        if reason and was == "cueing":
            self.event("cueing_paused", reason)
        self.cuer.tick(t, self.gate.state == "cueing", reason)
        if self.gate.state == "cueing" and self.last_t is not None:
            self.cueing_s += t - self.last_t
        self.last_t = t

    def epoch(self, t):
        chans, _ = self._chans(30)
        if chans is None:
            return
        was = self.gate.state
        r = self.gate.epoch(t, chans)
        if r["state"] != was:
            if r["state"] == "cueing":
                self.event("cueing_on", f"deep score {r['score']:.2f} (threshold {r['thr']:.2f})")
            elif was == "cueing":
                self.event("cueing_paused", "no usable forehead channel" if not np.isfinite(r["score"]) else
                           f"eye movements {r['eye']}" if r["eye"] is not None and r["eye"] > self.gate.eye_max else
                           f"slow waves {r['amp']:.2f}x the night's quiet level" if r["amp"] is not None and r["amp"] < self.gate.amp_ratio - 0.2 else
                           f"deep score {r['score']:.2f} below {r['thr'] - self.gate.hyst:.2f}")
            elif was == "settling":
                self.event("cue_gate_watching", f"wake delta {self.gate.wake_delta:.2f}, "
                                                f"threshold {r['thr']:.2f}, alpha limit {self.gate.alpha_thr:.2f}")
                if self.gate.wake_delta > 0.5:
                    self.event("cue_gate_warning", "forehead delta was already high while awake: eye movements or "
                                                   "loose electrodes; cues may land outside deep sleep tonight")
        self.w.writerow([f"{t:.1f}", "" if not np.isfinite(r["score"]) else f"{r['score']:.3f}",
                         "" if r["alpha"] is None else f"{r['alpha']:.3f}", "" if r["eye"] is None else r["eye"],
                         "" if r["amp"] is None else f"{r['amp']:.2f}",
                         r["n_good"], f"{r['thr']:.2f}",
                         r["state"], f"{self.cuer.gain_db:.1f}", self.cuer.n])
        if int(t) % 60 < 5:
            self.f.flush()
        return r

    def close(self):
        self.f.close()
        out = dict(cues_total=self.cuer.n, minutes_cueing=round(self.cueing_s / 60, 1),
                   cues_per_puzzle=dict(self.cuer.counts), final_gain_db=self.cuer.gain_db,
                   hearing_threshold_db=self.threshold, night_dir=str(self.night_dir),
                   closed=dt.datetime.now().isoformat(timespec="seconds"))
        (self.sdir / "night_cues.json").write_text(json.dumps(out, indent=1))
        return out
