"""What the evening, night and morning scripts of Vibing on the Sleep share: the consent gate,
the session folder, the microphone (a spoken report that ends when the speaker stops), and
speech-to-text on this computer."""
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "study"
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(ROOT / "rig"))
sys.path.insert(0, str(ROOT / "detector"))

try:
    import msvcrt
except ImportError:
    msvcrt = None

YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "okay", "ok", "do", "go"}
NO_WORDS = {"no", "nope", "nah", "don't", "dont", "stop", "not"}


def participant_dir(pid, allow_self):
    pdir = STUDY / "participants" / pid
    path = pdir / "consent_vibe.json"
    if not path.exists():
        sys.exit(f"No consent record for Vibing on the Sleep. Run:\n  python study/consent.py --participant {pid} "
                 f"--study vibe --self      (or --reb-code <number>)")
    c = json.loads(path.read_text())
    if not c.get("consented") or c.get("withdrawn"):
        sys.exit(f"{pid} has not consented to, or has withdrawn from, Vibing on the Sleep.")
    if c.get("reb_code", "") in ("", "SELF-EXPERIMENT") and not allow_self:
        sys.exit("No REB approval code on file. Self-experiments by an investigator need --self.")
    return pdir


def new_session(pdir):
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sdir = pdir / "vibe" / stamp
    k = 1
    while sdir.exists():
        k += 1
        sdir = pdir / "vibe" / f"{stamp}_{k}"
    sdir.mkdir(parents=True)
    return sdir, sdir.name


def find_session(pdir, which="latest"):
    d = pdir / "vibe"
    if which != "latest":
        p = Path(which)
        p = p if p.is_absolute() else (d / which if (d / which).exists() else ROOT / which)
        if not (p / "evening.json").exists():
            sys.exit(f"No evening.json in {p}: run vibe/evening.py first.")
        return p
    done = sorted(p for p in d.iterdir() if (p / "evening.json").exists()) if d.exists() else []
    if not done:
        sys.exit("No evening session yet. Run: python vibe/evening.py --participant <P> --self")
    return done[-1]


def flush_keys():
    while msvcrt and msvcrt.kbhit():
        msvcrt.getwch()


def key_pressed():
    return bool(msvcrt and msvcrt.kbhit())


class Ears:
    """The microphone and the transcriber. With enabled=False (tests, replays) nothing is recorded
    and `scripted` supplies what the sleeper 'said', one string per call."""

    def __init__(self, enabled=True, stt_model="base", scripted=None):
        self.enabled, self.stt_model = enabled, stt_model
        self.scripted = list(scripted or [])
        self._model = None
        if enabled:
            import sounddevice as sd
            import soundfile as sf
            self.sd, self.sf = sd, sf

    def check(self):
        """Open the microphone for a moment, so a blocked one stops the evening, not the night."""
        if not self.enabled:
            return
        try:
            self.sd.rec(1600, samplerate=16000, channels=1, dtype="float32")
            self.sd.wait()
        except Exception as e:
            sys.exit(f"The microphone cannot be opened ({e}). In Windows Settings > Privacy & security > "
                     "Microphone, turn on access for Python, then start again.")

    def record(self, path, max_seconds, min_seconds=8.0, quiet_seconds=6.0, fs=16000):
        """Record until the speaker has been quiet for `quiet_seconds` (after `min_seconds`), or
        `max_seconds`. Returns (path or None, seconds recorded)."""
        if not self.enabled:
            return None, 0.0
        blocks, levels, last_voice = [], [], 0.0
        with self.sd.InputStream(samplerate=fs, channels=1, dtype="float32") as stream:
            t = 0.0
            while t < max_seconds:
                x, _ = stream.read(fs // 2)
                blocks.append(x[:, 0].copy())
                levels.append(float(np.sqrt(np.mean(x ** 2)) + 1e-9))
                t += 0.5
                floor = float(np.percentile(levels, 20))
                if levels[-1] > max(2.5 * floor, 0.004):
                    last_voice = t
                if t >= min_seconds and t - last_voice >= quiet_seconds:
                    break
        y = np.concatenate(blocks)
        self.sf.write(str(path), y, fs)
        return Path(path), len(y) / fs

    def start(self, seconds, fs=16000):
        """Begin a fixed-length recording without waiting for it (the 12 s after a night question)."""
        if not self.enabled:
            return None
        return self.sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="float32")

    def finish(self, rec, path, fs=16000):
        if rec is None:
            return None
        self.sd.wait()
        self.sf.write(str(path), rec, fs)
        return Path(path)

    def transcribe(self, wav):
        """Speech to text on this computer (faster-whisper); '' when nothing was said."""
        if wav is None:
            return self.scripted.pop(0) if self.scripted else ""
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.stt_model, device="cpu", compute_type="int8")
        from transcribe import transcribe_file
        text = transcribe_file(self._model, wav)
        Path(wav).with_suffix(".txt").write_text(text, encoding="utf-8")
        return text


def yes_no_words(text):
    """'yes', 'no' or None from a few mumbled words (first clear word wins)."""
    for w in text.lower().replace(",", " ").replace(".", " ").split()[:4]:
        if w in YES_WORDS:
            return "yes"
        if w in NO_WORDS:
            return "no"
    return None


def hold_awake(on):
    """Keep Windows from sleeping while a night runs (the laptop otherwise freezes the rig)."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def clock():
    return time.strftime("%H:%M:%S")
