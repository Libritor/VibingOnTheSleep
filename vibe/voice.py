"""The voice of the night rig: soft, natural, and made on this computer.

Two styles. "day" is a warm speaking voice for the evening and the morning. "night" is a
near-whisper for anything said to someone who is asleep or half asleep. Both come from Kokoro
(an 82-million-parameter neural voice, run locally with onnxruntime: no account, no network, so a
Wi-Fi drop at 4 a.m. cannot silence it). Every sentence is rendered once and cached as a wav.

All speech is scaled to the same RMS as the cue sounds (study/puzzle_sounds.py), so gains are on
the one scale the bedtime hearing check measures: a night sentence is played a set number of dB
above the level at which the sleeper first heard the test noise. Each sentence fades in over
150 ms so that nothing starts with a click.

Other backends, chosen with backend=...:
    "edge"  Microsoft's online neural voices through edge-tts. Sends the text to Microsoft.
    "sapi"  Windows' built-in synthesiser (the nap rig's old voice). Robotic; the last resort.

    python vibe/voice.py                      # write samples of every style to study/audio/vibe/samples
    python vibe/voice.py --say "Good night."  # hear one sentence in the night voice
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "voice"
CACHE = ROOT / "study" / "audio" / "vibe"          # study/audio/ is not in git
RMS = 0.05                                         # the cue sounds' level (about -26 dBFS)
STYLES = {
    "day": dict(kokoro="af_heart", speed=1.0, edge="en-US-AvaNeural", edge_rate="+0%"),
    "night": dict(kokoro="af_nicole", speed=0.88, edge="en-GB-SoniaNeural", edge_rate="-15%"),
}
# another Kokoro voice for a style, for every script: set VIBE_NIGHT_VOICE or VIBE_DAY_VOICE
# (for example bf_emma, af_bella, af_sarah, am_michael; `python vibe/voice.py --list` prints them all)
for _style in STYLES:
    _v = os.environ.get(f"VIBE_{_style.upper()}_VOICE")
    if _v:
        STYLES[_style]["kokoro"] = _v
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"


def _shape(y, fs, fade_in=0.15, fade_out=0.08):
    """Mono float32 at the common RMS, silence trimmed, faded at both ends."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    loud = np.flatnonzero(np.abs(y) > 0.02 * (np.abs(y).max() + 1e-9))
    if len(loud):
        y = y[max(0, loud[0] - int(0.05 * fs)):loud[-1] + int(0.12 * fs)]
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-12) * RMS
    peak = np.abs(y).max()
    if peak > 0.7:
        y = y * (0.7 / peak)
    a, b = int(fade_in * fs), int(fade_out * fs)
    if len(y) > a + b:
        y[:a] *= np.linspace(0, 1, a) ** 2
        y[-b:] *= np.linspace(1, 0, b)
    return y.astype(np.float32)


def chime(fs=44100, seconds=3.2):
    """The wake sound: three soft glass notes rising out of silence (no beep)."""
    t = np.arange(int(seconds * fs)) / fs
    y = np.zeros_like(t)
    for k, f in enumerate((392.0, 523.25, 659.25)):
        t0 = 0.55 * k
        u = np.clip(t - t0, 0, None)
        env = np.minimum(1, u / 0.35) * np.exp(-1.3 * u) * (t >= t0)
        y += env * (np.sin(2 * np.pi * f * u) + 0.25 * np.sin(4 * np.pi * f * u) + 0.08 * np.sin(6 * np.pi * f * u))
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-12) * RMS
    return (y * np.minimum(1, (seconds - t) / 0.4)).astype(np.float32), fs


class Voice:
    def __init__(self, backend="kokoro", enabled=True, cache=CACHE, log=None):
        self.backend, self.enabled, self.cache = backend, enabled, Path(cache)
        self.log = log or (lambda *a, **k: None)
        self._kokoro = None
        self._sd = None
        self.busy_until = 0.0            # wall time until which a sentence is still playing

    # -- synthesis
    def _load_kokoro(self):
        if self._kokoro is None:
            model, voices = MODEL_DIR / "kokoro-v1.0.onnx", MODEL_DIR / "voices-v1.0.bin"
            if not (model.exists() and voices.exists()):
                raise FileNotFoundError(
                    f"The voice model is not in {MODEL_DIR}. Download kokoro-v1.0.onnx and voices-v1.0.bin "
                    f"from {MODEL_URL} (about 350 MB), or pass --voice sapi.")
            from kokoro_onnx import Kokoro
            self._kokoro = Kokoro(str(model), str(voices))
        return self._kokoro

    def _render(self, text, style):
        s = STYLES[style]
        if self.backend == "kokoro":
            y, fs = self._load_kokoro().create(text, voice=s["kokoro"], speed=s["speed"], lang="en-us")
            return y, fs
        if self.backend == "edge":
            import asyncio
            import edge_tts
            import soundfile as sf
            tmp = self.cache / "_edge.mp3"
            asyncio.run(edge_tts.Communicate(text, s["edge"], rate=s["edge_rate"]).save(str(tmp)))
            y, fs = sf.read(str(tmp), dtype="float32")
            tmp.unlink(missing_ok=True)
            return y, fs
        if self.backend == "sapi":
            import soundfile as sf
            tmp = self.cache / "_sapi.wav"
            safe = text.replace("'", "''")
            ps = ("Add-Type -AssemblyName System.Speech; "
                  "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = -2; "
                  f"$s.SetOutputToWaveFile('{tmp}'); $s.Speak('{safe}'); $s.Dispose()")
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)
            y, fs = sf.read(str(tmp), dtype="float32")
            tmp.unlink(missing_ok=True)
            return y, fs
        raise ValueError(f"unknown voice backend {self.backend}")

    def path(self, text, style):
        s = STYLES[style]
        key = hashlib.sha1(f"{self.backend}|{s}|{text}".encode()).hexdigest()[:20]
        return self.cache / f"{style}_{key}.wav"

    def wav(self, text, style="night"):
        """(samples, fs) for a sentence, rendered once. If the chosen backend fails in the night
        (a missing model, no network for "edge"), Windows' own voice says it instead of nothing."""
        import soundfile as sf
        p = self.path(text, style)
        if p.exists():
            y, fs = sf.read(str(p), dtype="float32")
            return y, fs
        self.cache.mkdir(parents=True, exist_ok=True)
        try:
            y, fs = self._render(text, style)
        except Exception as e:
            if self.backend == "sapi":
                raise
            self.log("voice_fallback", f"{self.backend} failed ({type(e).__name__}: {str(e)[:80]}); Windows voice")
            keep, self.backend = self.backend, "sapi"
            try:
                y, fs = self._render(text, style)
            finally:
                self.backend = keep
        y = _shape(y, fs)
        sf.write(str(p), y, fs, subtype="FLOAT")
        return y, fs

    def prerender(self, texts, style="night"):
        for t in texts:
            self.wav(t, style)

    # -- playing
    def _device(self):
        if self._sd is None:
            import sounddevice as sd
            self._sd = sd
        return self._sd

    def play(self, y, fs, gain_db=0.0, block=True):
        dur = len(y) / fs
        if not self.enabled:
            return dur
        sd = self._device()
        sd.play(np.clip(y * 10 ** (gain_db / 20), -1, 1), fs)
        self.busy_until = time.time() + dur
        if block:
            sd.wait()
        return dur

    def say(self, text, style="night", gain_db=0.0, block=True):
        """Speak; returns the sentence's length in seconds. With the voice off (tests, replays)
        nothing is rendered and the length is estimated from the word count."""
        if not self.enabled:
            return 0.4 + 0.38 * len(text.split())
        y, fs = self.wav(text, style)
        return self.play(y, fs, gain_db, block)

    def chime(self, gain_db=0.0, block=True):
        y, fs = chime()
        return self.play(y, fs, gain_db, block)

    def speaking(self):
        return time.time() < self.busy_until


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="kokoro", choices=("kokoro", "edge", "sapi"))
    ap.add_argument("--say", help="speak this sentence now")
    ap.add_argument("--style", default="night", choices=sorted(STYLES))
    ap.add_argument("--gain", type=float, default=0.0, help="dB on the cue scale")
    ap.add_argument("--kokoro", help="try another Kokoro voice for this one sentence (see --list)")
    ap.add_argument("--list", action="store_true", help="print the Kokoro voices")
    a = ap.parse_args()
    v = Voice(a.backend)
    if a.list:
        print(" ".join(sorted(v._load_kokoro().get_voices())))
        return
    if a.kokoro:
        STYLES[a.style]["kokoro"] = a.kokoro
    if a.say:
        v.say(a.say, a.style, a.gain)
        return
    import soundfile as sf
    out = CACHE / "samples"
    out.mkdir(parents=True, exist_ok=True)
    lines = {"day": "Good evening. Tell me what you would like built tonight, and I will take it from there.",
             "night": "The login page is working now. Should I add the dark theme? "
                      "Look left and right for yes. Frown for no."}
    for style, text in lines.items():
        t0 = time.time()
        y, fs = v.wav(text, style)
        sf.write(str(out / f"{a.backend}_{style}.wav"), y, fs)
        print(f"{a.backend} {style}: {len(y) / fs:.1f} s of speech, rendered in {time.time() - t0:.1f} s")
    y, fs = chime()
    sf.write(str(out / "wake_chime.wav"), y, fs)
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
