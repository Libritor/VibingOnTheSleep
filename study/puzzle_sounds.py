"""A short sound for every puzzle: the handle that ties the puzzle to its cue in sleep.

Each sound is a 3-4 note motif (pentatonic, about 1-1.5 s) in one of six timbres. Within one
evening's set every sound has a different (timbre, contour) pair, so no two are easily confused,
and a puzzle's sound is never used for another puzzle. All sounds are scaled to the same RMS so
that one gain setting fits every cue. The level check uses pink noise at the same RMS, which no
puzzle owns.

    python study/puzzle_sounds.py out_dir        # write a demo set of 20 wavs to listen to
"""
import random
import sys
from pathlib import Path

import numpy as np

FS = 44100
TIMBRES = ("bell", "marimba", "pluck", "reed", "glass", "flute")
CONTOURS = {"rise": [0, 2, 4], "fall": [4, 2, 0], "arch": [0, 4, 2], "valley": [4, 0, 2], "leap": [0, 0, 5]}
SCALE = [0, 2, 4, 7, 9, 12, 14, 16, 19, 21]                 # major pentatonic, two octaves
ROOTS = {"low": 220.0, "mid": 293.66, "high": 392.0}
RHYTHMS = ([0.22, 0.22, 0.55], [0.32, 0.16, 0.55], [0.16, 0.32, 0.55], [0.18, 0.18, 0.18, 0.5])
RMS = 0.05                                                # about -26 dBFS before the gain


def specs_for_set(n, seed):
    """n sound specifications with distinct (timbre, contour) pairs."""
    rng = random.Random(seed)
    pairs = [(t, c) for t in TIMBRES for c in CONTOURS]
    rng.shuffle(pairs)
    if n > len(pairs):
        raise ValueError(f"at most {len(pairs)} distinct sounds per set")
    out = []
    for k in range(n):
        t, c = pairs[k]
        out.append(dict(timbre=t, contour=c, register=rng.choice(list(ROOTS)),
                        start=rng.randrange(0, 4), rhythm=rng.randrange(len(RHYTHMS)),
                        seed=rng.randrange(1 << 30)))
    return out


def _note(timbre, f, dur, rng):
    n = int(dur * FS)
    t = np.arange(n) / FS
    if timbre == "bell":
        parts = [(1, 1.0, 3.0), (2.76, 0.5, 5.0), (5.40, 0.25, 8.0), (8.93, 0.12, 12.0)]
        y = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-d * t) for r, a, d in parts)
    elif timbre == "marimba":
        parts = [(1, 1.0, 9.0), (3.93, 0.35, 25.0), (9.0, 0.1, 40.0)]
        y = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-d * t) for r, a, d in parts)
    elif timbre == "pluck":                                  # Karplus-Strong
        p = max(2, int(FS / f))
        buf = rng.uniform(-1, 1, p)
        y = np.empty(n)
        for i in range(n):
            y[i] = buf[i % p]
            buf[i % p] = 0.5 * (buf[i % p] + buf[(i + 1) % p]) * 0.996
    elif timbre == "reed":
        y = sum(np.sin(2 * np.pi * f * k * t) / k for k in (1, 3, 5, 7))
        y *= np.minimum(1, t / 0.03) * np.exp(-2.5 * t)
    elif timbre == "glass":
        parts = [(1, 1.0, 1.5), (2.32, 0.4, 2.5), (4.25, 0.2, 4.0), (6.63, 0.1, 6.0)]
        y = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-d * t) for r, a, d in parts)
        y *= np.minimum(1, t / 0.02)
    else:                                                    # flute: sine, a little octave, breath
        breath = np.convolve(rng.normal(0, 1, n), np.ones(40) / 40, mode="same") * 0.05
        y = np.sin(2 * np.pi * f * t) + 0.2 * np.sin(4 * np.pi * f * t) + breath
        y *= np.minimum(1, t / 0.06) * np.minimum(1, (dur - t) / 0.08).clip(0)
    fade = np.minimum(1, (n - np.arange(n)) / (0.01 * FS))
    return y * fade


def make(spec):
    rng = np.random.default_rng(spec["seed"])
    steps = CONTOURS[spec["contour"]]
    durs = RHYTHMS[spec["rhythm"]]
    if len(durs) == 4:
        steps = steps + [steps[-1]]
    root = ROOTS[spec["register"]]
    notes = []
    for k, d in enumerate(durs):
        deg = SCALE[min(len(SCALE) - 1, spec["start"] + steps[min(k, len(steps) - 1)])]
        notes.append(_note(spec["timbre"], root * 2 ** (deg / 12), d, rng))
    y = np.concatenate(notes)
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-12) * RMS
    peak = np.abs(y).max()
    if peak > 0.5:
        y *= 0.5 / peak
    return y.astype(np.float32)


def level_noise(seconds=0.6, seed=7):
    """Pink noise at the cues' RMS, for the bedtime hearing-threshold check."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    X = np.fft.rfft(rng.normal(0, 1, n))
    f = np.fft.rfftfreq(n, 1 / FS)
    X[1:] /= np.sqrt(f[1:])
    X[0] = 0
    y = np.fft.irfft(X, n)
    y *= np.hanning(n) ** 0.25
    y = y / np.sqrt(np.mean(y ** 2)) * RMS
    return y.astype(np.float32)


def write(path, y):
    import soundfile as sf
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, FS, subtype="FLOAT")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "sound_demo")
    for k, s in enumerate(specs_for_set(20, 1)):
        write(out / f"{k + 1:02d}_{s['timbre']}_{s['contour']}.wav", make(s))
    write(out / "level_noise.wav", level_noise())
    print("wrote", out)
