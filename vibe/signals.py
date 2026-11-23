"""Two deliberate signals a sleeper can make, read from the Muse's forehead electrodes.

  YES  eyes: look left, right, left, right (eyes closed). The horizontal eye channel AF7-AF8
       steps one way and then the other. This is the signal lucid dreamers use (Konkoly et al.
       2021). Counted: fast steps (over 100 ms) of alternating sign, 0.15-1.5 s apart; four in a
       row make the signal.
  NO   brow: frown hard (or clench the jaw) twice. Muscle shows as bursts of 25 Hz and above on
       both forehead electrodes at once (Tuerker et al. 2023 used frowns and smiles). Counted:
       bursts 0.12-2 s long on every usable forehead electrode together; two make the signal.

Why this pair: the two live in different frequency bands, need only the forehead (the ear
electrodes were noise on this user's nights), and their false alarms fall on the safe side. Rapid
eye movements could mimic YES only in REM; an arousal with muscle activity reads as NO.
Eye steps inside a muscle burst are ignored (a frown drags the skin under the electrodes), and a
window with head movement gives no answer.

Thresholds come from the evening calibration (fit): a rest period sets the noise, a few made
signals set the size, and the threshold sits between. Before any calibration the defaults are
multiples of each window's own noise with absolute floors.

Nothing here knows the stage of sleep. The night rig logs every 10-s window's verdict all night,
so the rate of signal-like patterns when no question was asked is measured, not assumed.
"""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, sosfiltfilt

DEFAULTS = dict(eye_thr=40.0, eye_k=7.0, eyes_n=4, emg_thr=4.0, emg_k=4.0, muscle_n=2, move_mg=15.0,
                calibrated=False)


def _clean(x, fs):
    for f0 in (60.0, 120.0):
        if f0 < fs / 2 - 1:
            b, a = iirnotch(f0, 30.0, fs)
            x = filtfilt(b, a, x)
    return sosfiltfilt(butter(2, 0.3, btype="high", fs=fs, output="sos"), x)


def usable(x):
    """A forehead channel that is on the head: finite, not railed, not flat."""
    return bool(np.isfinite(x).all() and ((x < 5) | (x > 1640)).mean() < 0.01 and x.std() > 0.5)


def eye_steps(heog, fs):
    """|change over 100 ms| of the 10 Hz low-passed eye channel, and the signed change."""
    lp = sosfiltfilt(butter(2, 10.0, btype="low", fs=fs, output="sos"), heog)
    w = max(1, int(round(0.10 * fs)))
    d = np.zeros_like(lp)
    d[w:] = lp[w:] - lp[:-w]
    return d


def emg_envelope(x, fs):
    """RMS over 100 ms of the band from 25 Hz to 110 Hz (or just under Nyquist)."""
    hi = min(110.0, fs / 2 - 3)
    y = sosfiltfilt(butter(4, [25.0, hi], btype="band", fs=fs, output="sos"), x)
    return np.sqrt(uniform_filter1d(y * y, max(1, int(0.10 * fs))))


def _runs(mask):
    """[(start, stop)] of the True runs of a boolean array."""
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def bursts(env, fs, thr):
    """Muscle bursts in an envelope: above thr for 0.12-2 s, gaps under 0.15 s bridged."""
    out = []
    for a, b in _runs(env > thr):
        if out and a - out[-1][1] < 0.15 * fs:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return [(a, b) for a, b in out if 0.12 * fs <= b - a <= 2.0 * fs]


class Decoder:
    def __init__(self, params=None):
        self.p = dict(DEFAULTS)
        if params:
            self.p.update(params)

    @classmethod
    def load(cls, path):
        path = Path(path)
        return cls(json.loads(path.read_text())["params"]) if path.exists() else cls()

    # -- measurement
    def measure(self, rows, fs, acc=None):
        """rows: (n, 4) TP9 AF7 AF8 TP10 in uV. Everything the verdict is made from."""
        af7, af8 = rows[:, 1], rows[:, 2]
        ok7, ok8 = usable(af7), usable(af8)
        move = None
        if acc is not None and np.isfinite(acc).sum() > 30:
            move = float(np.nanstd(np.linalg.norm(acc, axis=1)) * 1000)
        out = dict(ok=ok7 and ok8, n_forehead=int(ok7) + int(ok8), move_mg=move, eyes=0, muscle=0,
                   eye_amp=0.0, emg_amp=0.0, eye_noise=0.0, emg_base=0.0)
        if not (ok7 or ok8):
            return out
        chans = [_clean(c - c.mean(), fs) for c, ok in ((af7, ok7), (af8, ok8)) if ok]
        # muscle: bursts present on every usable forehead electrode at once
        envs = [emg_envelope(c, fs) for c in chans]
        edge = int(0.25 * fs)                       # filter edges
        thr = [max(self.p["emg_thr"], self.p["emg_k"] * float(np.median(e[edge:-edge]))) for e in envs]
        both = np.all([e > t for e, t in zip(envs, thr)], axis=0)
        both[:edge] = both[-edge:] = False
        bl = bursts(both.astype(float), fs, 0.5)
        env = np.mean(envs, axis=0)
        out.update(muscle=len(bl), emg_base=float(np.median(env[edge:-edge])),
                   emg_amp=float(np.percentile(env[edge:-edge], 98)))
        # eyes: needs both sides of the forehead
        if ok7 and ok8:
            d = eye_steps(chans[0] - chans[1], fs)
            d[:edge] = d[-edge:] = 0
            noise = float(np.median(np.abs(d)) / 0.6745)
            ethr = max(self.p["eye_thr"], self.p["eye_k"] * noise)
            pk, _ = find_peaks(np.abs(d), height=ethr, distance=max(1, int(0.15 * fs)))
            pad = int(0.10 * fs)
            pk = [k for k in pk if not any(a - pad <= k <= b + pad for a, b in bl)]
            run = best = 0
            for i, k in enumerate(pk):
                gap = (k - pk[i - 1]) / fs if i else None
                if i and np.sign(d[k]) != np.sign(d[pk[i - 1]]) and 0.15 <= gap <= 1.5:
                    run += 1
                else:
                    run = 1
                best = max(best, run)
            top = np.sort(np.abs(d[pk]))[-4:] if len(pk) else np.array([0.0])
            out.update(eyes=best, eye_noise=noise, eye_amp=float(np.median(top)))
        return out

    def decode(self, rows, fs, acc=None):
        """'yes', 'no', 'ambiguous' or None, with the reason, for one response window."""
        m = self.measure(rows, fs, acc)
        eyes, muscle = m["eyes"] >= self.p["eyes_n"], m["muscle"] >= self.p["muscle_n"]
        if m["n_forehead"] == 0:
            ans, why = None, "no forehead electrode on the skin"
        elif m["move_mg"] is not None and m["move_mg"] > self.p["move_mg"]:
            ans, why = None, f"head moved ({m['move_mg']:.0f} milli-g)"
        elif eyes and muscle:
            ans, why = "ambiguous", "both signals"
        elif eyes:
            ans, why = "yes", f"{m['eyes']} alternating eye steps"
        elif muscle:
            ans, why = "no", f"{m['muscle']} muscle bursts"
        else:
            ans, why = None, "no signal"
        m.update(answer=ans, why=why)
        return m

    # -- calibration
    def fit(self, trials, fs):
        """trials: [(label, rows)] with label 'rest', 'eyes' or 'muscle'. Sets the thresholds from
        the rest noise and the size of the made signals, and returns the numbers used."""
        base = Decoder(dict(DEFAULTS, eye_thr=0.0, emg_thr=0.0))        # sizes without the floors
        ms = {lab: [base.measure(r, fs) for l, r in trials if l == lab] for lab in ("rest", "eyes", "muscle")}
        rest = [m for m in ms["rest"] if m["n_forehead"]]
        eye_noise = float(np.median([m["eye_noise"] for m in rest])) if rest else 5.0
        emg_base = float(np.median([m["emg_base"] for m in rest])) if rest else 1.0
        eye_amp = float(np.median([m["eye_amp"] for m in ms["eyes"]])) if ms["eyes"] else 0.0
        emg_amp = float(np.median([m["emg_amp"] for m in ms["muscle"]])) if ms["muscle"] else 0.0
        self.p.update(eye_thr=round(max(20.0, 6 * eye_noise, 0.4 * eye_amp), 1),
                      emg_thr=round(max(3.0, 3 * emg_base, 0.3 * emg_amp), 2), calibrated=True)
        return dict(eye_noise=eye_noise, eye_amp=eye_amp, emg_base=emg_base, emg_amp=emg_amp)

    def score(self, trials, fs):
        """Confusion counts on labelled trials: {label: {answer: n}}."""
        want = {"rest": None, "eyes": "yes", "muscle": "no"}
        table = {lab: {} for lab in want}
        right = 0
        for lab, rows in trials:
            a = self.decode(rows, fs)["answer"]
            table[lab][str(a)] = table[lab].get(str(a), 0) + 1
            right += a == want[lab]
        return table, right, len(trials)

    def save(self, path, **extra):
        Path(path).write_text(json.dumps(dict(params=self.p, **extra), indent=1))


# ---------------------------------------------------------------- made signals for tests
def synth_rest(n, fs, seed=0, rms=10.0):
    """Quiet forehead EEG for tests: 1/f noise (about 10 uV, little above 25 Hz) on the Muse's
    800 uV offset, independent per electrode."""
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n, 1 / fs)
    out = np.empty((n, 4))
    for ch in range(4):
        X = np.fft.rfft(rng.normal(0, 1, n))
        X[1:] /= f[1:]
        X[0] = 0
        X[f < 0.5] = 0
        y = np.fft.irfft(X, n)
        out[:, ch] = 800.0 + y / (y.std() + 1e-12) * rms + rng.normal(0, 0.8, n)
    return out


def synth_eyes(n, fs, amp=120.0, start=1.0, period=0.5, seed=0):
    """What AF7-AF8 does for left-right-left-right: returns (n, 4) to ADD to rows."""
    t = np.arange(n) / fs
    pos = np.zeros(n)
    for k, level in enumerate((1, -1, 1, -1, 0)):
        pos[t >= start + k * period] = level
    pos = uniform_filter1d(pos, max(1, int(0.04 * fs)))          # a saccade takes about 40 ms
    out = np.zeros((n, 4))
    out[:, 1], out[:, 2] = amp / 2 * pos, -amp / 2 * pos
    return out


def synth_muscle(n, fs, amp=25.0, starts=(1.0, 2.2), seconds=0.45, seed=0):
    """Two frowns: bursts of 30-100 Hz noise on both forehead electrodes."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    hi = min(100.0, fs / 2 - 3)
    sos = butter(4, [30.0, hi], btype="band", fs=fs, output="sos")
    out = np.zeros((n, 4))
    for ch in (1, 2):
        noise = sosfiltfilt(sos, rng.normal(0, 1, n))
        noise *= amp / (noise.std() + 1e-9)
        gate = np.zeros(n)
        for s in starts:
            gate[(t >= s) & (t < s + seconds)] = 1
        out[:, ch] = noise * uniform_filter1d(gate, max(1, int(0.03 * fs)))
    return out
