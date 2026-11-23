"""What the evening and morning puzzle sessions share: the consent check, the timed answer prompt,
the hidden cued/uncued split and its commitment, and a sound that repeats while a puzzle is shown."""
import hashlib
import json
import secrets
import sys
import threading
import time
from pathlib import Path

try:
    import msvcrt
except ImportError:
    msvcrt = None

STUDY = Path(__file__).resolve().parent


def participant_dir(pid, allow_self):
    pdir = STUDY / "participants" / pid
    path = pdir / "consent_puzzles.json"
    if not path.exists():
        sys.exit(f"No consent record for the puzzle study. Run: python study/consent.py --participant {pid} "
                 f"--study puzzles --self   (or --reb-code <number>)")
    c = json.loads(path.read_text())
    if not c.get("consented") or c.get("withdrawn"):
        sys.exit(f"{pid} has not consented to, or has withdrawn from, the puzzle study.")
    if c.get("reb_code", "") in ("", "SELF-EXPERIMENT") and not allow_self:
        sys.exit("No REB approval code on file. Self-experiments by an investigator need --self.")
    return pdir


def sessions(pdir):
    d = pdir / "puzzle_sessions"
    return sorted(p for p in d.iterdir() if (p / "manifest.json").exists()) if d.exists() else []


def stratified_split(rows, rng):
    """rows: [(puzzle id, family)]. Half of each family to 'cued', half to 'uncued'; a family's odd
    puzzle goes to whichever group is smaller so far (a coin decides ties). The analysis re-runs this
    same procedure for its randomisation test."""
    cued, uncued = [], []
    fams = sorted({f for _, f in rows})
    for fam in fams:
        ids = sorted(i for i, f in rows if f == fam)
        rng.shuffle(ids)
        h = len(ids) // 2
        cued += ids[:h]
        uncued += ids[h:2 * h]
        if len(ids) % 2:
            odd = ids[-1]
            if len(cued) < len(uncued) or (len(cued) == len(uncued) and rng.random() < 0.5):
                cued.append(odd)
            else:
                uncued.append(odd)
    return sorted(cued), sorted(uncued)


def commitment(cued, uncued, salt):
    body = json.dumps({"cued": sorted(cued), "uncued": sorted(uncued)}, sort_keys=True) + salt
    return hashlib.sha256(body.encode()).hexdigest()


def make_split(rows, rng):
    cued, uncued = stratified_split(rows, rng)
    salt = secrets.token_hex(16)
    return dict(cued=cued, uncued=uncued, salt=salt, commitment=commitment(cued, uncued, salt))


class SoundLoop:
    """Plays one sound now and then every `every` seconds until stopped."""

    def __init__(self, y, every, gain_db, enabled=True):
        self.y = y * 10 ** (gain_db / 20)
        self.every, self.enabled = every, enabled
        self.stop_ev = threading.Event()
        self.th = None

    def __enter__(self):
        if self.enabled:
            import sounddevice as sd
            self.sd = sd

            def run():
                while not self.stop_ev.is_set():
                    self.sd.play(self.y, 44100)
                    self.stop_ev.wait(self.every)
            self.th = threading.Thread(target=run, daemon=True)
            self.th.start()
        return self

    def __exit__(self, *exc):
        self.stop_ev.set()
        if self.th:
            self.th.join(timeout=2)
            self.sd.stop()


def timed_answer(seconds, checker):
    """Read answers until one is right, the time runs out, or the person types 'skip'.
    Returns (solved, seconds_taken, [(t, text, ok, message)])."""
    attempts = []
    start = time.time()
    end = start + seconds
    if msvcrt is None:                          # plain input(), no live countdown
        while time.time() < end:
            text = input("> ").strip()
            if time.time() > end:
                print("Time was up before that answer.")
                break
            if text.lower() == "skip":
                break
            if text:
                ok, msg = checker(text)
                attempts.append((round(time.time() - start, 1), text, ok, msg))
                print(("  CORRECT" if ok else "  no: ") + ("" if ok else msg))
                if ok:
                    return True, round(time.time() - start, 1), attempts
        return False, round(min(time.time(), end) - start, 1), attempts
    buf = ""
    shown = -1
    sys.stdout.write(f"[{int(seconds) // 60}:{int(seconds) % 60:02d}] > ")
    sys.stdout.flush()
    while time.time() < end:
        left = int(end - time.time())
        if left % 15 == 0 and left != shown:
            shown = left
            sys.stdout.write(f"\r[{left // 60}:{left % 60:02d}] > {buf}  \b\b")
            sys.stdout.flush()
        if not msvcrt.kbhit():
            time.sleep(0.02)
            continue
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            text = buf.strip()
            buf = ""
            sys.stdout.write("\n")
            if text.lower() == "skip":
                break
            if text:
                ok, msg = checker(text)
                attempts.append((round(time.time() - start, 1), text, ok, msg))
                print("  CORRECT" if ok else f"  no: {msg}")
                if ok:
                    return True, round(time.time() - start, 1), attempts
            left = int(end - time.time())
            sys.stdout.write(f"[{left // 60}:{left % 60:02d}] > ")
        elif ch == "\b":
            if buf:
                buf = buf[:-1]
                sys.stdout.write("\b \b")
        elif ch in ("\x00", "\xe0"):
            msvcrt.getwch()                      # arrow and function keys: ignore
        elif ch == "\x03":
            raise KeyboardInterrupt
        else:
            buf += ch
            sys.stdout.write(ch)
        sys.stdout.flush()
    else:
        print("\n  time")
    return False, round(min(time.time(), end) - start, 1), attempts
