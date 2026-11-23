"""Transcribe every spoken report on this computer (faster-whisper, no cloud). Writes report_N.txt
next to each report_N.wav and skips ones already done.

    python study/transcribe.py --participant P-0001 [--model base]
"""
import argparse
from pathlib import Path

STUDY = Path(__file__).resolve().parent


def load_16k(wav):
    """Read with soundfile and resample to 16 kHz mono (avoids faster-whisper's PyAV decoder,
    whose installed version does not match)."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    x, fs = sf.read(str(wav), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != 16000:
        from math import gcd
        g = gcd(int(fs), 16000)
        x = resample_poly(x, 16000 // g, int(fs) // g).astype(np.float32)
    return x


def transcribe_file(model, wav):
    segments, _ = model.transcribe(load_16k(wav), language="en", vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--model", default="base")
    a = ap.parse_args()
    from faster_whisper import WhisperModel
    model = WhisperModel(a.model, device="cpu", compute_type="int8")
    wavs = sorted((STUDY / "participants" / a.participant / "sessions").glob("*/report_*.wav"))
    for wav in wavs:
        txt = wav.with_suffix(".txt")
        if txt.exists():
            continue
        text = transcribe_file(model, wav)
        txt.write_text(text, encoding="utf-8")
        print(f"{wav.parent.name}/{wav.name}: {text[:80]}")
    print(f"{len(wavs)} reports")


if __name__ == "__main__":
    main()
