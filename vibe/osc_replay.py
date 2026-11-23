"""A stand-in headband: stream one of this rig's recordings over OSC in real time, the way MuseLog
does (/muse/eeg at 256 Hz, /muse/acc), so the evening and night scripts can be run live with
nobody wearing anything.

    python vibe/osc_replay.py data/muselog/20261028/muse_session_20261028_034217_Muse2.npz --port 5055 --start-minutes 8 --minutes 15
    python vibe/night.py --participant P-TEST --self --osc-port 5055 --no-audio --agent stub --cue-gain -30
"""
import argparse
import time

import numpy as np
from pythonosc.udp_client import SimpleUDPClient


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5055)
    ap.add_argument("--start-minutes", type=float, default=0.0)
    ap.add_argument("--minutes", type=float, default=15.0)
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True)
    t = np.asarray(z["t"], float)
    eeg = np.asarray(z["eeg"], float)[:, :4]
    acc = np.asarray(z["acc"], float) if "acc" in z.files and len(z["acc"]) else None
    i = int(np.searchsorted(t, t[0] + a.start_minutes * 60))
    j = int(np.searchsorted(t, t[i] + a.minutes * 60))
    out = SimpleUDPClient(a.host, a.port)
    t0, base = time.time(), t[i]
    print(f"streaming {(t[j - 1] - base) / 60:.1f} min of {a.npz} to {a.host}:{a.port}", flush=True)
    for k in range(i, j, 16):
        wait = t0 + (t[k] - base) - time.time()
        if wait > 0:
            time.sleep(wait)
        if acc is not None:
            m = min(len(acc) - 1, int(np.searchsorted(acc[:, 0], t[k])))
            out.send_message("/muse/acc", [float(v) for v in acc[m, 1:4]])
        for n in range(k, min(k + 16, j)):
            out.send_message("/muse/eeg", [float(v) for v in eeg[n]])
    print("stream ended", flush=True)


if __name__ == "__main__":
    main()
