"""Night watch: a whole night on the headband, recorded and staged live, with no sound.

The nap rig cues and wakes; this does neither. It keeps the laptop awake, takes MuseLog's OSC
stream (raw EEG at 256 Hz, accelerometer if the build sends it), writes every sample to raw.csv,
runs the validated detector causally every 5 s on the last 30 s, and logs the stage probabilities,
the moments the rig's triggers would have fired (sleep onset by EEG, the N2 guard, and a REM cue
once P(REM) has held --rem-p for 30 s, with a refractory period) without acting on any of them.
The detector calibrates itself on the first minutes of windows, which should be eyes-closed wake:
lie down, close your eyes, and let the first three minutes pass before sleeping if you can.

    python rig/night_watch.py --participant P-0001                  # waits for MuseLog on port 5000
    python rig/night_watch.py --participant P-TEST --replay data/sc/SC4001E0-PSG.edf --replay-hours 1

Output: study/participants/<P>/nights/<stamp>/{raw.csv, detector.csv, events.csv, summary.json}.
In the morning, muselog_night.py on the phone's own CSV gives the definitive scoring; this is the
live copy and the dry run of the triggers.

Puzzle cueing (rig/sws_cue.py): with --cues <evening session folder> from study/puzzle_evening.py,
the night also plays the cued puzzles' sounds in slow-wave sleep, after a bedtime hearing check.
That is the one case where something sounds in the night; without --cues nothing does.

    python rig/night_watch.py --participant P-0001 --cues study/participants/P-0001/puzzle_sessions/<stamp>
"""
import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "detector"))
sys.path.insert(0, str(ROOT / "rig"))
from model import LiveDetector  # noqa: E402
from sources import Buffer, MuseOSC, Replay  # noqa: E402
from sws_cue import NightCueing, level_check  # noqa: E402

try:
    import msvcrt
except ImportError:
    msvcrt = None

STAGES = ("W", "N1", "N2", "N3", "REM")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--osc-port", type=int, default=5000)
    ap.add_argument("--model", default=str(ROOT / "models" / "nap_base3min.joblib"),
                    help="nap_base3min calibrates on 3 min of eyes-closed wake, night_base10min on 10")
    ap.add_argument("--max-hours", type=float, default=10.0)
    ap.add_argument("--silence-minutes", type=float, default=15.0,
                    help="end the night this long after the stream stops")
    ap.add_argument("--p-sleep", type=float, default=0.7, help="EEG sleep-onset trigger: P(not wake)")
    ap.add_argument("--p-n2", type=float, default=0.7, help="N2 guard: P(N2 or N3)")
    ap.add_argument("--rem-p", type=float, default=0.8, help="a REM cue would fire once P(REM) holds this for 30 s")
    ap.add_argument("--rem-refractory", type=float, default=600.0, help="seconds between would-be REM cues")
    ap.add_argument("--replay", help="Sleep-EDF PSG file instead of the headband (a dry run of the code)")
    ap.add_argument("--replay-hours", type=float, default=1.0)
    ap.add_argument("--replay-start-hours", type=float, default=0.0, help="where in the replayed night to start")
    ap.add_argument("--cues", help="evening puzzle session folder: cue its cued puzzles in slow-wave sleep")
    ap.add_argument("--cue-gain", type=float, help="cue gain in dB, skipping the bedtime hearing check")
    ap.add_argument("--cue-above", type=float, default=8.0, help="cue gain above the hearing threshold, dB")
    ap.add_argument("--cue-interval", type=float, default=12.0, help="seconds between cues (+/- --cue-jitter)")
    ap.add_argument("--cue-jitter", type=float, default=2.0)
    ap.add_argument("--deep-thr", type=float, default=0.70, help="relative delta that counts as slow-wave sleep")
    ap.add_argument("--min-minutes", type=float, default=20.0, help="no cue before this far into the night")
    ap.add_argument("--move-mg", type=float, default=6.0, help="movement that pauses cueing, milli-g")
    ap.add_argument("--deep-max-sd", type=float, default=None,
                    help="largest usable channel sd, uV (default 60; 200 for a Sleep-EDF replay)")
    a = ap.parse_args(argv)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "study" / "participants" / a.participant / "nights" / stamp
    out.mkdir(parents=True, exist_ok=True)
    raw_f = open(out / "raw.csv", "w", newline="")
    raw_w = csv.writer(raw_f)
    raw_w.writerow(["t", "TP9", "AF7", "AF8", "TP10", "AUX", "ax", "ay", "az"])
    det_f = open(out / "detector.csv", "w", newline="")
    det_w = csv.writer(det_f)
    det_w.writerow(["t_s", "clock", "contact_ok", "calibrated", *[f"p_{s}" for s in STAGES],
                    "p_sleep", "p_deep", "stage", "would_cue"])
    ev_f = open(out / "events.csv", "w", newline="")
    ev_w = csv.writer(ev_f)
    ev_w.writerow(["t_s", "clock", "event", "detail"])
    t_start = time.time()

    def clock(t):
        return (dt.datetime.fromtimestamp(t_start + t)).strftime("%H:%M:%S")

    def event(name, detail="", t=None, show=None):
        """show: what the console prints instead of the detail (the cue log never names the puzzle)."""
        t = (sim_t if a.replay else time.time() - t_start) if t is None else t
        ev_w.writerow([f"{t:.1f}", clock(t), name, detail])
        ev_f.flush()
        txt = detail if show is None else show
        print(f"[{clock(t)}] {name}" + (f": {txt}" if txt else ""), flush=True)

    if sys.platform == "win32" and not a.replay:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        event("laptop_held_awake")

    buf = Buffer(seconds=120)
    raw_count = [0]

    def sink(t, row, acc):
        acc = acc or ["", "", ""]
        aux = buf.aux
        raw_w.writerow([f"{t:.4f}", *[f"{v:.2f}" for v in row], "" if aux is None else f"{aux:.2f}", *acc])
        raw_count[0] += 1
        if raw_count[0] % 2560 == 0:
            raw_f.flush()
    buf.sink = sink

    manifest = dict(participant=a.participant, started=dt.datetime.now().isoformat(timespec="seconds"),
                    model=Path(a.model).name, source="replay" if a.replay else "osc", args=vars(a))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    det = LiveDetector(joblib.load(a.model), step_s=5.0)

    sim_t = 0.0
    if a.replay:
        src = Replay(buf, a.replay, start_s=a.replay_start_hours * 3600.0)
        event("replay", a.replay)
    else:
        src = MuseOSC(buf, a.osc_port, log=lambda m: event("muse", m))
        src.start()
        print(f"Waiting for MuseLog on UDP {a.osc_port} (SnowballArcade format, Full-rate raw EEG on)...",
              flush=True)
        while buf.last is None or time.time() - buf.last > 2:
            time.sleep(0.5)
        event("headband_streaming")
        while True:
            n0 = buf.count
            time.sleep(5)
            rate = (buf.count - n0) / 5
            event("eeg_rate", f"{rate:.0f} Hz")
            if rate >= 200:
                break
            print("EEG under 200 Hz: turn on Full-rate raw EEG in MuseLog's OSC settings; waiting...",
                  flush=True)
        event("accelerometer", "present" if buf.accel is not None else "absent (no head angle tonight)")
        event("fifth_electrode", "present (AUX column)" if buf.aux is not None else "absent")
    cueing = None
    if a.cues:
        gate_kw = dict(thr=a.deep_thr, min_minutes=a.min_minutes, move_mg=a.move_mg,
                       max_sd=a.deep_max_sd or (200.0 if a.replay else 60.0))

        def flush_keys():
            while msvcrt and msvcrt.kbhit():
                msvcrt.getwch()

        def hearing(play):
            print("\nHEARING CHECK. Lie as you will sleep, with the speaker or earbuds where they will be "
                  "all night. Do not touch the Windows volume after this. A soft hiss plays every two "
                  "seconds, getting louder: press any key as soon as you hear it. It runs twice.",
                  flush=True)
            input("Press Enter to start the hearing check... ")
            return level_check(play, msvcrt.kbhit if msvcrt else (lambda: False), flush_keys,
                               log=lambda m: print("  " + m, flush=True))
        cueing = NightCueing(a.cues, buf, event, out, audio=not a.replay, gain_db=a.cue_gain,
                          above_db=a.cue_above, interval=a.cue_interval, jitter=a.cue_jitter,
                          gate_kw=gate_kw, level_check_fn=None if a.replay else hearing,
                          seed=0 if a.replay else None)
        manifest["cues"] = str(a.cues)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if not a.replay:
        print("Recording. Lie down, close your eyes; the detector calibrates on the first three minutes. "
              + ("Puzzle sounds will play quietly in deep sleep only. " if cueing else "Nothing will sound tonight. ")
              + "Ctrl+C ends the night.", flush=True)

    step = 5.0
    tick = 1.0 if cueing else step
    last_stream = time.time()
    first_sleep = first_n2 = None
    rem_hold = 0
    last_cue = -1e9
    was_calibrated = False
    was_connected = True
    end_t = (a.replay_hours if a.replay else a.max_hours) * 3600.0
    k = 0
    try:
        while sim_t < end_t:
            if a.replay:
                sim_t += tick
                if not src.advance_to(sim_t):
                    event("replay_end")
                    break
                t = sim_t
            else:
                time.sleep(tick)
                t = time.time() - t_start
                alive = buf.last is not None and time.time() - buf.last < 5
                if alive:
                    last_stream = time.time()
                if alive != was_connected:
                    event("stream_regained" if alive else "stream_lost")
                    was_connected = alive
                if not alive:
                    if time.time() - last_stream > a.silence_minutes * 60:
                        event("end_after_silence")
                        break
                    continue
            k += 1
            if cueing:
                cueing.tick(t)
            if k % int(round(step / tick)):
                continue
            w = buf.window(30)
            ok = bool(w and w[3])
            p = det.push(w[0], w[1], w[2]) if (w and ok) else None
            if det.calibrated and not was_calibrated:
                was_calibrated = True
                event("calibrated", "the detector's baseline is set", t)
            stage = cue = ""
            ps = pdp = None
            if p:
                ps, pdp = 1 - p["W"], p["N2"] + p["N3"]
                stage = max(STAGES, key=lambda s: p[s])
                if first_sleep is None and ps >= a.p_sleep:
                    first_sleep = t
                    event("sleep_onset_trigger", f"P(sleep) {ps:.2f} (the nap rig would cue here)", t)
                if first_n2 is None and pdp >= a.p_n2:
                    first_n2 = t
                    event("n2_guard", f"P(N2+N3) {pdp:.2f} (the nap rig would wake here)", t)
                rem_hold = rem_hold + 1 if p["REM"] >= a.rem_p else 0
                if rem_hold >= 6 and t - last_cue >= a.rem_refractory:
                    last_cue = t
                    cue = "REM"
                    event("rem_cue_would_fire", f"P(REM) {p['REM']:.2f} held 30 s", t)
            det_w.writerow([f"{t:.1f}", clock(t), int(ok), int(det.calibrated)]
                           + ([f"{p[s]:.4f}" for s in STAGES] + [f"{ps:.4f}", f"{pdp:.4f}"] if p else [""] * 7)
                           + [stage, cue])
            g = cueing.epoch(t) if cueing else None
            if int(t) % 60 < step:
                det_f.flush()
                if p and not a.replay:
                    print(f"[{clock(t)}] {stage:3s} P(sleep) {ps:.2f} P(REM) {p['REM']:.2f} contact {int(ok)}"
                          + (f" | deep {g['score']:.2f} (thr {g['thr']:.2f}) {g['state']}, {cueing.cuer.n} cues"
                             if g and np.isfinite(g["score"]) else (f" | {g['state']}" if g else "")),
                          flush=True)
    except KeyboardInterrupt:
        event("stopped_by_hand")
    finally:
        raw_f.close(); det_f.close()
        if cueing:
            c = cueing.close()
            event("cueing_summary", f"{c['cues_total']} cues over {c['minutes_cueing']} min, "
                                    f"final gain {c['final_gain_db']:.1f} dB")
        summary = summarise(out, first_sleep, first_n2)
        event("night_end", json.dumps(summary)[:200])
        ev_f.close()
        if sys.platform == "win32" and not a.replay:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    print("\nwritten", out)
    return 0


def summarise(out, first_sleep, first_n2):
    """Stage shares and 30-s epoch runs from detector.csv, like muselog_night.py."""
    import pandas as pd
    T = pd.read_csv(out / "detector.csv")
    have = T.dropna(subset=["p_W"])
    summary = dict(windows_scored=int(len(have)), minutes_scored=round(len(have) * 5 / 60, 1),
                   contact_ok_share=round(float(T.contact_ok.mean()), 3) if len(T) else None,
                   first_sleep_trigger_s=first_sleep, first_n2_guard_s=first_n2,
                   would_be_rem_cues=int((T.would_cue == "REM").sum()) if len(T) else 0)
    if len(have):
        summary["stage_share"] = have.stage.value_counts(normalize=True).round(3).to_dict()
        ep = have.assign(epoch=(have.t_s // 30).astype(int)).groupby("epoch").agg(
            clock=("clock", "first"), stage=("stage", lambda s: s.mode().iloc[0]),
            p_sleep=("p_sleep", "mean"), p_rem=("p_REM", "mean"))
        ep.to_csv(out / "epochs.csv")
        runs = (ep.stage != ep.stage.shift()).cumsum()
        lines = [f"{g.clock.iloc[0]}  {g.stage.iloc[0]:3s}  {len(g) * 0.5:5.1f} min  P(sleep) {g.p_sleep.mean():.2f}"
                 for _, g in ep.groupby(runs)]
        summary["runs"] = lines
        print("\nstage runs (30-s epochs):")
        for ln in lines[:60]:
            print("  " + ln)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    sys.exit(main())
