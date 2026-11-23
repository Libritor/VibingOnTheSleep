"""Run the slow-wave cue gate (rig/sws_cue.py) offline, exactly as the night rig would, and say
where its cues would have landed.

  Sleep-EDF   the scored stage (W, N1, N2, N3, REM) at every would-be cue, and the share of
              scored N3 the gate covered. AF7 and AF8 are built from Fpz-Cz and the horizontal EOG
              the same way the replay source builds them, so the montage is not the Muse's: this
              checks the logic and the timing, not the threshold.
  MuseLog     a phone recording converted by muselog_load.py (AF7, AF8 and the Oz cup on the
              REF column, accelerometer for movement), compared with the rule-based state sketch
              of night_overview.py when one is given.

    python study/sws_gate_sim.py edf data/sc/SC4001E0-PSG.edf data/sc/SC4001EC-Hypnogram.edf --start-hours 8 --hours 4
    python study/sws_gate_sim.py edf-all data/sc --hours 6 --out detector    # every night, from 30 min before sleep
    python study/sws_gate_sim.py npz data/muselog/20261028/muse_session_20261028_034217_Muse1.npz --device-clock \
        --sketch study/participants/P-0001/nights/phone_20261028/state_sketch.csv --anchor "2026-10-28 03:42:00"         --start "2026-10-28 03:51:00"
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rig"))
from sws_cue import Cuer, DeepGate  # noqa: E402

STAGE = {"Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2", "Sleep stage 3": "N3",
         "Sleep stage 4": "N3", "Sleep stage R": "REM", "Movement time": "MT", "Sleep stage ?": "?"}


def run(fs, chans_at, move_at, t_end, gate_kw, interval=12.0, jitter=2.0, t_begin=30.0):
    """chans_at(t, seconds) -> {name: raw array} ending at t; move_at(t) -> milli-g or None."""
    gate = DeepGate(fs, **gate_kw)
    cues, pauses = [], collections.Counter()
    cuer = Cuer({"X": np.zeros(1, np.float32)}, -30.0, -42.0, play=lambda y: None,
                log=lambda n, d, show=None: cues.append(t_now[0]) if n == "cue" else None,
                interval=interval, jitter=jitter, seed=0)
    t_now = [0.0]
    states = []
    t = max(30.0, float(np.ceil(t_begin)))
    while t <= t_end:
        t_now[0] = t
        reason = gate.fast(t, chans_at(t, 10), move_at(t))
        if reason and gate.state != "cueing":
            pass
        if reason:
            pauses[reason.split(" ")[0]] += 1
        cuer.tick(t, gate.state == "cueing", reason)
        if int(round(t)) % 5 == 0:
            r = gate.epoch(t, chans_at(t, 30))
            states.append(dict(t_s=t, score=r["score"], alpha=r["alpha"], eye=r["eye"], amp=r["amp"], thr=r["thr"], state=r["state"],
                               n_good=r["n_good"]))
        t += 1.0
    return pd.DataFrame(states), np.array(cues), pauses, gate


def first_sleep_hours(hyp_path):
    import mne
    mne.set_log_level("ERROR")
    ann = mne.read_annotations(hyp_path)
    for on, de in zip(ann.onset, ann.description):
        if STAGE.get(de, "?") in ("N1", "N2", "N3", "REM"):
            return on / 3600.0
    return 0.0


def edf_all(a, gate_kw):
    """Every PSG/hypnogram pair in a folder, each from 30 min before its first scored sleep."""
    d = Path(a.folder)
    rows = []
    for psg in sorted(d.glob("SC4*E0-PSG.edf")):
        hyp = next(iter(sorted(d.glob(psg.name[:6] + "*-Hypnogram.edf"))), None)
        if hyp is None:
            continue
        b = argparse.Namespace(psg=str(psg), hyp=str(hyp), hours=a.hours,
                               start_hours=max(0.0, first_sleep_hours(hyp) - 0.5))
        r, _ = edf(b, gate_kw)
        print(f"{psg.name}: {r['cues']} cues, {r['share_in_N3']:.0%} in N3, {r['share_in_N2_N3']:.0%} in N2/N3, "
              f"{r['stage_at_cue'].get('W', 0)} in W, {r['stage_at_cue'].get('REM', 0)} in REM; "
              f"N3 covered {r['share_of_N3_cued']}", flush=True)
        rows.append(r)
    tot = collections.Counter()
    for r in rows:
        tot.update(r["stage_at_cue"])
    n = sum(tot.values())
    cov = [r["share_of_N3_cued"] for r in rows if r["share_of_N3_cued"] is not None]
    summary = dict(nights=len(rows), hours_per_night=a.hours, cues=n, stage_at_cue=dict(tot),
                   share_in_N3=round(tot["N3"] / max(1, n), 3), share_in_N2_N3=round((tot["N2"] + tot["N3"]) / max(1, n), 3),
                   share_in_W=round(tot["W"] / max(1, n), 4), share_in_REM=round(tot["REM"] / max(1, n), 4),
                   nights_with_cues=sum(r["cues"] > 0 for r in rows),
                   median_share_of_N3_cued=round(float(np.median(cov)), 3) if cov else None,
                   per_night=rows)
    return summary, pd.DataFrame([{k: v for k, v in r.items() if k not in ("stage_at_cue", "pauses")} for r in rows])


def edf(a, gate_kw):
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(a.psg, include=["EEG Fpz-Cz", "EOG horizontal"], preload=False)
    fs = raw.info["sfreq"]
    t1 = min(raw.times[-1], a.start_hours * 3600 + a.hours * 3600)
    raw.crop(a.start_hours * 3600, t1).load_data()               # only the slice, not the whole night
    e = raw.get_data(picks=0)[0] * 1e6
    o = raw.get_data(picks=1)[0] * 1e6
    i0, i1 = 0, len(e)
    af7 = 800 + e + o / 2
    af8 = 800 + e - o / 2
    del raw, e, o

    def chans_at(t, sec):
        j = int(t * fs)
        n = int(sec * fs)
        return {"AF7": af7[j - n:j], "AF8": af8[j - n:j]}
    S, cues, pauses, gate = run(fs, chans_at, lambda t: None, (i1 - i0) / fs, gate_kw)
    ann = mne.read_annotations(a.hyp)
    hyp = []
    for on, du, de in zip(ann.onset, ann.duration, ann.description):
        hyp += [STAGE.get(de, "?")] * int(du // 30)

    def stage_at(t_rel):
        k = int((a.start_hours * 3600 + t_rel) // 30)
        return hyp[k] if k < len(hyp) else "?"
    at = collections.Counter(stage_at(t) for t in cues)
    k0 = int(a.start_hours * 3600 // 30)
    n3 = [k for k in range(k0, k0 + int(a.hours * 120)) if k < len(hyp) and hyp[k] == "N3"
          and (k - k0) * 30 >= gate_kw.get("min_minutes", 20) * 60]
    cueing_ep = set()
    for r in S[S.state == "cueing"].itertuples():
        cueing_ep.add(k0 + int(r.t_s // 30))
    cover = (sum(k in cueing_ep for k in n3) / len(n3)) if n3 else None
    return dict(source=Path(a.psg).name, start_hours=a.start_hours, hours=a.hours, cues=int(len(cues)),
                stage_at_cue={k: v for k, v in sorted(at.items())},
                share_in_N2_N3=round((at["N2"] + at["N3"]) / max(1, len(cues)), 3),
                share_in_N3=round(at["N3"] / max(1, len(cues)), 3),
                share_in_W_REM_N1=round((at["W"] + at["REM"] + at["N1"]) / max(1, len(cues)), 3),
                scored_N3_minutes=len(n3) / 2, share_of_N3_cued=None if cover is None else round(cover, 3),
                threshold=round(gate.thr_eff, 3), pauses=dict(pauses)), S


def npz(a, gate_kw):
    z = np.load(a.npz, allow_pickle=True)
    eeg = z["eeg"]                     # float32 (n, 6); windows are cast to float64 one at a time
    fs = 256.0
    t = np.arange(len(eeg), dtype=np.float64) / fs if a.device_clock else z["t"]
    acc = z["acc"]
    if len(acc):
        acc_t = np.arange(len(acc)) / fs if (a.device_clock and len(acc) == len(eeg)) else acc[:, 0]
        mag = np.linalg.norm(acc[:, 1:4], axis=1)
    clock0 = pd.Timestamp(str(z["clock0"]))
    anchor = pd.Timestamp(a.anchor) if a.anchor else clock0
    off = (clock0 - anchor).total_seconds()
    oz = eeg[:, 5]
    use_oz = bool(np.isfinite(oz).all() and oz[::256].std() > 0)

    def chans_at(tt, sec):
        j = np.searchsorted(t, tt)
        n = int(sec * fs)
        if j < n:
            return {}
        d = {"AF7": eeg[j - n:j, 1].astype(float), "AF8": eeg[j - n:j, 2].astype(float)}
        if use_oz:
            d["Oz"] = oz[j - n:j].astype(float)
        return d

    def move_at(tt):
        if not len(acc):
            return None
        i, j = np.searchsorted(acc_t, tt - 10), np.searchsorted(acc_t, tt)
        m = mag[i:j]
        m = m[np.isfinite(m)]
        return float(m.std() * 1000) if len(m) > 10 else None
    t_begin = (pd.Timestamp(a.start) - clock0).total_seconds() if a.start else 30.0
    S, cues, pauses, gate = run(fs, chans_at, move_at, float(t[-1]), gate_kw, t_begin=t_begin)
    S["t_anchor"] = S.t_s + off
    out = dict(source=Path(a.npz).name, cues=int(len(cues)), threshold=round(gate.thr_eff, 3),
               wake_delta=round(float(np.median([b[0] for b in gate.base])), 3) if gate.base else None,
               minutes_cueing=round(float((S.state == "cueing").sum() * 5 / 60), 1), pauses=dict(pauses),
               first_cue=(anchor + pd.Timedelta(seconds=float(cues[0] + off))).strftime("%H:%M:%S") if len(cues) else None,
               last_cue=(anchor + pd.Timedelta(seconds=float(cues[-1] + off))).strftime("%H:%M:%S") if len(cues) else None)
    if a.sketch:
        K = pd.read_csv(a.sketch)
        st = K.set_index((K.t_s // 30).astype(int)).state
        at = collections.Counter(st.get(int((c + off) // 30), "none") for c in cues)
        out["sketch_state_at_cue"] = dict(at)
    return out, S


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    e = sub.add_parser("edf")
    e.add_argument("psg")
    e.add_argument("hyp")
    e.add_argument("--start-hours", type=float, default=0.0)
    e.add_argument("--hours", type=float, default=4.0)
    ea = sub.add_parser("edf-all")
    ea.add_argument("folder")
    ea.add_argument("--hours", type=float, default=6.0)
    n = sub.add_parser("npz")
    n.add_argument("npz")
    n.add_argument("--device-clock", action="store_true")
    n.add_argument("--sketch")
    n.add_argument("--anchor")
    n.add_argument("--start", help="wall clock the night starts at (lights out), e.g. '2026-10-28 03:51:00'")
    for p in (e, ea, n):
        p.add_argument("--deep-thr", type=float, default=0.70)
        p.add_argument("--min-minutes", type=float, default=20.0)
        p.add_argument("--max-sd", type=float)
        p.add_argument("--resume-s", type=float, default=60.0)
        p.add_argument("--out")
    a = ap.parse_args(argv)
    gate_kw = dict(thr=a.deep_thr, min_minutes=a.min_minutes, resume_s=a.resume_s,
                   max_sd=a.max_sd or (60.0 if a.kind == "npz" else 200.0))
    res, S = {"edf": edf, "edf-all": edf_all, "npz": npz}[a.kind](a, gate_kw)
    print(json.dumps({k: v for k, v in res.items() if k != "per_night"}, indent=1))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        stem = "sleepedf_all" if a.kind == "edf-all" else Path(a.psg if a.kind == "edf" else a.npz).stem
        (Path(a.out) / f"gate_{stem}.json").write_text(json.dumps(res, indent=1))
        S.to_csv(Path(a.out) / f"gate_{stem}.csv", index=False)
    return res


if __name__ == "__main__":
    main()
