"""First-deployment analysis of the live rig on a real Muse (MuseLog over OSC), one person.

Reads every session folder of a participant and reports, per session:
  * link: delivered EEG rate, share of 1-s blocks within 5% of 256 Hz, longest arrival gap
  * contact: share of 30-s windows passing the contact rule
  * replay: the detector re-run offline on raw.csv with the rig's own schedule (calibration from
    30 s to 210 s after calibration_start, one window every 5 s) against the probabilities the rig
    logged live, as a check that the deployed code computed what the offline code computes
  * physiology the classifier does not see on its own: alpha (8-12 Hz) relative power and slow eye
    movement power on the derived channels, against the session's calibration median
  * awake specificity: P(sleep) on windows where the protocol says the person was awake (the typed
    task, eyes open), scored with that session's calibration
  * posture: head angle against calibration, and the nod rule (drop against 10-40 s earlier)

No condition (cue or sham) is read or printed.

    python study/pilot_analysis.py --participant P-0001
"""
import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "detector"))
sys.path.insert(0, str(ROOT / "rig"))
from features import band_feats, eog_feats  # noqa: E402
from model import LiveDetector  # noqa: E402
from sources import contact_ok, derive  # noqa: E402

FS = 256.0
STEP = 5.0


def load(d):
    ev = pd.read_csv(d / "events.csv")
    raw = pd.read_csv(d / "raw.csv")
    # raw.t is wall-clock arrival time; the first sample arrives at the 'headband_streaming' event
    hs = ev.loc[ev.event == "headband_streaming", "t"]
    raw["rel"] = raw.t - raw.t.iloc[0] + (float(hs.iloc[0]) if len(hs) else 0.0)
    det = pd.read_csv(d / "detector.csv") if (d / "detector.csv").stat().st_size else pd.DataFrame()
    return ev, raw, det


def t_of(ev, name):
    x = ev.loc[ev.event == name, "t"]
    return float(x.iloc[0]) if len(x) else None


def window(raw, end, sec=30.0):
    """The last `sec` seconds of samples that had arrived by `end` (what the live buffer held)."""
    r = raw[raw.rel <= end].tail(int(sec * FS))
    if len(r) < int(sec * FS):
        return None
    return r[["TP9", "AF7", "AF8", "TP10"]].to_numpy(float), r[["ax", "ay", "az"]].to_numpy(float)


def link_stats(raw):
    t = raw.t.to_numpy()
    dur = t[-1] - t[0]
    blocks = np.bincount((t - t[0]).astype(int))[:-1]       # drop the partial last second
    return {"seconds": round(float(dur), 1), "rate_hz": round(len(t) / dur, 1),
            "blocks_within_5pct": round(float(np.mean(np.abs(blocks - FS) <= 0.05 * FS)), 3),
            "max_gap_s": round(float(np.diff(t).max()), 3)}


def angle(a, b):
    c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def analyse(d, bundle):
    ev, raw, det = load(d)
    out = {"session": d.name, "events": ev[["t", "event"]].assign(t=ev.t.round(1)).values.tolist(),
           "link": link_stats(raw)}
    cs = t_of(ev, "calibration_start")
    task0 = t_of(ev, "headband_streaming")
    task1 = t_of(ev, "aut_pre_done")
    end = float(raw.rel.iloc[-1])
    if cs is None or end < cs + 210:
        out["note"] = "no complete calibration; link statistics only"
        return out, None

    # the rig's schedule: windows at calibration_start + 35, 40, ... (the first 30 s are skipped)
    ends = np.arange(cs + 35.0, end + 1e-6, STEP)
    lad = LiveDetector(bundle, step_s=STEP)
    rows, calib_feats = [], []
    for e in ends:
        w = window(raw, e)
        if w is None:
            continue
        rows4, acc = w
        eeg, eog = derive(rows4)
        ok = contact_ok(rows4, FS)
        b = band_feats(eeg - eeg.mean(), FS)
        o = eog_feats(eog - eog.mean(), FS)
        p = lad.push(eeg, eog, FS) if ok else None
        if not lad.calibrated:
            calib_feats.append((b["rel_alpha"], o["log_sem"]))
        rows.append({"t": round(e, 1), "contact_ok": ok, "rel_alpha": b["rel_alpha"],
                     "alpha_theta": b["alpha_theta"], "log_sem": o["log_sem"],
                     "acc": np.nanmedian(acc[-256:], axis=0),
                     **({f"p_{k}": v for k, v in p.items()} if p else {})})
    T = pd.DataFrame(rows)
    calib_end = cs + 210.0
    cal = T[T.t <= calib_end + 0.1]
    a0, s0 = cal.rel_alpha.median(), cal.log_sem.median()
    T["alpha_vs_calib"] = T.rel_alpha / a0
    T["sem_vs_calib_db"] = 10 * (T.log_sem - s0)
    T["p_sleep"] = 1 - T.get("p_W", np.nan)

    # posture: angle against the calibration posture, and the nod rule
    up = np.median(np.stack(cal.acc.to_numpy()), axis=0)
    T["angle_vs_calib"] = [angle(a, up) for a in T.acc]
    sec = raw.groupby(raw.rel.astype(int))[["ax", "ay", "az"]].mean().dropna()
    nod = []
    for t in T.t:
        ref = sec.loc[(sec.index >= t - 40) & (sec.index <= t - 10)]
        cur = sec.loc[sec.index <= t].tail(1)
        nod.append(angle(cur.to_numpy()[0], ref.median().to_numpy()) if len(ref) >= 20 and len(cur) else np.nan)
    T["nod_rel"] = nod

    # replay against the live log
    live = det.dropna(subset=["p_W"]) if len(det) else pd.DataFrame()
    if len(live):
        m = pd.merge_asof(live.sort_values("t"), T[["t", "p_W"]].dropna().sort_values("t"),
                          on="t", direction="nearest", tolerance=1.0, suffixes=("_live", "_offline"))
        m = m.dropna(subset=["p_W_offline"])
        out["replay"] = {"windows": int(len(m)),
                         "max_abs_diff_pW": round(float((m.p_W_live - m.p_W_offline).abs().max()), 4),
                         "median_abs_diff_pW": round(float((m.p_W_live - m.p_W_offline).abs().median()), 4)}

    # awake specificity on the typed task (eyes open, typing): same calibration, separate pass
    if task1 is not None and task1 - task0 >= 60:
        lad2 = LiveDetector(bundle, step_s=STEP)
        for e in ends[ends <= calib_end + 0.1]:
            w = window(raw, e)
            if w is not None and contact_ok(w[0], FS):
                lad2.push(*derive(w[0]), FS)
        ps = []
        for e in np.arange(task0 + 30.0, task1 + 1e-6, STEP):
            w = window(raw, e)
            if w is None or not contact_ok(w[0], FS):
                continue
            p = lad2.push(*derive(w[0]), FS)
            if p:
                ps.append(1 - p["W"])
        if ps:
            ps = np.array(ps)
            out["awake_task"] = {"windows": int(len(ps)), "median_p_sleep": round(float(np.median(ps)), 3),
                                 "max_p_sleep": round(float(ps.max()), 3),
                                 "false_eeg_triggers_at_0.7": int((ps >= 0.7).sum())}

    post = T[T.t > calib_end]
    out["contact_ok_share"] = round(float(T.contact_ok.mean()), 3)
    out["post_calibration"] = {
        "windows": int(len(post)),
        "first_p_sleep_ge_0.7_s_after_calibration":
            (round(float(post.t[post.p_sleep >= 0.7].iloc[0] - calib_end), 1)
             if (post.p_sleep >= 0.7).any() else None),
        "max_angle_vs_calib_deg": round(float(post.angle_vs_calib.max()), 1) if len(post) else None,
        "max_nod_rel_deg": round(float(post.nod_rel.max()), 1) if post.nod_rel.notna().any() else None,
        "nod_rule_hits": int(((post.nod_rel >= 20) & (post.p_sleep >= 0.3)).sum()),
        "old_rule_hits": int((post.angle_vs_calib >= 20).sum()),
    }
    trig = t_of(ev, "trigger")
    if trig is not None:
        pre = T[(T.t > calib_end) & (T.t <= calib_end + 120)]
        at = T[(T.t >= trig - 0.1) & (T.t <= trig + 60)]
        out["at_trigger"] = {
            "trigger_s_after_calibration": round(trig - calib_end, 1),
            "alpha_vs_calib_first_2min_median": round(float(pre.alpha_vs_calib.median()), 2),
            "alpha_vs_calib_hold_median": round(float(at.alpha_vs_calib.median()), 2),
            "sem_db_first_2min_median": round(float(pre.sem_vs_calib_db.median()), 1),
            "sem_db_hold_median": round(float(at.sem_vs_calib_db.median()), 1),
            "p_sleep_hold_median": round(float(at.p_sleep.median()), 2),
            "argmax_stage_hold": at[[c for c in at.columns if c.startswith("p_") and c[2:] in
                                     ("W", "N1", "N2", "N3", "REM")]].idxmax(axis=1).str[2:]
                                 .value_counts().to_dict(),
        }
    T["session"] = d.name
    T["t_from_calibration_end"] = T.t - calib_end
    return out, T.drop(columns=["acc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    a = ap.parse_args()
    pdir = ROOT / "study" / "participants" / a.participant
    bundle = joblib.load(ROOT / "models" / "nap_base3min.joblib")
    results, tables = [], []
    for d in sorted((pdir / "sessions").iterdir()):
        if not (d / "raw.csv").exists():
            continue
        r, T = analyse(d, bundle)
        results.append(r)
        if T is not None:
            tables.append(T)
        print(json.dumps({k: v for k, v in r.items() if k != "events"}, indent=1))
    outdir = pdir / "pilot"
    outdir.mkdir(exist_ok=True)
    (outdir / "pilot_results.json").write_text(json.dumps(results, indent=1))
    if tables:
        pd.concat(tables).to_csv(outdir / "pilot_timeline.csv", index=False)
    print("wrote", outdir)


if __name__ == "__main__":
    main()
