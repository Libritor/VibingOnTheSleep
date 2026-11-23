"""Cued against uncued, over every scored night of one participant (or of several).

The test is a randomisation test of the split that was actually used: every night's unsolved
puzzles are re-split many times with the same stratified procedure (study/puzzle_common.py), and
the observed pooled difference in morning solve rate (cued minus uncued) is compared with the
differences those re-splits give. Because the split really was random, this test is exact for one
person; it needs no assumption about people or puzzles.

  intention to treat   every scored night, whether or not any cue was played (the primary test)
  cued nights          only nights with at least --min-cues-per-puzzle cues per cued puzzle
  not heard            only nights on which the sleeper did not report hearing sounds

    python study/puzzle_analysis.py --participant P-0001
    python study/puzzle_analysis.py --power            # how many nights a real effect needs
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
from puzzle_common import stratified_split  # noqa: E402


def load(pids, include_simulated=False):
    rows, nights = [], []
    for pid in pids:
        d = STUDY / "participants" / pid / "puzzle_sessions"
        for s in sorted(d.iterdir()) if d.exists() else []:
            if not (s / "result.json").exists():
                continue
            res = json.loads((s / "result.json").read_text())
            if res.get("simulated") and not include_simulated:
                continue
            split = json.loads((s / "split.json").read_text())
            nc = json.loads((s / "night_cues.json").read_text()) if (s / "night_cues.json").exists() else {}
            cued = set(split["cued"])
            M = pd.read_csv(s / "morning.csv")
            for r in M.itertuples():
                rows.append(dict(participant=pid, session=s.name, id=r.id, family=r.family, cued=int(r.id in cued),
                                 solved=int(r.solved), cues=int(nc.get("cues_per_puzzle", {}).get(r.id, 0))))
            nights.append(dict(participant=pid, session=s.name, heard=res["heard_sounds"],
                               cues_total=nc.get("cues_total", 0), minutes_cueing=nc.get("minutes_cueing", 0),
                               cued_n=res["cued_n"], uncued_n=res["uncued_n"],
                               cued_solved=res["cued_solved"], uncued_solved=res["uncued_solved"]))
    return pd.DataFrame(rows), pd.DataFrame(nights)


def diff(T):
    c, u = T[T.cued == 1], T[T.cued == 0]
    if not len(c) or not len(u):
        return np.nan
    return c.solved.mean() - u.solved.mean()


def randomisation(T, draws=20000, seed=1):
    """Observed pooled difference, its re-split distribution, one- and two-sided p."""
    obs = diff(T)
    groups = [(g.id.tolist(), g.family.tolist(), dict(zip(g.id, g.solved)))
              for _, g in T.groupby(["participant", "session"])]
    rng = random.Random(seed)
    null = np.empty(draws)
    for k in range(draws):
        sc = nc = su = nu = 0
        for ids, fams, solved in groups:
            cu, un = stratified_split(list(zip(ids, fams)), rng)
            sc += sum(solved[i] for i in cu)
            nc += len(cu)
            su += sum(solved[i] for i in un)
            nu += len(un)
        null[k] = (sc / nc if nc else 0) - (su / nu if nu else 0)
    p1 = (1 + np.sum(null >= obs - 1e-12)) / (draws + 1)
    p2 = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (draws + 1)
    return obs, null, float(p1), float(p2)


def power(p_uncued=0.205, p_cued=0.317, per_night=13, nights=(5, 10, 15, 20, 30), studies=1000, perms=400, seed=3):
    """Share of simulated studies in which the one-sided randomisation test reaches p < 0.05."""
    rng = np.random.default_rng(seed)
    out = {}
    h = per_night // 2
    for n in nights:
        hits = 0
        for _ in range(studies):
            lab = np.zeros((n, per_night), bool)
            lab[:, :h] = True
            solved = np.where(lab, rng.random((n, per_night)) < p_cued, rng.random((n, per_night)) < p_uncued)
            obs = solved[lab].mean() - solved[~lab].mean()
            perm = np.argsort(rng.random((perms, n, per_night)), axis=2) < h
            s = solved[None]
            null = (s * perm).sum((1, 2)) / perm.sum((1, 2)) - (s * ~perm).sum((1, 2)) / (~perm).sum((1, 2))
            hits += (1 + np.sum(null >= obs - 1e-12)) / (perms + 1) < 0.05
        out[n] = round(hits / studies, 2)
    return out


def figure(T, N, obs, null, p1, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    x = np.arange(len(N))
    pc = N.cued_solved / N.cued_n.replace(0, np.nan)
    pu = N.uncued_solved / N.uncued_n.replace(0, np.nan)
    ax[0].plot(x, pc, "o-", color="#1f4e79", lw=1.5, ms=5, label="cued in slow-wave sleep")
    ax[0].plot(x, pu, "s--", color="#9a9a9a", lw=1.5, ms=5, label="not cued")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([s[4:8] + "\n" + f"{int(c)} cues" for s, c in zip(N.session, N.cues_total)], fontsize=6)
    ax[0].set_ylim(-0.03, 1.03)
    ax[0].set_ylabel("solved next morning\n(of puzzles unsolved the evening before)")
    ax[0].legend(frameon=False, fontsize=7, loc="upper left")
    ax[0].spines[["top", "right"]].set_visible(False)
    ax[1].hist(null, bins=41, color="#c8c8c8")
    ax[1].axvline(obs, color="#1f4e79", lw=2)
    ax[1].text(obs, ax[1].get_ylim()[1] * 0.92, f"  observed {obs:+.2f}\n  p = {p1:.3f} (one-sided)", color="#1f4e79",
               fontsize=7, va="top")
    ax[1].set_xlabel("cued minus uncued solve rate")
    ax[1].set_ylabel("re-splits")
    ax[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(Path(path).with_suffix(".png"), dpi=150)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", nargs="*", default=[])
    ap.add_argument("--min-cues-per-puzzle", type=float, default=3.0)
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--include-simulated", action="store_true")
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.power:
        print("Power of the one-sided randomisation test at p < 0.05, Sanders 2019 rates (31.7% cued, 20.5% uncued):")
        for k in (10, 13, 16):
            print(f"  {k} unsolved puzzles a night: nights -> power {power(per_night=k)}")
        if not a.participant:
            return 0
    T, N = load(a.participant, a.include_simulated)
    if T.empty:
        sys.exit("No scored nights yet (a night is scored once study/puzzle_morning.py has run).")
    report = {"nights": len(N), "puzzles": len(T)}
    for name, keep in (("intention_to_treat", N.session),
                       ("cued_nights", N.session[N.cues_total >= a.min_cues_per_puzzle * N.cued_n.clip(lower=1)]),
                       ("not_heard", N.session[N.heard != "y"])):
        S = T[T.session.isin(set(keep))]
        if S.empty or S.cued.nunique() < 2:
            report[name] = {"nights": int(len(set(keep))), "note": "nothing to compare"}
            continue
        obs, null, p1, p2 = randomisation(S, a.draws)
        report[name] = dict(nights=int(S.session.nunique()), cued=f"{S[S.cued == 1].solved.sum()}/{(S.cued == 1).sum()}",
                            uncued=f"{S[S.cued == 0].solved.sum()}/{(S.cued == 0).sum()}",
                            difference=round(float(obs), 3), p_one_sided=round(p1, 4), p_two_sided=round(p2, 4))
        if name == "intention_to_treat":
            fig_args = (S, N, obs, null, p1)
    by_fam = {f: dict(cued=f"{g[g.cued == 1].solved.sum()}/{(g.cued == 1).sum()}",
                      uncued=f"{g[g.cued == 0].solved.sum()}/{(g.cued == 0).sum()}") for f, g in T.groupby("family")}
    report["by_family"] = by_fam
    out = Path(a.out) if a.out else STUDY / "participants" / (a.participant[0] if len(a.participant) == 1 else "pooled") / "puzzle_results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "puzzle_results.json").write_text(json.dumps(report, indent=1))
    T.to_csv(out / "puzzle_rows.csv", index=False)
    N.to_csv(out / "puzzle_nights.csv", index=False)
    if "fig_args" in locals():
        figure(*fig_args, out / "puzzle_cueing.pdf")
    print(json.dumps(report, indent=1))
    print("wrote", out)
    return report


if __name__ == "__main__":
    main()
