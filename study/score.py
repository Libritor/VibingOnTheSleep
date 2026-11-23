"""Blind scoring of nap reports and alternative-uses lists.

The rater (a person, and optionally a language model) sees each nap's object, the transcribed
reports and the post-nap uses, under a random item ID. The rater never sees the condition: cue and
sham naps both involve the object, because the object was worked on before every nap. The key that
maps item IDs to conditions is written to a separate file the rater must not open.

    python study/score.py --participant P-0001 make-sheet     # blind_sheet.csv + key.csv
    python study/score.py --participant P-0001 llm            # llm_ratings.csv (needs consent)
    (a person fills the rating columns of blind_sheet.csv and saves it as human_ratings.csv)

Ratings per item:
  incorporation 0-3: 0 nothing about the object; 1 a loose association (its material, shape or
                     use) without naming it; 2 the object named or clearly present; 3 the object
                     central to an extended dream scene or line of thought
  For each post-nap use not already on the pre-nap list: originality 1-5 (1 common, 5 surprising
  and workable). Item score = number of new uses and their mean originality.
"""
import argparse
import csv
import json
import re
import secrets
from pathlib import Path

STUDY = Path(__file__).resolve().parent


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def read_uses(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [r["use"] for r in csv.DictReader(f)]


def collect(pid):
    items = []
    for man in sorted((STUDY / "participants" / pid / "sessions").glob("*/manifest.json")):
        m = json.loads(man.read_text())
        if not m.get("completed"):
            continue
        d = man.parent
        reports = []
        for k in range(1, m.get("cycles", 0) + 1):
            t = d / f"report_{k}.txt"
            reports.append(t.read_text(encoding="utf-8").strip() if t.exists() else "")
        pre = read_uses(d / "aut_pre.csv")
        post = read_uses(d / "aut_post.csv")
        seen = {norm(u) for u in pre}
        new = [u for u in post if norm(u) and norm(u) not in seen]
        items.append({"session": d.name, "nap_index": m["nap_index"], "condition": m["condition"],
                      "object": m["object"], "reports": reports, "pre_uses": pre, "new_uses": new})
    return items


def make_sheet(pid):
    out = STUDY / "participants" / pid / "scoring"
    out.mkdir(exist_ok=True)
    items = collect(pid)
    ids = {it["session"]: secrets.token_hex(3) for it in items}
    order = sorted(items, key=lambda it: ids[it["session"]])
    with open(out / "blind_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "object", "reports", "new_uses", "incorporation_0_3",
                    "originality_1_5_per_new_use"])
        for it in order:
            w.writerow([ids[it["session"]], it["object"], " || ".join(it["reports"]),
                        " || ".join(it["new_uses"]), "", ""])
    with open(out / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "session", "nap_index", "condition", "object"])
        for it in order:
            w.writerow([ids[it["session"]], it["session"], it["nap_index"], it["condition"], it["object"]])
    print(f"{len(items)} items. Give the rater {out / 'blind_sheet.csv'} only; key.csv stays closed.")


SCHEMA = {
    "type": "object",
    "properties": {
        "incorporation": {"type": "integer", "enum": [0, 1, 2, 3]},
        "incorporation_reason": {"type": "string"},
        "originality": {"type": "array", "items": {"type": "integer", "enum": [1, 2, 3, 4, 5]}},
    },
    "required": ["incorporation", "incorporation_reason", "originality"],
    "additionalProperties": False,
}

PROMPT = """You are scoring one item from a sleep-onset study. Score only what is written; do not guess.

Object: {object}

What the person reported thinking about when woken from sleep onset (several wakings separated by ||):
{reports}

New uses for the object they listed after the nap, one per line:
{uses}

1. incorporation, 0 to 3. 0: nothing about the object. 1: a loose association with the object (its
material, shape or typical use) without naming it. 2: the object named or clearly present. 3: the
object central to an extended scene or line of thought. Give a one-sentence reason quoting the report.
2. originality: one integer per listed use, in order, 1 to 5 (1 common, 5 surprising and workable).
Return an empty list if there are no uses."""


def llm(pid, model):
    consent = json.loads((STUDY / "participants" / pid / "consent.json").read_text())
    if not consent.get("llm_scoring_allowed"):
        raise SystemExit(f"{pid} did not agree to language-model scoring; use a person.")
    import anthropic
    client = anthropic.Anthropic()
    out = STUDY / "participants" / pid / "scoring"
    with open(out / "blind_sheet.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    results = []
    for r in rows:
        uses = [u for u in r["new_uses"].split(" || ") if u]
        resp = client.beta.messages.create(
            model=model, max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": PROMPT.format(
                object=r["object"], reports=r["reports"] or "(no report)",
                uses="\n".join(uses) or "(none)")}],
        )
        if resp.stop_reason == "refusal":
            results.append({"item": r["item"], "incorporation": "", "originality": "", "reason": "refused"})
            continue
        data = json.loads(next(b.text for b in resp.content if b.type == "text"))
        orig = data["originality"][:len(uses)]
        results.append({"item": r["item"], "incorporation": data["incorporation"],
                        "originality": ";".join(map(str, orig)), "reason": data["incorporation_reason"]})
        print(r["item"], data["incorporation"], orig)
    with open(out / "llm_ratings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item", "incorporation", "originality", "reason"])
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {out / 'llm_ratings.csv'} ({model}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("action", choices=("make-sheet", "llm"))
    ap.add_argument("--model", default="claude-opus-5-5")
    a = ap.parse_args()
    if a.action == "make-sheet":
        make_sheet(a.participant)
    else:
        llm(a.participant, a.model)


if __name__ == "__main__":
    main()
