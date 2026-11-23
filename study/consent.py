"""Walk a participant through CONSENT.md and record the answers. The rig will not run without it.

    python study/consent.py --participant P-0001 --reb-code <REB number>
    python study/consent.py --participant P-0001 --self       # investigator self-experiment
    python study/consent.py --participant P-0001 --withdraw   # stop, and mark data for deletion
    python study/consent.py --participant P-0001 --self --study puzzles   # the overnight puzzle study
    python study/consent.py --participant P-0001 --self --study vibe      # Vibing on the Sleep

Names never go in this file or anywhere in the study folder; the pseudonym is the only key.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent

ITEMS = [
    ("read", "I have read the information sheet (study/CONSENT.md) and had my questions answered."),
    ("voluntary", "I understand taking part is voluntary and I can stop any nap, or the whole study, "
                  "at any time without giving a reason."),
    ("sounds", "I agree to hear spoken prompts and tones while I fall asleep, and to be woken by them."),
    ("eeg", "I agree to wear a Muse headband that records my brain activity, eye movements and head "
            "position during the nap."),
    ("voice", "I agree to my spoken reports of what I was thinking being recorded as audio and "
              "transcribed to text."),
    ("ai", "I agree to the transcripts (never the audio, never my name) being scored by a language "
           "model as well as by a person. I may say no to this and still take part."),
    ("storage", "I understand my data are stored under a pseudonym on the researcher's computer, "
                "not on any website or blockchain, and that I can ask for them to be deleted."),
    ("not_medical", "I understand this is not a medical device or treatment, and that I should not "
                    "take part if I have a sleep disorder, epilepsy, or am on sedating medication."),
]
OPTIONAL = {"ai"}

# the overnight puzzle-cueing study (CONSENT_PUZZLES.md): its own sheet, items and record file
PUZZLE_ITEMS = [
    ("read", "I have read the information sheet (study/CONSENT_PUZZLES.md) and had my questions answered."),
    ("voluntary", "I understand taking part is voluntary and I can stop any night, or the whole study, "
                  "at any time without giving a reason."),
    ("sounds", "I agree to quiet sounds being played to me during deep sleep, and understand they "
               "may sometimes wake me or lighten my sleep."),
    ("eeg", "I agree to wear a Muse headband that records my brain activity, eye movements and head "
            "position through the night."),
    ("puzzles", "I agree to my typed puzzle answers, their timing and my morning answers being recorded."),
    ("storage", "I understand my data are stored under a pseudonym on the researcher's computer, "
                "not on any website or blockchain, and that I can ask for them to be deleted."),
    ("not_medical", "I understand this is not a medical device or treatment, and that I should not "
                    "take part if I have a sleep disorder, epilepsy, or am on sedating medication."),
]
# Vibing on the Sleep (CONSENT_VIBE.md): the night in which a coding agent works from the sleeper's words
VIBE_ITEMS = [
    ("read", "I have read the information sheet (study/CONSENT_VIBE.md) and had my questions answered."),
    ("voluntary", "I understand taking part is voluntary and I can stop any night, or the whole study, "
                  "at any time without giving a reason."),
    ("wakes", "I agree to be woken at sleep onset the number of times I set each night (one by default), "
              "to hear whispered questions when I am asleep or have woken on my own, and to quiet sounds "
              "in deep sleep, and understand these may lighten or shorten my sleep."),
    ("eeg", "I agree to wear a Muse headband that records my brain activity, eye movements, muscle "
            "activity and head position through the night."),
    ("voice", "I agree to my spoken reports, and the twelve seconds after a night-time question, being "
              "recorded as audio on this computer and transcribed to text on this computer."),
    ("agent", "I agree to the text of what I say (never the audio, never the headband data) being sent to "
              "an AI coding agent run by an outside company, which writes code in a separate folder; "
              "nothing it writes is merged, published or sent anywhere unless I do that myself, awake."),
    ("storage", "I understand my data are stored under a pseudonym on the researcher's computer, "
                "not on any website or blockchain, and that I can ask for them to be deleted."),
    ("not_medical", "I understand this is not a medical device or treatment, and that I should not "
                    "take part if I have a sleep disorder, epilepsy, or am on sedating medication."),
]
STUDIES = {"naps": ("CONSENT.md", ITEMS, "consent.json"),
           "puzzles": ("CONSENT_PUZZLES.md", PUZZLE_ITEMS, "consent_puzzles.json"),
           "vibe": ("CONSENT_VIBE.md", VIBE_ITEMS, "consent_vibe.json")}


def ask(text):
    while True:
        r = input(f"\n{text}\n  [y/n] ").strip().lower()
        if r in ("y", "n"):
            return r == "y"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True)
    ap.add_argument("--reb-code", default="")
    ap.add_argument("--self", action="store_true")
    ap.add_argument("--withdraw", action="store_true")
    ap.add_argument("--study", choices=sorted(STUDIES), default="naps")
    a = ap.parse_args()
    sheet_name, items, record = STUDIES[a.study]
    pdir = STUDY / "participants" / a.participant
    pdir.mkdir(parents=True, exist_ok=True)
    path = pdir / record
    if a.withdraw:
        rec = json.loads(path.read_text()) if path.exists() else {}
        rec.update(withdrawn=True, withdrawn_at=dt.datetime.now().isoformat(timespec="seconds"),
                   delete_data=ask("Delete all of your recorded data as well?"))
        path.write_text(json.dumps(rec, indent=1))
        print("Withdrawal recorded. The rig will not run for this pseudonym.")
        return
    if not a.reb_code and not a.self:
        raise SystemExit("Give --reb-code for participants, or --self for an investigator self-experiment.")
    sheet = (STUDY / sheet_name).read_bytes()
    answers = {k: ask(t) for k, t in items}
    required_ok = all(v for k, v in answers.items() if k not in OPTIONAL)
    rec = {
        "participant": a.participant,
        "reb_code": a.reb_code or "SELF-EXPERIMENT",
        "consent_sheet_sha256": hashlib.sha256(sheet).hexdigest(),
        "answers": answers,
        "study": a.study,
        "consented": required_ok,
        "llm_scoring_allowed": answers.get("ai", False),
        "withdrawn": False,
        "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(rec, indent=1))
    print("\nConsent recorded." if required_ok else
          "\nNot every required item was agreed, so this person will not take part. Thank you.")


if __name__ == "__main__":
    main()
