# Vibing on the Sleep

Vibe coding across a night of sleep, with a Muse headband: the sleeper directs, an AI coding agent writes.
Nothing is read from the brain's content. The headband times the night and reads two deliberate signals
(eyes left-right-left-right for yes, two frowns for no); a spoken thought on waking becomes the agent's next
instruction, and the morning shows what was built.

It grew out of *Jobbing on the Sleep*, a sleep-onset and REM cueing rig, whose code it reuses:
`rig/` (the headband stream and the slow-wave cue gate), `detector/` (a sleep-stage detector validated on
20 Sleep-EDF nights; trained models in `models/`) and `study/` (consent, puzzle cueing, analysis).

MannLab Canada, University of Toronto.

## Needs

- Python 3.11+ with numpy, scipy, pandas, scikit-learn, joblib, python-osc, soundfile, sounddevice; `kokoro-onnx` for the voice.
- A Muse headband streaming over OSC (MuseLog, raw EEG on) to this computer, port 5000.
- Claude Code installed (`claude` on the PATH) for the agent.
- The voice model files (350 MB, not in git) in `models/voice/`: `kokoro-v1.0.onnx` and `voices-v1.0.bin` from the kokoro-onnx release page.

## Run

Vibe coding across a night of sleep: the sleeper directs, an AI coding agent writes. Nothing is
read from the brain's content. The headband times the night and reads two deliberate signals.

```bash
python study/consent.py --participant P-0001 --self --study vibe   # once (CONSENT_VIBE.md)
python vibe/evening.py --participant P-0001 --self    # ~15 min: brief, plan, signal practice, open questions
python vibe/night.py   --participant P-0001 --self    # bedtime, headband streaming; Ctrl+C in the morning
python vibe/morning.py --participant P-0001 --self    # ~10 min: report, answers, the reveal, the branch
python vibe/voice.py --say "Good night."              # hear the night voice
```

| Way in | When | What the sleeper gives | File |
|---|---|---|---|
| 1 Sleep-onset dictation | at sleep onset, `--wakes` times (default 1, 0 = never) | a spoken thought on being woken a minute after drifting off; transcribed here, it is the agent's next instruction | `night.py` `onset_round` |
| 2 Yes/no in sleep | at each sleep-onset trigger (and in REM with `--probe-rem`) | eyes left-right-left-right = yes, two frowns = no, nothing = no answer | `signals.py`, `night.py` `probe` |
| 3 Natural awakenings | when the sleeper has surfaced on their own, at most twice a night | a signal or a few words after a whispered status and question | `night.py` `awakening` |
| 4 Cues in deep sleep | slow-wave sleep, the puzzle study's gate | nothing in the night; the morning's answers to the open questions | `rig/sws_cue.py` gate, `evening.py`, `morning.py` |

**The agent** (`vibe/agent.py`) is Claude Code run headless in a folder of its own (a new
repository under `~/VibingOnTheSleep`, or a worktree of `--project` on a branch `vibe/<stamp>`).
No web, no MCP servers, user settings not loaded; shell commands only from an allow list;
pushes, remotes, installs and fetches denied by name; a dollar cap per turn and a cap on turns.
No answer is no. Nothing is merged: the morning prints the command and the person decides.
These are permission rules, not an operating-system sandbox.

**The voice** (`vibe/voice.py`) is Kokoro, a neural voice run on this computer (no account, no
network): a warm voice for the evening and morning, a near-whisper for the night, a soft chime
to wake. Model files (350 MB, not in git) go in `models/voice/`: `kokoro-v1.0.onnx` and
`voices-v1.0.bin` from the kokoro-onnx release page. Every level in the night is a set number of
dB above the bedtime hearing threshold. To try another voice: `python vibe/voice.py --list`,
`python vibe/voice.py --say "Good night." --kokoro bf_emma`, then set `VIBE_NIGHT_VOICE` (or
`VIBE_DAY_VOICE`) for the scripts. Samples: `python vibe/voice.py` writes them to
`study/audio/vibe/samples/`.

**What was checked** (`python vibe/test_vibe.py`, no sleeper; `python vibe/agent.py --smoke` for
the real agent):

- Accidental signals, 20 Sleep-EDF nights, 57,537 ten-second windows, nobody asked anything
  (`vibe/false_alarms.py edf`, default thresholds): a false YES in 0.09% of N1 windows, 0.02% of
  N2, 0.01% of N3, 0.79% of REM, 0.68% of wake; a false NO in 3.6% of N1, 0.9% of N2, 1.0% of N3,
  1.0% of REM, 2.6% of wake. The eye electrodes there sit beside the eyes, so the YES figures are
  an upper bound for the pattern, not Muse numbers. On the author's three headband recordings:
  YES in 1, 1 and 6 windows (0.2%, 0.3%, 2.1% of the usable still ones), NO in 1, 0 and 1.
- The author's Athena recording of 28 October replayed through the night's logic
  (`vibe/replay_own_night.py`): the validated detector read wake throughout (ear electrodes), the
  forehead rule fired at 04:01:42, two minutes before the first K-complex and spindles that the
  night's own analysis found at 04:03:30; cues ran 04:22-05:00 inside the stretch that analysis
  called slow-wave sleep (04:16-05:02); a stir at 05:00 would have had one whispered question;
  no accidental signal in 425 usable windows. One night, no scored stages: consistent, not validated.
- The replayed Sleep-EDF night: sleep onset found in scored N1, the made eye signal read as yes
  and passed to the agent, the dictation passed to the agent, cues only for the cued questions.

**Not known yet:** whether the sleeper's face makes signals the forehead electrodes can read
(the evening practice measures it: six check trials the thresholds were not set on), whether
anyone answers in sleep on a Muse (healthy nappers answered 22% of spoken trials in N1 and under
7% in N2 and REM with facial electrodes, Tuerker et al. 2023), whether the forehead rule holds on
other nights, and whether any of the four ways improves what gets built.

## Test without a sleeper

```bash
python vibe/osc_replay.py            # a fake headband over OSC
python vibe/test_vibe.py             # needs Sleep-EDF nights in data/ (python detector/fetch_sleepedf.py)
python vibe/agent.py --smoke         # one real agent turn (costs money)
```
