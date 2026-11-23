# Information sheet: Vibing on the Sleep

MannLab Canada and the University of Toronto, Department of Electrical and Computer Engineering.
Investigator: Alexander Vicol. Supervisor: Prof. Steve Mann.
Version 0.1, 19 November 2026. Not yet reviewed by a Research Ethics Board. Until an REB approval
number is on file, only the investigators may take part, as a self-experiment.

## What the study asks

Whether a person can direct a software-writing AI agent across a night of sleep: by stating what
they want before bed, by speaking a thought when woken from the edge of sleep, by answering yes
or no with an eye or brow signal while asleep or half awake, and by sleeping on open questions.
The sleeper never writes code and nothing is read from the brain's content. The headband times
the night and reads two deliberate signals.

## What happens in one night

1. **Evening, at the computer, about 15 minutes.** You say or type what you want built. The agent
   writes a plan and a few open questions. With the headband on you practise two signals: eyes
   left-right-left-right (yes) and two frowns (no). You hear each open question with its own short
   sound and may answer it or leave it for the night.
2. **Bedtime.** A quiet noise plays, getting louder; you press a key when you hear it. This sets
   how loud everything in the night is. You hear one sentence in the night voice and may ask for
   it louder or quieter.
3. **Sleep onset.** When the computer sees you falling asleep it whispers a yes/no question (you
   may answer with a signal or ignore it), whispers "think about" your project, and about a
   minute later wakes you with a soft chime and asks what you were thinking. You answer aloud.
   This happens as many times as you set (one by default, zero if you choose), then not again.
4. **Night.** The agent works. In deep sleep, the sounds of half of the questions you left open
   are played very quietly; they stop at any sign of waking. If you wake on your own, a whisper
   may tell you where things stand and ask one question, at most twice a night.
5. **Morning, about 10 minutes.** You say anything you remember, answer the open questions, and
   are shown what the agent built, every input it acted on, and which questions were played.

## Risks

Being woken at sleep onset delays your sleep by some minutes each time. Whispers and sounds can
lighten sleep or wake you; if they do, lower the levels, set the wakes to zero, or stop. The
headband is a consumer product with dry electrodes and no electrical stimulation. Do not take
part if you have a diagnosed sleep disorder, epilepsy or a history of seizures, are taking
sedating medication, or are pregnant.

An AI agent acting on words spoken half asleep, or on a signal read wrongly, can build the wrong
thing. It is confined to a separate folder, cannot publish or install anything, treats no answer
as no, and nothing it writes is merged anywhere unless you do it yourself the next day.

## Your data

- Everything is stored under a pseudonym (for example P-0003) on the investigator's computer.
- Recorded: the headband's signals, the times of every sound and whisper, your spoken reports as
  audio and text, your typed answers, and the agent's log.
- Your words, as text, are sent to the company that runs the AI agent (never the audio, never the
  headband data). Speech is turned into text on this computer.
- Nothing is put on a website or a blockchain. You can ask for all of it to be deleted at any time.

## Taking part is voluntary

You can stop a night by pressing Ctrl+C or taking the headband off, and leave the study at any
time without giving a reason (`python study/consent.py --participant <P> --study vibe --withdraw`).
