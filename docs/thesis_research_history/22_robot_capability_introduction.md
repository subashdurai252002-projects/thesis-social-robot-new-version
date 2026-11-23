# Robot Capability Introduction

Date: 24 November 2026  
Task: 4 — robot capability introduction  
Status: laptop implementation and focused live validation complete

## Purpose

A newly activated user needs a short, accurate explanation of what the current
robot can do. The former greeting, “Hello. How can I help you?”, did not expose
the validated adaptive-speech controls. The introduction must also avoid
claiming unfinished identity, speaker-attribution, movement or dashboard work.

## Implemented behaviour

`run_social_robot_interaction_laptop.py` defines this fresh-interaction text:

> Hello. I can answer questions and explain information. You can also ask me
> to speak faster, slower, or normally. How can I help you?

The top-level coordinator speaks it after fresh attention and passive-liveness
activation and before the conversation loop. A five-second post-goodbye
reopening resumes the existing conversation loop, so the introduction is not
repeated. An ordinary in-conversation greeting remains short.

## Claim boundary

The text includes only behavior already available on the laptop:

- answer questions through the existing conversation path;
- explain information;
- accept local fast, slow and normal Piper controls.

It does not claim that the robot remembers or recognizes a person, identifies
the acoustic speaker, moves the Unitree, or exposes the future dashboard.
Those capabilities require later tasks and evidence.

## Verification

Deterministic evidence:

- `gaze_pipeline/experiments/test_capability_introduction.py`: 3/3 passed;
- combined capability-introduction and conversation/reopening tests: 26/26;
- interaction-state regression: 10/10;
- relevant compilation and `git diff --check`: passed.

Live evidence:

- a fresh attention/liveness activation spoke the exact full introduction;
- “Goodbye” completed the conversation and entered the reopening window;
- reopening the same conversation did not replay the introduction;
- shutdown completed cleanly.

During the reopening window, a background sound was once transcribed as
“Thank you” and caused a false reopening. This is retained as Task 8
non-speech/noise evidence. It does not change the Task 4 fresh-versus-reopening
boundary and no phrase-specific workaround was added.

## Remaining boundary

The introduction is fixed, concise guidance rather than a complete final
demonstration script. By user decision, Task 3's 30–50-second demonstration is
deferred until the local dashboard is connected so the final demonstration can
cover the integrated system.
