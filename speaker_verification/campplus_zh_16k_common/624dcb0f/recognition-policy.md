# Conversation Person recognition policy

This artifact is distinct from the historical admission trial. It returns advisory identity evidence, never admission, authentication or personal-memory permission. Channel reads it alongside the existing shared model; no second model or weights download is needed.

## v3: per-Person, per-duration thresholds (`calibration`)

Nobody labels anything. Each Person's thresholds are derived on the Host from three kinds of evidence that carry their labels for free:

- **The Person's own registration.** Every registration recording is also cut into windows (1.0 / 2.0 / 3.5 s anchored to the voiced span); each window's PCM is embedded like a conversation probe, and its voiced length is read from the recording's own voice-activity pass. A sample's windows are scored against the mean of the *other* samples, the trial's windows against the full template, so no window meets a template built from itself. These are the Person's same-person scores per band.
- **`impostor-cohort.npz`**: 69 public AISHELL-1 speakers (embeddings only — no audio, transcripts or names), each recording levelled to device-like loudness, sent through an Opus round trip at 48 kHz and embedded by Channel's own extractor, utterance plus the same windows. See `impostor-cohort.json` for provenance and `evals/speaker-verification/cohort-20261031/` for the build.
- **The household's other registrations**: other Persons' windows are impostors too.

Per band (voiced speech 0.6–1.2 s / 1.2–2.5 s / ≥ 2.5 s, measured by the extractor) a Person's threshold is the lowest score that keeps the cohort false-accept rate within `max_false_accept` divided by the number of Persons — the target is per turn, and a stranger can be mistaken for any of them — never below the band floor, and above every household window. If that threshold accepts fewer than `min_genuine_acceptance` of the Person's own windows in the band (or the band has fewer than `min_genuine_scores` windows), the band does not decide that Person; it only extends a track already bound (design §3.3). The best Person over its own threshold is identified only if it leads every other Person by the band margin. A Person registered before windows existed decides with the fixed `threshold` and `min_person_margin`, never below what the cohort requires.

The table is recomputed from the recognition snapshot, so it changes when the household changes; nothing about it is stored.

## Evidence (2026-10-31, `evals/speaker-verification/cohort-20261031/evaluate.py`)

30 held-out speakers (never in the cohort), 15 two-person households, registration through Channel's own `evaluate_enrollment`; 836 genuine and 23 394 stranger probes.

| | v3 (`max_false_accept` 0.003) | v2 fixed 0.55 / 0.10 |
|---|---|---|
| strangers identified as a household Person, all probes | 0.38 % | 0.69 % |
| … voiced ≥ 1 s | 0.42 % | 0.82 % |
| … voiced ≥ 2.5 s | 0.44 % | 1.12 % |
| own probes identified correctly, voiced ≥ 1 s | 96.2 % | 99.9 % |

A cohort budget of 0.005 measured 0.53 % (0.59 % at ≥ 1 s) on held-out strangers: 69 speakers under-estimate the impostor tail, so the budget is set to 0.003. Most remaining false accepts come from a few look-alike voice pairs (14 of 840 Person–stranger pairs; five pairs carry 84 %), which no threshold separates without giving up the Person.

**Not shown by this evaluation.** Genuine probes here come from the same recording session and microphone as registration, so own-voice recognition is optimistic; real conversations on another device score lower (10-30: 0.37–0.53). Coverage on real devices is what observation templates per device channel (design §3.2, stage 3) are for. The extractor's voiced-speech measure depends on input level (it saturates near RMS 0.06); quiet speech is measured shorter and lands in a shorter band.

## Unchanged

Below `min_speech_ms` the result is insufficient speech. A probe below threshold stays uncertain/short_speech_inconclusive until it has `min_unknown_speech_ms` of voiced speech; a weak short probe is not evidence of an unregistered speaker. Neither silence padding nor audio from previous turns counts as new identity evidence. All positive decisions remain controlled_trial. Missing assets, a missing or mismatched cohort, model mismatch, ambiguous candidates, dependency failure and expired evidence never produce a Person. Recognition owns the acceptance decision; evaluation labels remain diagnostic metadata. Memory separately verifies the current signed Person and Data authorization for an explicitly enabled Realm. The UI consumes the recognition status; unknown and unavailable remain normal conversation states.
