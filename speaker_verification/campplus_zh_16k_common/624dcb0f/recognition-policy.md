# Conversation Person recognition trial

This artifact is distinct from the historical admission trial. It returns advisory identity evidence, never admission, authentication or personal-memory permission. Channel reads it alongside the existing shared model; no second model or weights download is needed.

The initial similarity floor 0.55 follows the existing enrollment consistency/trial floor, with an additional 0.10 margin between different Persons. These are conservative trial settings, **not a calibrated false-accept rate**. Multiple templates for the same Person are aggregated before the margin check. The 1500 ms speech floor matches the supported extraction domain. Missing assets, model mismatch, ambiguous candidates, dependency failure and expired evidence must never produce a Person.

Before this policy may authorize automatic access to private memory, evaluate held-out A/B/unknown speech recorded on the intended devices, measure false accepts and rejects including short speech, TV/replay, similar voices and overlapping speakers, and publish a separately reviewed policy. A successful deterministic pipeline test is not acoustic validation. The UI must say the voice *may* match; unknown and unavailable remain normal conversation states.
