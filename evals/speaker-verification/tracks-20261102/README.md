# Conversation tracks and device templates on live multi-device speech (design §3.2, §8.2 B1)

Registration happens on the phone; speech arrives on a far device. `track_eval.py` simulates households on the 3D-Speaker test set, held out twice as in `identity-20261101`: each half of the speakers is tested with a back-end trained on the other, and every fold holds out one close device (the phone) and one far device (the board).

- Two registered members talk on the held-out far device, conversation by conversation (A, B, A, B, …); each conversation is one track of six segments, and every turn is decided on its track so far with the delivered policy.
- Once a conversation's track is attributed (`personal_memory` met, at least 8 s of voiced speech), it becomes that member's observation template on the device, and later conversations there also compare on the same device (`far_same`).
- Strangers (other held-out speakers on the device) probe the household with and without its templates; each member is probed against a household where only the other is registered.
- `cold_start_accumulation`: one voice's segments accumulated across its conversations against the phone template alone, with the coverage at the policy's thresholds and the best any threshold could reach there while strangers and members stay within each use's limit (chosen on the test data itself, so an upper bound).
- `track_continuation`: a segment against a track of its own voice and of the other member's on the same device.

A track is scored on its latest segments within the longest track its condition was calibrated on (`max_track_segments` in the back-end), as Channel does.

**Simultaneous recording.** 3D-Speaker records each sentence on up to seven devices at once. 41.9 % of the far segments are the very sentences of the speaker's own close-device registration, recorded at the same moment; the product never hears its registration again. They are left out here. `identity-20261101` kept them, which overstated its held-out `far_cross` coverage (strangers' acceptance, the threshold's constraint, is unaffected).

```bash
eidolon_channel/.venv/bin/python -I \
    eidolon_models/evals/speaker-verification/tracks-20261102/track_eval.py \
    --corpus /Users/manson/ai/data/speaker-corpora/3d-speaker \
    --embeddings /Users/manson/ai/data/speaker-corpora/3d-speaker/embeddings.json \
    --policy eidolon_models/speaker_verification/campplus_zh_16k_common/624dcb0f/identity-policy.json \
    --out <dir>
```

## Results (`results.json`, back-end v4, policy v5 thresholds)

| | `personal_memory` | `addressing` |
|---|---|---|
| Phone template only, turns on a track of ≥ 3 segments | 13 % | 27 % |
| Best any threshold could reach, 3–17 segments accumulated | 62–65 % | 74–77 % |
| With the member's device template, single turn (n = 57) | 58 % | 82 % |
| Strangers / a member when only the other is registered, named | ≤ 0.06 % / 0 % | ≤ 0.25 % / ≤ 0.3 % |

- The embedding separates far voices from phone registrations far better than the calibration turns into posteriors on devices it never saw: at a fixed posterior threshold, accumulating a voice's segments does not raise coverage.
- A device template is the strong lever, but making one needs an attributed track first, and cold start attributes few (53 of 673 conversations).
- Continuation on a far device is weak: holding another member's voice to joining a track 1 % of the time lets a voice continue its own track only 20 % of the time (3 % at 0.1 %). A segment that joins the wrong track takes that track's identity, so the policy's continuation threshold is strict (`identity-20261101`).

Calibrating each range of track length separately was tried (one QMF for 1, 2–3 and 4–8 segments): on held-out devices it was badly overconfident (Cllr 2.4–3.7), switched `far_same` off altogether and cut the phone's single-turn `personal_memory` coverage from 77 % to 20 %. It was not adopted.

## Under Channel's real continuation (A1, 11-03)

The tables above take each conversation as one track. Channel continues a track only on the segment's own acoustic evidence, and on a far device a voice continues its own track about a fifth of the time, so tracks there are mostly one or two segments: per turn, far cold start is nearer the single-turn row (`cold/k1`: 15 % / 31 %) than the track row.

**Two members in one conversation** (`two_person_eval.py`, `two_person_results.json`; 76 households, device templates from earlier single-speaker conversations, segments interleaved at random, tracked as Channel tracks them). Nobody was named wrongly for `personal_memory` at any continuation threshold; `addressing` wrongly at most 0.14 %. The threshold barely matters (tracks rarely continue). No mixed conversation made a template: no track reached 8 s.

**Templates pooled from attributed turns** (`pooled_eval.py`, `pooled_results.json`; speakers with at least 24 segments on the held-out far device; two thirds as use, one third as test). Every pooled segment was the member's own (792 of 792); 44 of 152 members formed templates.

| Per turn (`personal_memory` / `addressing`) | Phone template only | With pooled templates |
|---|---|---|
| Members who formed templates (same test turns) | 56 % / 79 % | 53 % / 75 % |
| Members who did not | 5.5 % / 17 % | — |

Strangers named with the pooled templates: 0.05 % / 0.32 %.

Templates made from segments that met the bar add nothing: those segments are the ones the phone template already recognises, and the members who need a device template never produce one (selection). A device template that helps needs a label that does not come from the voice match itself (the Owner, an explicit confirmation, a face).
