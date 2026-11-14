# Face evidence: score bands and what they were measured on

A comparison returns advisory evidence for addressing someone by name. It is
never admission, authentication or a memory grant, and there is no liveness
check: a printed photo of an enrolled person scores like the person.

Vision reads `face-policy.json` beside the weights and turns each cosine score
into a band; Channel decides what a band may be used for.

| Field | Value | Meaning |
|---|---|---|
| `thresholds.strong` | 0.50 | At or above: the face may be named |
| `thresholds.weak` | 0.363 | At or above, below strong: resembles someone enrolled, not named |
| `margin` | 0.08 | A strong leader within this of the runner-up is not named |
| `min_probe_face_px` | 28 | Shorter side of the face box in the frame; smaller faces are reported, not compared |
| `min_enroll_face_px` / `min_enroll_score` | 80 / 0.8 | A reference photo below either is refused |
| `min_probe_score` | 0.7 | Detector confidence below which a face is reported, not compared |
| `confidence` | `unvalidated` | See "What has not been measured" |

## Evidence (`evals/face-verification/lfw-20261114`)

Dataset: LFW (funneled), web photographs of public figures, faces about 96 px
across. Every image went through eidolon_vision's own encoder (YuNet detection,
SFace alignment and embedding), the code that serves requests.

**Standard protocol** (`pairs.json`, `lfw_pairs_eval.py`): 6000 pairs, 10 folds,
accuracy 0.9940 ± 0.0038. The upstream model card reports 0.9940, so the
pipeline here reproduces the published figure.

**Strangers at scale** (`strangers.json`, `lfw_impostor_eval.py`): the first image
of each of 5749 identities is the enrolled photo; the second image of each of
the 1680 identities that have one is the probe. 9.66 million different-person
comparisons, 1680 same-person comparisons. The probe is shrunk to see what
distance costs; the enrolled photo stays full size.

| Probe face (median px) | Own photo ≥ strong | Own photo ≥ weak | Stranger pair ≥ strong | Stranger pair ≥ weak |
|---|---|---|---|---|
| 96 | 93.3 % | 98.8 % | 3.5 in a million | 9.2 in ten thousand |
| 59 | 93.3 % | 98.9 % | 3.6 in a million | 9.6 in ten thousand |
| 40 | 90.7 % | 98.6 % | 3.3 in a million | 8.4 in ten thousand |
| 30 | 83.3 % | 96.7 % | 3.2 in a million | 6.5 in ten thousand |
| 20 | 61.4 % | 88.8 % | 2.0 in a million | 5.6 in ten thousand |

A stranger in front of a household of eight enrolled people reached `weak`
against at least one of them in 0.8 % of probes (0.4 % for four, 0.1 % for
one), and `strong` in none of 1680. With 1680 probes, "none" bounds that rate
below roughly 0.2 % at 95 % confidence, no tighter.

How the values were chosen:

- `strong` 0.50: the lowest threshold in the table's grid at which stranger
  pairs are a few in a million. The 34 pairs above it at full size include
  LFW's known label errors (one person under two names; the highest stranger
  score is 0.95), so the true rate is lower than stated.
- `weak` 0.363: the cosine threshold OpenCV publishes for this model. Here it
  accepts about one stranger pair in a thousand, which is why it only ever
  yields "resembles", never a name.
- `min_probe_face_px` 28: below about 30 px the share of own photos reaching
  `strong` falls quickly (83 % at 30 px, 61 % at 20 px) and the detector starts
  missing faces (16 of 1680 at 20 px).
- `margin` 0.08 and the enrollment floors are set by judgement, not measured:
  LFW has no households of similar-looking relatives to measure a margin on.

## What has not been measured

Nothing here was measured on a device camera. LFW is frontal-to-three-quarter
web photography; a robot on a table looks up at a face, indoors, often backlit,
and the enrolled photo comes from a phone gallery. Scores for the same person
are expected to be lower in that setting and the table above does not say by
how much. Family members who resemble each other, children, masks and glasses
are also unmeasured.

`confidence` stays `unvalidated` until own-person and stranger scores have been
collected on the target devices. While it is, Channel does not publish a face
match as an identified person observation; the conversation may still use the
name, with wording that leaves room for error.
