# Face verification: offline checks

Scripts that score a public face dataset through `eidolon_vision`'s own encoder,
so the numbers describe the code that serves requests. Run them with
`eidolon_vision`'s environment (see each script's docstring). They read images
and write aggregate numbers only: no image, name or face feature is stored here.

- `lfw_pairs_eval.py`: the standard LFW protocol (6000 pairs, 10 folds).
- `lfw_impostor_eval.py`: every identity's first image as the enrolled photo,
  all cross-identity comparisons, probes optionally shrunk.
- `lfw-20261114/`: results for `face_recognition/sface_yunet_opencv_zoo/48b10fa5`.
  How they set the score policy is in that package's `face-policy.md`.

LFW itself is not in this repository. The 2026-11-14 run used the funneled set
as fetched by scikit-learn's `fetch_lfw_pairs`.
