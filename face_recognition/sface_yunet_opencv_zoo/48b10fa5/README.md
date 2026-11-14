# YuNet (2023mar) + SFace (2021dec) — OpenCV zoo 48b10fa5

Canonical model asset owner: `eidolon_models`. Face detection (YuNet, MIT) and
face embedding (SFace, Apache-2.0), unchanged from
https://github.com/opencv/opencv_zoo at commit
`48b10fa505aeb51fbd127fc60a5c660040e823b7`. `manifest.json` lists the files and
their SHA-256; `LICENSE-yunet`, `LICENSE-sface` and the two `upstream-README-*`
files preserve the upstream terms and model cards.

`eidolon_vision` owns inference: it is given this directory
(`EIDOLON_VISION_FACE_IDENTITY_MODEL_DIR`), verifies the checksums at start and
never downloads weights. Admin orchestrates enrollment, Data stores each
Person's 128-number template beside the photo it came from, Channel decides
from Vision's evidence. No file here contains a face, a name or a template.

How a score becomes a band, and what that was measured on, is in
`face-policy.md`. The policy file names the model and preprocessing it belongs
to; Vision refuses to pair it with anything else.

Not yet delivered by Ops: there is no component artifact entry for this package
and `eidolon_vision` is not a product unit. For local development point the
setting at this directory.
