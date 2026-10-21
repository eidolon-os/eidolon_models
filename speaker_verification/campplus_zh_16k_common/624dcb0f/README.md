# CAM++ zh-cn 16 kHz — 624dcb0f

Canonical model asset owner: `eidolon_models`. Source is the unchanged CAM++
package formerly stored in Channel. Weight SHA256 is
`3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8`.
See `manifest.json` for the full immutable upstream revision and file checksums;
`upstream-README.md` preserves that revision's model card and Apache-2.0 declaration.
`LICENSE` contains the upstream project's Apache-2.0 license text.

Ops delivers this package as one checksum-verified component artifact to
`/var/lib/eidolon/models/campplus-zh-16k-624dcb0f` when the Host selects
`local_speaker_verification`. This selects model availability, not a Companion's
voice restriction. It adds no service or port. The runtime never downloads weights.
Channel owns inference; Admin orchestrates mobile enrollment; Data owns personal
profiles and Companion allowlists. No model asset contains personal voice data.

For local development set `EIDOLON_3DSPEAKER_MODEL_DIR` to this directory.
The numerical reference accepts contiguous mono PCM16 at 16 kHz. Upstream's 0.31
and the withdrawn feature's 0.65 are not calibrated product authorization thresholds.
This migration changes asset ownership and delivery, not matching quality or policy.
Cross-device positive/negative evaluation remains required before enabling admission.
