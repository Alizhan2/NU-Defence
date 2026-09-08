# GeoWatch object detector — model card

## Status

**Candidate, unverified. Do not present as a competition-quality model.**

The active `models/best.pt` is byte-identical to
`runs/obb/models/dota8_smoke/weights/best.pt`. It is the local eight-epoch
pipeline smoke checkpoint, not demonstrated to be the DOTA4 checkpoint produced
by the Colab run.

`runs/metrics_test.json` contains plausible-looking four-class values, but the
local dataset cannot reproduce them: its test split has one aircraft instance
and none of ship, small vehicle or large vehicle. The metrics file also lacks an
explicit `split=test` marker and dataset fingerprint. The values remain an
unverified external artifact until matched dataset provenance and checkpoint
hashes are supplied.

The retained Colab output confirms that a previous DOTA4 run reached 77 epochs
and evaluated 13,244 instances across 282 scene-separated test images. Its
historical values are recorded in
`data/runs/colab_dota4_evidence_2026-09-07.json`. The Colab runtime was reset
before `best.pt`, the manifest fingerprint and checkpoint hash were recovered,
so these values are not eligible for model promotion yet.

## Intended use

- preliminary analyst-assisted detection of aircraft, ships, small vehicles and
  large vehicles in suitable overhead imagery;
- demonstration and pipeline testing;
- every result requires human review.

Not intended for autonomous operational, targeting, legal or safety decisions.
It does not detect building change; that requires a separately evaluated change
detection model and paired imagery.

## Required promotion evidence

1. Scene-separated train/val/test manifest with all four classes in test.
2. Dataset fingerprint and provenance/license record.
3. Final checkpoint SHA-256 produced by the recorded training run.
4. Metrics produced only once on test after selection on validation.
5. Successful local checkpoint load and probe inference.
6. Per-class metrics, error examples and target-machine latency.

Run `scripts/audit_model.py`; only `evidence_status=verified` is eligible for a
verified model card. A successful probe proves operability only, not accuracy.
