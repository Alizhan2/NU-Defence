# GeoWatch object detector — model card

## Status

**Verified DOTA4 object-detection baseline. Human review remains mandatory.**

The active `models/best.pt` is `yolov8n-obb-dota4-seed42-v1`. Its SHA-256 is
`6909f7c506404816671562de8f8c33185fa352e5fabb6672b78ce60e72af8a14`.
The promoted model card and metric report bind this checkpoint to the
scene-separated internal DOTA4 test split with dataset fingerprint
`sha256:e068a4ba8860e4f5bbf16834e8428e70f6b4d88772d4671ce2b1ff228d31db4d`.

Test metrics: Precision `0.8741`, Recall `0.8354`, F1 `0.8543`, mAP50 `0.8915`,
and mAP50-95 `0.6719`. Per-class mAP50-95 is `0.6909` for aircraft, `0.6804`
for ship, `0.5605` for small vehicle, and `0.7557` for large vehicle.

## Intended use

- preliminary analyst-assisted detection of aircraft, ships, small vehicles and
  large vehicles in suitable overhead imagery;
- demonstration and pipeline testing;
- every result requires human review.

Not intended for autonomous operational, targeting, legal or safety decisions.
It does not detect building change; that requires a separately evaluated change
detection model and paired imagery.

## Evidence and reproduction

1. `models/model_card.json` records the promoted checkpoint hash and model version.
2. `data/runs/metrics.json` records the dataset fingerprint, split, seed, threshold and per-class metrics.
3. `data/runs/dataset_manifest.json` records split sizes, class counts and the matching dataset fingerprint.
4. Local checkpoint load and end-to-end application inference have passed.

The raw DOTA4 split is not vendored because of size and licensing. A fresh
file-level `scripts/audit_model.py` run therefore requires the same official
dataset to be supplied with `--data`; the repository alone verifies the
checkpoint/report/manifest binding, not every source image again.

The metrics support the four-class object detector only. They do not validate
building chronology, Google Earth Engine ingestion, or performance on a new
customer domain. A successful probe proves operability only, not accuracy.
