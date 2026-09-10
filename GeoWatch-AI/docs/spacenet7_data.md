# SpaceNet 7 data decision

GeoWatch uses SpaceNet 7 as the planned building chronology dataset because the official release provides monthly Planet imagery, per-date building footprints and an explicit CC BY-SA 4.0 dataset license. Attribution to SpaceNet and Planet must be retained.

The data is not vendored into this repository. Download the official training archive from the SpaceNet AWS bucket, extract it, then build a manifest without copying the imagery:

```powershell
aws s3 cp --no-sign-request s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz D:\datasets\SN7_buildings_train.tar.gz
tar -xf D:\datasets\SN7_buildings_train.tar.gz -C D:\datasets\SN7_buildings
python scripts/prepare_spacenet7_pairs.py --source D:\datasets\SN7_buildings\train --output data\spacenet7\pairs.json --hash-files
```

The split unit is the AOI, not the tile or month. Every pair uses adjacent dates and requires building-footprint masks at both dates. This makes `appeared` and `disappeared` derivable from per-date footprints; a binary change mask alone would not establish direction.

This manifest is only a data gate. It does not claim that the dataset was downloaded, that a model was trained, or that test metrics exist.

## Three real demo cases

After the manifest passes its data gate, select three deterministic crops from held-out test AOIs:

```powershell
python scripts/build_real_demo_cases.py --pairs data\spacenet7\pairs.json --output data\demo_cases
```

The generated `catalog.json` has `appeared`, `disappeared`, and `uncertain`
records with dates, AOI, source paths, crop coordinates,
checksums, license and attribution. Event direction is derived from official
per-date footprint masks with a two-pixel registration tolerance. These are
reference annotations for the demo, not model predictions or accuracy claims.

## Building-change baseline

The reproducible baseline receives six channels (RGB before + RGB after) and
predicts three pixel classes: `unchanged/background`, `appeared`, and
`disappeared`. Training uses deterministic change-aware crops; model selection
uses validation AOIs. The final test command covers every test pair with a
deterministic tile grid and binds the metrics to both the checkpoint and dataset
manifest using SHA-256.

```powershell
python scripts/train_spacenet7_change.py --manifest data\spacenet7\pairs.json --epochs 20 --patch-size 256 --samples-per-pair 8 --batch 8 --device cuda
python scripts/evaluate_spacenet7_change.py --manifest data\spacenet7\pairs.json --checkpoint models\spacenet7_change_baseline\best.pt --split test --batch 8 --device cuda
```

The test evaluator writes `data/runs/spacenet7_metrics.json`. The application
shows it as verified only when it is a default test artifact containing the
checkpoint hash, manifest hash, held-out AOIs, split fingerprint and full-grid
coverage metadata. On an 8 GB GPU, reduce `--batch` to `4` if CUDA runs out of
memory.

Before downloading the full archive, the end-to-end plumbing can be checked on
generated fixtures. Smoke output is deliberately marked unverified and never
replaces the real test artifact:

```powershell
python scripts/smoke_spacenet7_change.py --keep data\spacenet7_smoke
```
