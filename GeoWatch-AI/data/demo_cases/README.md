# Real temporal demo cases

This directory is the materialization target for three SpaceNet 7 cases:

- `appeared` — new building-footprint pixels;
- `disappeared` — lost building-footprint pixels (removal or footprint change);
- `uncertain` — no confirmed directional footprint event inside the crop.

The images are real SpaceNet 7 / Planet monthly imagery. Evidence is derived from
the official per-date footprint masks and is **not** a GeoWatch model prediction.
The negative case only means that the footprint annotations do not confirm a
building event; it does not prove that nothing else changed.

Generated imagery is intentionally not committed. After downloading and extracting
the official training archive, run from the repository root:

```powershell
python scripts/prepare_spacenet7_pairs.py --source D:\datasets\SN7_buildings\train --output data\spacenet7\pairs.json --hash-files
python scripts/build_real_demo_cases.py --pairs data\spacenet7\pairs.json --output data\demo_cases
```

The second command writes `catalog.json`, three `cases/<id>/record.json` records,
before/after PNGs, footprint masks and directional evidence masks. Every record
contains dates, AOI, original relative paths, crop coordinates, checksums, source,
license and attribution.

Official source: <https://www.spacenet.ai/sn7-challenge/>  
License: Creative Commons Attribution-ShareAlike 4.0. Keep the attribution
“SpaceNet 7 / SpaceNet Partners; imagery courtesy of Planet” in the UI and exports.
