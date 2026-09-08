# SpaceNet 7 data decision

GeoWatch uses SpaceNet 7 as the planned building chronology dataset because the official release provides monthly Planet imagery, per-date building footprints and an explicit CC BY-SA 4.0 dataset license. Attribution to SpaceNet and Planet must be retained.

The data is not vendored into this repository. Download the official training archive from the SpaceNet AWS bucket, extract it, then build a manifest without copying the imagery:

```powershell
python scripts/prepare_spacenet7_pairs.py --source D:\datasets\SN7_buildings\train --output data\spacenet7\pairs.json --hash-files
```

The split unit is the AOI, not the tile or month. Every pair uses adjacent dates and requires building-footprint masks at both dates. This makes `appeared` and `disappeared` derivable from per-date footprints; a binary change mask alone would not establish direction.

This manifest is only a data gate. It does not claim that the dataset was downloaded, that a model was trained, or that test metrics exist.
