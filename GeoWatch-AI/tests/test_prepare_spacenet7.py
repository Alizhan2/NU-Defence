import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("prepare_spacenet7_pairs", Path(__file__).parents[1] / "scripts" / "prepare_spacenet7_pairs.py")
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


def test_manifest_splits_by_aoi_and_pairs_adjacent_dates(tmp_path):
    for index in range(6):
        aoi = tmp_path / f"AOI_{index:02d}"
        images, labels = aoi / "images_masked", aoi / "labels_match_pix"
        images.mkdir(parents=True); labels.mkdir()
        for month in ("2018_01", "2018_02", "2018_03"):
            (images / f"mosaic_{month}.tif").write_bytes(b"image")
            (labels / f"mask_{month}.tif").write_bytes(b"label")
    manifest = MODULE.build_manifest(tmp_path)
    assert manifest["valid"]
    assert len(manifest["pairs"]) == 12
    assert all(manifest["counts"].values())
    split_sets = [set(manifest["split_aois"][name]) for name in ("train", "val", "test")]
    assert not split_sets[0] & split_sets[1]
    assert not split_sets[0] & split_sets[2]
    assert not split_sets[1] & split_sets[2]


def test_manifest_rejects_missing_per_date_footprints(tmp_path):
    for index in range(3):
        aoi = tmp_path / f"AOI_{index:02d}"
        images, labels = aoi / "images_masked", aoi / "labels_match_pix"
        images.mkdir(parents=True); labels.mkdir()
        for month in ("2018_01", "2018_02"):
            (images / f"mosaic_{month}.tif").write_bytes(b"image")
        (labels / "mask_2018_01.tif").write_bytes(b"label")
    manifest = MODULE.build_manifest(tmp_path)
    assert not manifest["valid"]
    assert len(manifest["missing_labels"]) == 3
