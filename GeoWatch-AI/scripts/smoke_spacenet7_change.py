"""Create a synthetic 3-AOI fixture and run one CPU train/evaluation cycle."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOCAL_RUNTIME = ROOT / ".runtime"
if LOCAL_RUNTIME.is_dir():
    sys.path.insert(0, str(LOCAL_RUNTIME))
    os.environ["PYTHONPATH"] = str(LOCAL_RUNTIME) + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, str(ROOT))
from scripts.prepare_spacenet7_pairs import build_manifest
from src.spacenet7_baseline import atomic_json


def create_fixture(root: Path) -> Path:
    for index in range(3):
        aoi = root / f"AOI_{index:02d}"
        images, labels = aoi / "images_masked", aoi / "labels_match_pix"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(exist_ok=True)
        before = np.full((64, 64, 3), 50 + index * 15, dtype=np.uint8)
        after = before.copy()
        mask_before = np.zeros((64, 64), dtype=np.uint8)
        mask_after = np.zeros((64, 64), dtype=np.uint8)
        mask_before[8:24, 8:24] = 255
        mask_after[8:24, 8:24] = 255
        mask_after[34:52, 36:54] = 255
        after[34:52, 36:54] = (210, 210, 210)
        Image.fromarray(before).save(images / "mosaic_2018_01.tif")
        Image.fromarray(after).save(images / "mosaic_2018_02.tif")
        Image.fromarray(mask_before).save(labels / "mask_2018_01.tif")
        Image.fromarray(mask_after).save(labels / "mask_2018_02.tif")
    manifest_path = root / "pairs.json"
    atomic_json(manifest_path, build_manifest(root, seed=42))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", type=Path, help="Keep artifacts in this directory instead of a temporary directory")
    args = parser.parse_args()
    context = None if args.keep else tempfile.TemporaryDirectory(prefix="geowatch-sn7-smoke-")
    work = args.keep.resolve() if args.keep else Path(context.name)
    work.mkdir(parents=True, exist_ok=True)
    manifest = create_fixture(work / "fixture")
    output = work / "run"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train_spacenet7_change.py"), "--manifest", str(manifest), "--output", str(output), "--epochs", "1", "--patch-size", "64", "--samples-per-pair", "1", "--batch", "1", "--base-channels", "4", "--device", "cpu"], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate_spacenet7_change.py"), "--manifest", str(manifest), "--checkpoint", str(output / "best.pt"), "--output", str(output / "smoke_metrics_test.json"), "--split", "test", "--batch", "1", "--device", "cpu"], check=True)
    print(f"SMOKE_OK artifacts={output}")
    if context:
        context.cleanup()


if __name__ == "__main__":
    main()
