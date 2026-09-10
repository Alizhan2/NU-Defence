"""Small, reproducible SpaceNet 7 building-change segmentation baseline.

The target is derived from the two per-date building masks:
0 = unchanged/background, 1 = appeared, 2 = disappeared.  The module keeps
PyTorch imports lazy so dataset preparation remains usable without an ML stack.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CLASS_NAMES = ("unchanged", "appeared", "disappeared")
SCHEMA = "geowatch-spacenet7-change-baseline-v1"


def seed_everything(seed: int, torch: Any | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_fingerprint(records: list[dict[str, Any]]) -> str:
    """Stable identity of an evaluated split, independent of manifest ordering."""
    identity = [
        {"pair_id": row["pair_id"], "aoi": row["aoi"], "before_date": row["before_date"], "after_date": row["after_date"]}
        for row in sorted(records, key=lambda item: item["pair_id"])
    ]
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "geowatch-spacenet7-pairs-v1":
        raise ValueError("Unsupported pair manifest schema")
    if not payload.get("valid"):
        raise ValueError("Pair manifest is not valid")
    return payload


def _read_raster(path: Path, *, mask: bool = False) -> np.ndarray:
    """Read GeoTIFF through rasterio when available, otherwise through Pillow."""
    try:
        import rasterio

        with rasterio.open(path) as src:
            array = src.read()
        if mask:
            return np.any(array > 0, axis=0)
        return np.moveaxis(array[:3], 0, -1)
    except ImportError:
        array = np.asarray(Image.open(path))
        if mask:
            if array.ndim == 3:
                array = np.any(array > 0, axis=-1)
            return array > 0
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        return array[..., :3]


def _normalize_image(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    output = np.empty_like(array, dtype=np.float32)
    for band in range(array.shape[-1]):
        values = array[..., band]
        finite = values[np.isfinite(values)]
        if not finite.size:
            output[..., band] = 0
            continue
        low, high = np.percentile(finite, (2, 98))
        output[..., band] = np.clip((values - low) / max(float(high - low), 1e-6), 0, 1)
    return output


def derive_target(before_mask: np.ndarray, after_mask: np.ndarray) -> np.ndarray:
    if before_mask.shape != after_mask.shape:
        raise ValueError(f"Mask shape mismatch: {before_mask.shape} vs {after_mask.shape}")
    before = before_mask.astype(bool)
    after = after_mask.astype(bool)
    target = np.zeros(before.shape, dtype=np.uint8)
    target[~before & after] = 1
    target[before & ~after] = 2
    return target


@dataclass(frozen=True)
class PairArrays:
    image: np.ndarray
    target: np.ndarray
    pair_id: str


def load_pair(root: Path, record: dict[str, Any]) -> PairArrays:
    before = _normalize_image(_read_raster(root / record["before_image"]))
    after = _normalize_image(_read_raster(root / record["after_image"]))
    before_mask = _read_raster(root / record["before_footprints"], mask=True)
    after_mask = _read_raster(root / record["after_footprints"], mask=True)
    if before.shape[:2] != after.shape[:2] or before.shape[:2] != before_mask.shape:
        raise ValueError(f"Raster shape mismatch for pair {record['pair_id']}")
    image = np.concatenate([before, after], axis=-1)
    return PairArrays(image=image, target=derive_target(before_mask, after_mask), pair_id=record["pair_id"])


def _pad(array: np.ndarray, size: int) -> np.ndarray:
    height, width = array.shape[:2]
    pad_h, pad_w = max(0, size - height), max(0, size - width)
    if not pad_h and not pad_w:
        return array
    pads = ((0, pad_h), (0, pad_w)) + (((0, 0),) if array.ndim == 3 else ())
    return np.pad(array, pads, mode="constant")


def make_dataset(torch: Any, manifest_path: Path, split: str, patch_size: int, samples_per_pair: int, seed: int):
    manifest = load_manifest(manifest_path)
    records = [item for item in manifest["pairs"] if item["split"] == split]
    if not records:
        raise ValueError(f"No records for split={split}")
    root = Path(manifest["root"])

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(records) * samples_per_pair

        def __getitem__(self, index):
            record_index = index % len(records)
            sample_slot = index // len(records)
            arrays = load_pair(root, records[record_index])
            image, target = _pad(arrays.image, patch_size), _pad(arrays.target, patch_size)
            height, width = target.shape
            rng = random.Random(seed + index * 104729)
            changed = np.argwhere(target > 0)
            if sample_slot % 2 == 0 and changed.size:
                row, column = changed[rng.randrange(len(changed))]
                top = max(0, min(int(row) - patch_size // 2, height - patch_size))
                left = max(0, min(int(column) - patch_size // 2, width - patch_size))
            else:
                top = rng.randrange(height - patch_size + 1)
                left = rng.randrange(width - patch_size + 1)
            image = image[top:top + patch_size, left:left + patch_size]
            target = target[top:top + patch_size, left:left + patch_size]
            return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float(), torch.from_numpy(np.ascontiguousarray(target)).long()

    return Dataset(), records, manifest


def _grid_positions(length: int, patch_size: int) -> list[int]:
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, patch_size))
    edge = length - patch_size
    if positions[-1] != edge:
        positions.append(edge)
    return positions


def make_eval_dataset(torch: Any, manifest_path: Path, split: str, patch_size: int):
    """Cover every image deterministically with a non-overlap grid plus edge tiles."""
    manifest = load_manifest(manifest_path)
    records = [item for item in manifest["pairs"] if item["split"] == split]
    if not records:
        raise ValueError(f"No records for split={split}")
    root = Path(manifest["root"])
    tiles: list[tuple[int, int, int]] = []
    for record_index, record in enumerate(records):
        arrays = load_pair(root, record)
        height, width = arrays.target.shape
        for top in _grid_positions(height, patch_size):
            for left in _grid_positions(width, patch_size):
                tiles.append((record_index, top, left))

    class EvalDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(tiles)

        def __getitem__(self, index):
            record_index, top, left = tiles[index]
            arrays = load_pair(root, records[record_index])
            image = _pad(arrays.image, patch_size)
            target = arrays.target
            pad_h, pad_w = max(0, patch_size - target.shape[0]), max(0, patch_size - target.shape[1])
            if pad_h or pad_w:
                target = np.pad(target, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255)
            image = image[top:top + patch_size, left:left + patch_size]
            target = target[top:top + patch_size, left:left + patch_size]
            return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float(), torch.from_numpy(np.ascontiguousarray(target)).long()

    return EvalDataset(), records, manifest, len(tiles)


def build_model(torch: Any, base_channels: int = 16):
    nn = torch.nn

    def block(in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = block(6, base_channels)
            self.enc2 = block(base_channels, base_channels * 2)
            self.bridge = block(base_channels * 2, base_channels * 4)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
            self.dec2 = block(base_channels * 4, base_channels * 2)
            self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
            self.dec1 = block(base_channels * 2, base_channels)
            self.head = nn.Conv2d(base_channels, len(CLASS_NAMES), 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            bridge = self.bridge(self.pool(e2))
            d2 = self.dec2(torch.cat([self.up2(bridge), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)

    return TinyUNet()


def _update_confusion(torch: Any, matrix: Any, prediction: Any, target: Any) -> None:
    valid = (target >= 0) & (target < len(CLASS_NAMES))
    encoded = len(CLASS_NAMES) * target[valid] + prediction[valid]
    matrix += torch.bincount(encoded, minlength=len(CLASS_NAMES) ** 2).reshape(len(CLASS_NAMES), -1).cpu()


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    valid_f1, valid_iou = [], []
    for index, name in enumerate(CLASS_NAMES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        support = int(matrix[index, :].sum())
        iou = tp / (tp + fp + fn) if tp + fp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        rows[name] = {"iou": iou, "f1": f1, "support_pixels": support, "tp": tp, "fp": fp, "fn": fn}
        if index > 0 and support > 0:
            if iou is not None: valid_iou.append(iou)
            if f1 is not None: valid_f1.append(f1)
    return {
        "classes": rows,
        "macro_change_iou": float(np.mean(valid_iou)) if valid_iou else None,
        "macro_change_f1": float(np.mean(valid_f1)) if valid_f1 else None,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def evaluate_loader(torch: Any, model: Any, loader: Any, device: Any) -> dict[str, Any]:
    model.eval()
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
    with torch.inference_mode():
        for images, target in loader:
            prediction = model(images.to(device)).argmax(1).cpu()
            _update_confusion(torch, confusion, prediction, target)
    return metrics_from_confusion(confusion.numpy())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
