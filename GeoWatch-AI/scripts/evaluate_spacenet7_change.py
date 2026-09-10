"""Evaluate a trained SpaceNet 7 baseline on one untouched manifest split."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.spacenet7_baseline import SCHEMA, atomic_json, build_model, evaluate_loader, make_eval_dataset, sha256, split_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required: install torch before evaluation") from exc
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != SCHEMA:
        raise SystemExit("Checkpoint schema mismatch")
    expected_hash = checkpoint.get("manifest_sha256")
    actual_hash = sha256(args.manifest.resolve())
    if expected_hash != actual_hash:
        raise SystemExit("Manifest hash differs from the training manifest; refusing incomparable evaluation")
    config = checkpoint["config"]
    dataset, records, manifest, tile_count = make_eval_dataset(torch, args.manifest.resolve(), args.split, int(config["patch_size"]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    model = build_model(torch, int(config["base_channels"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    metrics = evaluate_loader(torch, model, loader, device)
    production_artifact = args.split == "test" and args.output is None
    result = {
        "schema": SCHEMA,
        "verified": production_artifact,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": actual_hash,
        "split_unit": manifest["split_unit"],
        "aois": sorted({record["aoi"] for record in records}),
        "pairs_evaluated": len(records),
        "tiles_evaluated": tile_count,
        "evaluation_coverage": "deterministic_nonoverlap_full_grid_with_ignored_edge_padding",
        "split_fingerprint": split_fingerprint(records),
        "metrics": metrics,
        "limitations": ["Pixel-level footprint-mask change baseline", "No claim of operational performance without real test metrics and analyst review"],
    }
    output = args.output or (ROOT / "data" / "runs" / "spacenet7_metrics.json" if args.split == "test" else args.checkpoint.parent / "metrics_val.json")
    atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
