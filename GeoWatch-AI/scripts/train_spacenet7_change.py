"""Train the reproducible Tiny U-Net SpaceNet 7 change baseline."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.spacenet7_baseline import SCHEMA, atomic_json, build_model, evaluate_loader, make_dataset, make_eval_dataset, seed_everything, sha256, split_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "spacenet7_change_baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--samples-per-pair", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0, help="Use 0 on Windows for deterministic loading")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.patch_size % 4:
        raise SystemExit("--patch-size must be divisible by 4")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required: install torch before training") from exc

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    seed_everything(args.seed, torch)
    manifest_path = args.manifest.resolve()
    train_data, train_records, manifest = make_dataset(torch, manifest_path, "train", args.patch_size, args.samples_per_pair, args.seed)
    print(json.dumps({"stage": "indexing_validation", "train_pairs": len(train_records), "device": str(device)}), flush=True)
    val_data, val_records, _, val_tiles = make_eval_dataset(torch, manifest_path, "val", args.patch_size)
    print(json.dumps({"stage": "training_ready", "val_pairs": len(val_records), "val_tiles": val_tiles}), flush=True)
    # Samples are grouped by temporal pair so the dataset can reuse decoded
    # GeoTIFFs across that pair's patches. Patch locations remain deterministic.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    model = build_model(torch, args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.2, 1.0, 1.0], device=device))
    args.output.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    history = []
    config = vars(args) | {"manifest": str(manifest_path), "output": str(args.output.resolve()), "resolved_device": str(device)}

    for epoch in range(1, args.epochs + 1):
        print(json.dumps({"stage": "epoch_start", "epoch": epoch, "epochs": args.epochs}), flush=True)
        model.train()
        losses = []
        for images, target in train_loader:
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate_loader(torch, model, val_loader, device)
        score = validation["macro_change_f1"]
        comparable_score = -1.0 if score is None else score
        row = {"epoch": epoch, "train_loss": sum(losses) / max(len(losses), 1), "val_macro_change_f1": score}
        history.append(row)
        state = {"schema": SCHEMA, "epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "config": config, "class_names": list(validation["classes"]), "manifest_sha256": sha256(manifest_path)}
        torch.save(state, args.output / "last.pt")
        if comparable_score > best_score:
            best_score = comparable_score
            torch.save(state, args.output / "best.pt")
            atomic_json(args.output / "metrics_val.json", {"split": "val", "epoch": epoch, **validation})
        print(json.dumps(row, ensure_ascii=False), flush=True)

    provenance = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "trained_baseline_not_promoted",
        "config": config,
        "dataset": {"manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "source": manifest["source"], "split_unit": manifest["split_unit"], "counts": manifest["counts"], "train_aois": sorted({r["aoi"] for r in train_records}), "val_aois": sorted({r["aoi"] for r in val_records}), "train_fingerprint": split_fingerprint(train_records), "val_fingerprint": split_fingerprint(val_records), "val_tiles": val_tiles},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)},
        "history": history,
        "test_metrics_present": False,
        "note": "Validation selected best.pt. Run evaluate_spacenet7_change.py on test exactly once for final reporting.",
    }
    provenance["artifacts"] = {
        "best_checkpoint": str((args.output / "best.pt").resolve()),
        "best_checkpoint_sha256": sha256(args.output / "best.pt"),
        "last_checkpoint": str((args.output / "last.pt").resolve()),
        "last_checkpoint_sha256": sha256(args.output / "last.pt"),
        "validation_metrics": str((args.output / "metrics_val.json").resolve()),
    }
    atomic_json(args.output / "run_manifest.json", provenance)
    print(f"Best checkpoint: {args.output / 'best.pt'}")


if __name__ == "__main__":
    main()
