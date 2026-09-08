"""Write a machine-readable audit of model, metrics and dataset evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_audit import build_audit_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "dota8_stage2" / "dota4.yaml")
    parser.add_argument("--metrics", type=Path, default=ROOT / "runs" / "metrics_test.json")
    parser.add_argument("--sample", type=Path, default=ROOT / "data" / "dota8_stage2" / "images" / "test" / "P1142__1024__0___824.jpg")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "runs" / "model_audit.json")
    args = parser.parse_args()
    report = build_audit_report(args.weights, args.data, args.metrics, args.sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["evidence_status"] == "verified" else 2)


if __name__ == "__main__":
    main()
