from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class WatchZone:
    zone_id: str
    name: str
    west: float
    south: float
    east: float
    north: float
    cadence_days: int = 5
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-f0-9]{24}", self.zone_id):
            raise ValueError("Invalid zone ID")
        if not self.name.strip() or len(self.name) > 120:
            raise ValueError("Zone name must contain 1..120 characters")
        if not (-180 <= self.west < self.east <= 180 and -90 <= self.south < self.north <= 90):
            raise ValueError("Invalid WGS-84 bounding box")
        if not 1 <= self.cadence_days <= 365:
            raise ValueError("Cadence must be between 1 and 365 days")


class WatchZoneStore:
    def __init__(self, root: Path):
        self.root = root / "watch_zones"
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, name: str, west: float, south: float, east: float, north: float, cadence_days: int = 5) -> WatchZone:
        record = WatchZone(uuid4().hex[:24], name.strip(), west, south, east, north, cadence_days)
        self._write(record)
        return record

    def _path(self, zone_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{24}", zone_id):
            raise KeyError("Invalid zone ID")
        return self.root / f"zone_{zone_id}.json"

    def _write(self, record: WatchZone) -> None:
        target = self._path(record.zone_id)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    def list(self) -> list[WatchZone]:
        records = []
        for path in self.root.glob("zone_*.json"):
            try:
                records.append(WatchZone(**json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return sorted(records, key=lambda item: (not item.enabled, item.name.lower()))

    def set_enabled(self, zone_id: str, enabled: bool) -> WatchZone:
        record = next((item for item in self.list() if item.zone_id == zone_id), None)
        if record is None:
            raise KeyError("Watch zone not found")
        updated = WatchZone(**{**asdict(record), "enabled": enabled})
        self._write(updated)
        return updated
