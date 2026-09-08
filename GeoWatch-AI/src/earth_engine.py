"""Optional Google Earth Engine imagery adapter.

The module deliberately does not authenticate on import.  Earth Engine is an
optional data source and the rest of GeoWatch must continue to work when the
``earthengine-api`` package or Google credentials are unavailable.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from itertools import product
from typing import Any, Mapping, Protocol, Sequence


DEFAULT_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"


class EarthEngineError(RuntimeError):
    """Base class for recoverable imagery-source failures."""


class EarthEngineUnavailableError(EarthEngineError):
    """Raised when the optional Earth Engine dependency is not installed."""


class EarthEngineConfigurationError(EarthEngineError):
    """Raised when required local configuration is missing or invalid."""


class EarthEngineInitializationError(EarthEngineError):
    """Raised when Google rejects or cannot complete initialization."""


class EarthEngineQueryError(EarthEngineError):
    """Raised when a remote collection query fails."""


class SceneNotFoundError(EarthEngineError):
    """Raised when no chronologically valid before/after pair is available."""


@dataclass(frozen=True)
class EarthEngineSettings:
    project_id: str | None = None
    collection_id: str = DEFAULT_COLLECTION
    max_cloud_percent: float = 35.0
    search_window_days: int = 14
    candidate_limit: int = 20

    @classmethod
    def from_env(cls) -> "EarthEngineSettings":
        project = os.getenv("EARTHENGINE_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        return cls(
            project_id=project.strip() if project and project.strip() else None,
            collection_id=os.getenv("GEOWATCH_EE_COLLECTION", DEFAULT_COLLECTION),
            max_cloud_percent=float(os.getenv("GEOWATCH_EE_MAX_CLOUD", "35")),
            search_window_days=int(os.getenv("GEOWATCH_EE_SEARCH_DAYS", "14")),
            candidate_limit=int(os.getenv("GEOWATCH_EE_CANDIDATE_LIMIT", "20")),
        )


@dataclass(frozen=True)
class EarthEngineStatus:
    dependency_available: bool
    project_configured: bool
    initialized: bool
    ready: bool
    project_id: str | None
    collection_id: str
    detail: str


@dataclass(frozen=True)
class GeoArea:
    """A WGS-84 Polygon or MultiPolygon accepted by ``ee.Geometry``."""

    geojson: Mapping[str, Any]

    def __post_init__(self) -> None:
        geometry_type = self.geojson.get("type")
        coordinates = self.geojson.get("coordinates")
        if geometry_type not in {"Polygon", "MultiPolygon"} or not coordinates:
            raise ValueError("AOI must be a non-empty GeoJSON Polygon or MultiPolygon")

    @classmethod
    def from_bbox(cls, west: float, south: float, east: float, north: float) -> "GeoArea":
        values = (west, south, east, north)
        if not all(isinstance(value, (int, float)) for value in values):
            raise ValueError("bbox coordinates must be numeric")
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("bbox must be valid WGS-84 coordinates with west < east and south < north")
        ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
        return cls({"type": "Polygon", "coordinates": [ring]})


@dataclass(frozen=True)
class ScenePairRequest:
    aoi: GeoArea
    before_date: date
    after_date: date
    search_window_days: int | None = None
    max_cloud_percent: float | None = None
    candidate_limit: int | None = None

    def __post_init__(self) -> None:
        if self.before_date >= self.after_date:
            raise ValueError("before_date must be earlier than after_date")
        if self.search_window_days is not None and self.search_window_days < 0:
            raise ValueError("search_window_days cannot be negative")
        if self.max_cloud_percent is not None and not 0 <= self.max_cloud_percent <= 100:
            raise ValueError("max_cloud_percent must be between 0 and 100")
        if self.candidate_limit is not None and self.candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")


@dataclass(frozen=True)
class Sentinel2Scene:
    scene_id: str
    acquired_at: datetime
    cloud_percent: float | None
    collection_id: str = DEFAULT_COLLECTION

    def __post_init__(self) -> None:
        if self.acquired_at.tzinfo is None:
            raise ValueError("acquired_at must be timezone-aware")


@dataclass(frozen=True)
class Sentinel2ScenePair:
    before: Sentinel2Scene
    after: Sentinel2Scene
    requested_before: date
    requested_after: date
    before_candidates: int
    after_candidates: int
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def gap_days(self) -> float:
        return (self.after.acquired_at - self.before.acquired_at).total_seconds() / 86_400


class EarthEngineGateway(Protocol):
    def initialize(self, project_id: str) -> None: ...

    def search_sentinel2(
        self,
        *,
        aoi: GeoArea,
        start: date,
        end: date,
        collection_id: str,
        max_cloud_percent: float,
        limit: int,
    ) -> Sequence[Sentinel2Scene]: ...


class GoogleEarthEngineGateway:
    """Thin wrapper around ``earthengine-api`` with a single metadata request.

    Supplying ``ee_module`` is intended for tests. In normal use the package is
    imported only when ``initialize`` or a query is explicitly requested.
    """

    def __init__(self, ee_module: Any | None = None) -> None:
        self._ee = ee_module

    def _module(self) -> Any:
        if self._ee is not None:
            return self._ee
        try:
            self._ee = importlib.import_module("ee")
        except ImportError as exc:
            raise EarthEngineUnavailableError(
                "Install the optional dependency with: pip install earthengine-api"
            ) from exc
        return self._ee

    def initialize(self, project_id: str) -> None:
        try:
            self._module().Initialize(project=project_id)
        except EarthEngineUnavailableError:
            raise
        except Exception as exc:  # Google exposes multiple credential/transport errors.
            raise EarthEngineInitializationError(
                "Earth Engine initialization failed. Authenticate outside GeoWatch "
                "with `earthengine authenticate`, then verify the Cloud project."
            ) from exc

    def search_sentinel2(
        self,
        *,
        aoi: GeoArea,
        start: date,
        end: date,
        collection_id: str,
        max_cloud_percent: float,
        limit: int,
    ) -> Sequence[Sentinel2Scene]:
        ee = self._module()
        try:
            geometry = ee.Geometry(dict(aoi.geojson))
            collection = (
                ee.ImageCollection(collection_id)
                .filterBounds(geometry)
                .filterDate(start.isoformat(), (end + timedelta(days=1)).isoformat())
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_percent))
                .sort("CLOUDY_PIXEL_PERCENTAGE")
                .limit(limit)
            )
            # Only three scalar properties per scene cross the network. Imagery is
            # exported separately by a future tile/export service.
            summary = collection.reduceColumns(
                ee.Reducer.toList(3),
                ["system:index", "system:time_start", "CLOUDY_PIXEL_PERCENTAGE"],
            ).getInfo()
            rows = summary.get("list", []) if isinstance(summary, Mapping) else []
            return tuple(_scene_from_row(row, collection_id) for row in rows)
        except EarthEngineError:
            raise
        except Exception as exc:
            raise EarthEngineQueryError(
                f"Earth Engine query failed for {start.isoformat()}..{end.isoformat()}"
            ) from exc


def _scene_from_row(row: Sequence[Any], collection_id: str) -> Sentinel2Scene:
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) < 2:
        raise EarthEngineQueryError("Earth Engine returned malformed scene metadata")
    scene_index = str(row[0])
    try:
        acquired = datetime.fromtimestamp(float(row[1]) / 1000, tz=UTC)
        cloud = None if len(row) < 3 or row[2] is None else float(row[2])
    except (TypeError, ValueError, OSError) as exc:
        raise EarthEngineQueryError("Earth Engine returned invalid scene metadata") from exc
    return Sentinel2Scene(
        scene_id=f"{collection_id}/{scene_index}",
        acquired_at=acquired,
        cloud_percent=cloud,
        collection_id=collection_id,
    )


class EarthEngineAdapter:
    """Application-facing facade for status, initialization and pair search."""

    def __init__(
        self,
        settings: EarthEngineSettings | None = None,
        gateway: EarthEngineGateway | None = None,
    ) -> None:
        self.settings = settings or EarthEngineSettings.from_env()
        self._gateway = gateway
        self._gateway_injected = gateway is not None
        self._initialized = False
        self._project_id: str | None = None

    def status(self) -> EarthEngineStatus:
        dependency_available = self._gateway_injected or self._dependency_available()
        project_id = self._project_id or self.settings.project_id
        project_configured = bool(project_id)
        ready = dependency_available and project_configured and self._initialized
        if ready:
            detail = "Earth Engine is initialized and ready for Sentinel-2 metadata queries."
        elif not dependency_available:
            detail = "Optional package earthengine-api is not installed."
        elif not project_configured:
            detail = "Set EARTHENGINE_PROJECT or GOOGLE_CLOUD_PROJECT."
        else:
            detail = "Configured but not initialized; credentials have not been checked."
        return EarthEngineStatus(
            dependency_available=dependency_available,
            project_configured=project_configured,
            initialized=self._initialized,
            ready=ready,
            project_id=project_id,
            collection_id=self.settings.collection_id,
            detail=detail,
        )

    @staticmethod
    def _dependency_available() -> bool:
        try:
            return importlib.util.find_spec("ee") is not None
        except (ImportError, ValueError):
            return False

    def initialize(self, project_id: str | None = None) -> EarthEngineStatus:
        resolved_project = (project_id or self.settings.project_id or "").strip()
        if not resolved_project:
            raise EarthEngineConfigurationError(
                "Earth Engine project is required; set EARTHENGINE_PROJECT or pass project_id."
            )
        if self._gateway is None:
            self._gateway = GoogleEarthEngineGateway()
        self._gateway.initialize(resolved_project)
        self._initialized = True
        self._project_id = resolved_project
        return self.status()

    def search_scene_pair(self, request: ScenePairRequest) -> Sentinel2ScenePair:
        if not self._initialized or self._gateway is None:
            raise EarthEngineInitializationError("Call initialize() before searching scenes.")

        window_days = (
            request.search_window_days
            if request.search_window_days is not None
            else self.settings.search_window_days
        )
        max_cloud = (
            request.max_cloud_percent
            if request.max_cloud_percent is not None
            else self.settings.max_cloud_percent
        )
        limit = request.candidate_limit or self.settings.candidate_limit
        if window_days < 0 or not 0 <= max_cloud <= 100 or limit < 1:
            raise EarthEngineConfigurationError("Invalid Earth Engine search settings")

        before = self._gateway.search_sentinel2(
            aoi=request.aoi,
            start=request.before_date - timedelta(days=window_days),
            end=request.before_date + timedelta(days=window_days),
            collection_id=self.settings.collection_id,
            max_cloud_percent=max_cloud,
            limit=limit,
        )
        after = self._gateway.search_sentinel2(
            aoi=request.aoi,
            start=request.after_date - timedelta(days=window_days),
            end=request.after_date + timedelta(days=window_days),
            collection_id=self.settings.collection_id,
            max_cloud_percent=max_cloud,
            limit=limit,
        )
        selected_before, selected_after = _select_chronological_pair(
            before, after, request.before_date, request.after_date
        )

        warnings: list[str] = []
        if selected_before.acquired_at.date() != request.before_date:
            warnings.append("before scene is the nearest available date, not the exact requested date")
        if selected_after.acquired_at.date() != request.after_date:
            warnings.append("after scene is the nearest available date, not the exact requested date")
        return Sentinel2ScenePair(
            before=selected_before,
            after=selected_after,
            requested_before=request.before_date,
            requested_after=request.after_date,
            before_candidates=len(before),
            after_candidates=len(after),
            warnings=tuple(warnings),
        )


def _select_chronological_pair(
    before_candidates: Sequence[Sentinel2Scene],
    after_candidates: Sequence[Sentinel2Scene],
    before_target: date,
    after_target: date,
) -> tuple[Sentinel2Scene, Sentinel2Scene]:
    valid_pairs = [
        pair
        for pair in product(before_candidates, after_candidates)
        if pair[0].scene_id != pair[1].scene_id and pair[0].acquired_at < pair[1].acquired_at
    ]
    if not valid_pairs:
        raise SceneNotFoundError(
            "No chronological Sentinel-2 pair matched the AOI, date windows and cloud limit."
        )

    def score(pair: tuple[Sentinel2Scene, Sentinel2Scene]) -> tuple[float, float, str, str]:
        before, after = pair
        date_error = abs((before.acquired_at.date() - before_target).days) + abs(
            (after.acquired_at.date() - after_target).days
        )
        cloud_score = (before.cloud_percent or 0.0) + (after.cloud_percent or 0.0)
        return float(date_error), cloud_score, before.scene_id, after.scene_id

    return min(valid_pairs, key=score)


__all__ = [
    "DEFAULT_COLLECTION",
    "EarthEngineAdapter",
    "EarthEngineConfigurationError",
    "EarthEngineError",
    "EarthEngineInitializationError",
    "EarthEngineQueryError",
    "EarthEngineSettings",
    "EarthEngineStatus",
    "EarthEngineUnavailableError",
    "GeoArea",
    "GoogleEarthEngineGateway",
    "SceneNotFoundError",
    "ScenePairRequest",
    "Sentinel2Scene",
    "Sentinel2ScenePair",
]
