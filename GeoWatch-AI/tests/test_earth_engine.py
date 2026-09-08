from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from src.earth_engine import (
    DEFAULT_COLLECTION,
    EarthEngineAdapter,
    EarthEngineConfigurationError,
    EarthEngineInitializationError,
    EarthEngineSettings,
    GeoArea,
    GoogleEarthEngineGateway,
    SceneNotFoundError,
    ScenePairRequest,
    Sentinel2Scene,
)


def scene(name: str, day: int, cloud: float = 5.0) -> Sentinel2Scene:
    return Sentinel2Scene(
        scene_id=name,
        acquired_at=datetime(2026, 8, day, 10, tzinfo=UTC),
        cloud_percent=cloud,
    )


class FakeGateway:
    def __init__(self, responses: list[list[Sentinel2Scene]] | None = None) -> None:
        self.responses = list(responses or [])
        self.initialized_with: str | None = None
        self.calls: list[dict] = []

    def initialize(self, project_id: str) -> None:
        self.initialized_with = project_id

    def search_sentinel2(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def request(**overrides) -> ScenePairRequest:
    values = {
        "aoi": GeoArea.from_bbox(56.0, 50.0, 57.0, 51.0),
        "before_date": date(2026, 8, 10),
        "after_date": date(2026, 8, 20),
    }
    values.update(overrides)
    return ScenePairRequest(**values)


def test_status_is_side_effect_free_and_explains_missing_project():
    gateway = FakeGateway()
    adapter = EarthEngineAdapter(EarthEngineSettings(project_id=None), gateway)

    status = adapter.status()

    assert status.dependency_available
    assert not status.project_configured
    assert not status.initialized
    assert not status.ready
    assert gateway.initialized_with is None


def test_initialize_requires_project_and_never_authenticates_implicitly():
    adapter = EarthEngineAdapter(EarthEngineSettings(project_id=None), FakeGateway())

    with pytest.raises(EarthEngineConfigurationError):
        adapter.initialize()


def test_initialize_uses_explicit_project():
    gateway = FakeGateway()
    adapter = EarthEngineAdapter(EarthEngineSettings(project_id="env-project"), gateway)

    status = adapter.initialize("chosen-project")

    assert gateway.initialized_with == "chosen-project"
    assert status.ready
    assert status.project_id == "chosen-project"


def test_search_requires_initialization():
    adapter = EarthEngineAdapter(EarthEngineSettings(project_id="project"), FakeGateway())

    with pytest.raises(EarthEngineInitializationError):
        adapter.search_scene_pair(request())


def test_search_selects_nearest_chronological_pair_and_passes_filters():
    gateway = FakeGateway(
        [
            [scene("before-cloudier", 10, 20), scene("before-near", 9, 4)],
            [scene("after-far", 25, 1), scene("after-exact", 20, 12)],
        ]
    )
    adapter = EarthEngineAdapter(
        EarthEngineSettings(project_id="project", search_window_days=7, max_cloud_percent=30, candidate_limit=8),
        gateway,
    )
    adapter.initialize()

    pair = adapter.search_scene_pair(request())

    assert pair.before.scene_id == "before-cloudier"
    assert pair.after.scene_id == "after-exact"
    assert pair.gap_days == pytest.approx(10)
    assert pair.before_candidates == 2 and pair.after_candidates == 2
    assert pair.warnings == ()
    assert gateway.calls[0]["start"] == date(2026, 8, 3)
    assert gateway.calls[1]["end"] == date(2026, 8, 27)
    assert gateway.calls[0]["max_cloud_percent"] == 30
    assert gateway.calls[0]["limit"] == 8


def test_search_rejects_same_or_reverse_scene_pair():
    gateway = FakeGateway([[scene("same", 20)], [scene("same", 20)]])
    adapter = EarthEngineAdapter(EarthEngineSettings(project_id="project"), gateway)
    adapter.initialize()

    with pytest.raises(SceneNotFoundError):
        adapter.search_scene_pair(request())


def test_nearest_non_exact_dates_are_disclosed():
    gateway = FakeGateway([[scene("before", 9)], [scene("after", 21)]])
    adapter = EarthEngineAdapter(EarthEngineSettings(project_id="project"), gateway)
    adapter.initialize()

    pair = adapter.search_scene_pair(request())

    assert len(pair.warnings) == 2


def test_request_and_bbox_validation():
    with pytest.raises(ValueError):
        GeoArea.from_bbox(57, 50, 56, 51)
    with pytest.raises(ValueError):
        request(after_date=date(2026, 8, 10))
    with pytest.raises(ValueError):
        request(max_cloud_percent=101)


class FakeComputed:
    def getInfo(self):
        return {"list": [["T42UXV_20260810T000000", 1_786_340_400_000, 7.5]]}


class FakeCollection:
    def __init__(self) -> None:
        self.operations = []

    def filterBounds(self, value):
        self.operations.append(("bounds", value))
        return self

    def filterDate(self, start, end):
        self.operations.append(("dates", start, end))
        return self

    def filter(self, value):
        self.operations.append(("filter", value))
        return self

    def sort(self, value):
        self.operations.append(("sort", value))
        return self

    def limit(self, value):
        self.operations.append(("limit", value))
        return self

    def reduceColumns(self, reducer, selectors):
        self.operations.append(("reduce", reducer, selectors))
        return FakeComputed()


class FakeEE:
    def __init__(self) -> None:
        self.collection = FakeCollection()
        self.initialized_project = None

        class Filters:
            @staticmethod
            def lte(name, value):
                return name, value

        class Reducers:
            @staticmethod
            def toList(size):
                return "list", size

        self.Filter = Filters
        self.Reducer = Reducers

    def Initialize(self, *, project):
        self.initialized_project = project

    def Geometry(self, value):
        return value

    def ImageCollection(self, collection_id):
        assert collection_id == DEFAULT_COLLECTION
        return self.collection


def test_google_gateway_builds_metadata_only_query_without_real_network():
    fake_ee = FakeEE()
    gateway = GoogleEarthEngineGateway(fake_ee)
    gateway.initialize("demo-project")

    scenes = gateway.search_sentinel2(
        aoi=GeoArea.from_bbox(56, 50, 57, 51),
        start=date(2026, 8, 1),
        end=date(2026, 8, 10),
        collection_id=DEFAULT_COLLECTION,
        max_cloud_percent=25,
        limit=10,
    )

    assert fake_ee.initialized_project == "demo-project"
    assert scenes[0].scene_id.endswith("T42UXV_20260810T000000")
    assert scenes[0].cloud_percent == 7.5
    assert ("dates", "2026-08-01", "2026-08-11") in fake_ee.collection.operations
    assert ("filter", ("CLOUDY_PIXEL_PERCENTAGE", 25)) in fake_ee.collection.operations
