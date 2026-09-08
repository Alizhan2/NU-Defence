import pytest

from src.watch_zones import WatchZoneStore


def test_watch_zone_is_persisted_and_can_be_paused(tmp_path):
    store = WatchZoneStore(tmp_path)
    created = store.create("Aktobe demo", 57.0, 50.1, 57.3, 50.4, 5)

    assert store.list() == [created]
    paused = store.set_enabled(created.zone_id, False)
    assert paused.enabled is False


def test_watch_zone_rejects_invalid_bbox(tmp_path):
    with pytest.raises(ValueError, match="bounding box"):
        WatchZoneStore(tmp_path).create("Bad", 58, 50, 57, 51)
