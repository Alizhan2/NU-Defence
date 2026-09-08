from pathlib import Path

from scripts.verify_dataset import scene_id


def test_scene_id_groups_dota_tiles() -> None:
    assert scene_id("P1142__1024__0___824") == "P1142"
    assert scene_id("ordinary_image") == "ordinary_image"


def test_target_config_lists_four_neutral_classes() -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "dota_target.example.yaml"
    text = config.read_text(encoding="utf-8")
    for name in ("aircraft", "ship", "small vehicle", "large vehicle"):
        assert name in text
