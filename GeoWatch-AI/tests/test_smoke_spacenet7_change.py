from scripts.smoke_spacenet7_change import create_fixture


def test_fixture_creation_is_idempotent(tmp_path):
    first = create_fixture(tmp_path / "fixture")
    second = create_fixture(tmp_path / "fixture")

    assert first == second
    assert second.is_file()

