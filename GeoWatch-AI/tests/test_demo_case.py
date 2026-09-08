import numpy as np

from src.demo_case import annotate_demo, load_demo_temporal_case


def test_demo_case_is_deterministic_and_marks_a_real_pixel_change():
    first = load_demo_temporal_case()
    second = load_demo_temporal_case()

    assert first.before.shape == first.after.shape == (600, 960, 3)
    assert np.array_equal(first.before, second.before)
    assert not np.array_equal(first.before, first.after)
    assert first.after_date > first.before_date
    assert first.events[0].status == "appeared"
    assert "не являются метрикой" in first.disclaimer


def test_demo_annotation_does_not_mutate_the_source_image():
    case = load_demo_temporal_case()
    source = case.after.copy()
    annotated = annotate_demo(case.after, case.events)

    assert np.array_equal(case.after, source)
    assert annotated.size == (960, 600)
