from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.taste import TasteParams, WatchEvent, build_base_taste_vector, event_weight

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
PARAMS = TasteParams()
VECTORS = {1: np.array([1.0, 0.0, 0.0]), 2: np.array([0.0, 1.0, 0.0]), 3: np.array([3.0, 2.0, 1.0])}


def test_a_favourite_weighs_more_than_the_same_movie_unfavourited():
    plain = WatchEvent(movie_id=1, completion=0.9)
    assert event_weight(replace(plain, favourite=True), NOW, PARAMS) > event_weight(
        plain, NOW, PARAMS
    )
    assert event_weight(plain, NOW, PARAMS) > 0


def test_a_rewatched_movie_weighs_more():
    once = WatchEvent(movie_id=1, play_count=1)
    assert event_weight(replace(once, play_count=3), NOW, PARAMS) > event_weight(once, NOW, PARAMS)


def test_recency_halves_the_weight_every_half_life():
    params = TasteParams(half_life_days=180)
    today = WatchEvent(movie_id=1, watched_at=NOW)
    a_year_ago = replace(today, watched_at=NOW - timedelta(days=360))
    ratio = event_weight(a_year_ago, NOW, params) / event_weight(today, NOW, params)
    assert ratio == pytest.approx(0.25, abs=1e-6)


def test_an_abandoned_movie_pulls_the_taste_away_from_itself():
    liked = WatchEvent(movie_id=1)
    base = build_base_taste_vector([liked], VECTORS, NOW, PARAMS)
    with_abandoned = build_base_taste_vector(
        [liked, WatchEvent(movie_id=2, completion=0.1)], VECTORS, NOW, PARAMS
    )
    assert np.dot(with_abandoned - base, VECTORS[2]) < 0


def test_not_interested_is_negative_even_for_a_favourite():
    event = WatchEvent(movie_id=1, favourite=True, not_interested=True)
    assert event_weight(event, NOW, PARAMS) < 0


def test_default_signals_give_the_unweighted_mean():
    taste = build_base_taste_vector([WatchEvent(1), WatchEvent(3)], VECTORS, NOW, PARAMS)
    np.testing.assert_allclose(taste, [2.0, 1.0, 0.5])


def test_same_inputs_give_byte_identical_vectors():
    events = [
        WatchEvent(1, completion=0.7, play_count=2, watched_at=NOW - timedelta(days=40)),
        WatchEvent(2, completion=0.05),
        WatchEvent(3, favourite=True, watched_at=NOW - timedelta(days=400)),
    ]
    first = build_base_taste_vector(events, VECTORS, NOW, PARAMS)
    second = build_base_taste_vector(list(events), dict(VECTORS), NOW, TasteParams())
    assert first.tobytes() == second.tobytes()


def test_no_events_is_an_error():
    with pytest.raises(ValueError, match="at least one watch event"):
        build_base_taste_vector([], VECTORS, NOW, PARAMS)
