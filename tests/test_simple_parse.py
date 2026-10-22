import pytest

from app.intent import Intent
from app.simple_parse import simple_parse

CASES = [
    ("something from the 90s", {"era": {"from": 1990, "to": 1999}}),
    ("films from the 2000s", {"era": {"from": 2000, "to": 2009}}),
    ("I like sci-fi", {"genre_adjust": {"Science Fiction": 1.0}}),
    ("nothing over 2 hours", {"runtime": {"max": 120}}),
    ("not another Marvel", {"hard_filters": {"exclude": {"terms": ["marvel"]}}}),
    ("only horror movies please", {"hard_filters": {"require": {"genres": ["Horror"]}}}),
    (
        "no more horror, something lighter",
        {"mood_pace": "lighter", "hard_filters": {"exclude": {"genres": ["Horror"]}}},
    ),
    ("less romance please", {"genre_adjust": {"Romance": -1.0}}),
    ("I don't like westerns", {"genre_adjust": {"Western": -1.0}}),
    (
        "a Korean thriller under 100 minutes",
        {"genre_adjust": {"Thriller": 1.0}, "runtime": {"max": 100}, "language": ["ko"]},
    ),
    (
        "something like Parasite but funnier",
        {"similar_to": ["Parasite"], "genre_adjust": {"Comedy": 1.0}},
    ),
    (
        "something like Dances with Wolves and Heat",
        {"similar_to": ["Dances with Wolves and Heat"]},
    ),
    ("no more than 2 hours", {"runtime": {"max": 120}}),
    ("not another horror movie", {"hard_filters": {"exclude": {"genres": ["Horror"]}}}),
    ("hello there", {}),
]


@pytest.mark.parametrize(("text", "expected"), CASES, ids=[text for text, _ in CASES])
def test_simple_parse(text, expected):
    assert simple_parse(text) == Intent.model_validate(expected)


@pytest.mark.parametrize(
    "text",
    [
        "something similar to " + "x" * 300,
        "not another " + "x" * 300,
        "nothing over " + "9" * 400 + " minutes",
    ],
    ids=["long title", "long term", "huge runtime"],
)
def test_text_the_schema_cannot_hold_is_dropped_rather_than_raising(text):
    assert simple_parse(text) == Intent()
