import numpy as np
import pytest
from pydantic import ValidationError

from app.intent import DEFAULT_MAX_DELTA, Intent, Reaction, apply_intent, blend
from app.vectors import EMBEDDING_DIM, FULL_DIMENSION, GENRES, LAYOUT, Movie, build_full_vector

ZERO = np.zeros(FULL_DIMENSION)


def _genre_slot(genre: str) -> int:
    return LAYOUT["genres"].start + GENRES.index(genre)


@pytest.fixture
def movie_index(library) -> dict[str, np.ndarray]:
    embedding = np.full(EMBEDDING_DIM, 1 / np.sqrt(EMBEDDING_DIM))
    return {m["title"]: build_full_vector(Movie(**m), embedding) for m in library["movies"]}


@pytest.fixture
def base(movie_index) -> np.ndarray:
    return movie_index["The Godfather"] + movie_index["Raiders of the Lost Ark"]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_a_genre_adjustment_raises_that_genre_and_moves_nothing_beyond_max_delta(movie_index):
    start = np.linspace(-1.0, 1.0, FULL_DIMENSION)

    session, _ = apply_intent(start, Intent(genre_adjust={"Horror": 1.0}), movie_index)

    assert session[_genre_slot("Horror")] > start[_genre_slot("Horror")]
    assert np.abs(session - start).max() <= DEFAULT_MAX_DELTA


def test_every_component_stays_within_max_delta_when_intents_stack(movie_index):
    intent = Intent(
        genre_adjust={"Horror": 1.0, "Comedy": -1.0},
        mood_pace="darker",
        similar_to={"Alien", "Blade Runner"},
        reaction=Reaction(title="Alien", liked=True),
    )

    session, _ = apply_intent(ZERO, intent, movie_index, max_delta=0.2)

    assert np.abs(session).max() == pytest.approx(0.2)
    assert np.abs(session).max() <= 0.2


def test_apply_intent_leaves_its_input_untouched(movie_index):
    start = np.ones(FULL_DIMENSION)

    apply_intent(start, Intent(genre_adjust={"Drama": -1.0}), movie_index)

    assert np.array_equal(start, np.ones(FULL_DIMENSION))


def test_an_era_becomes_a_year_range_constraint(movie_index):
    intent = Intent.model_validate({"era": {"from": 1990, "to": 1999}})

    session, constraints = apply_intent(ZERO, intent, movie_index)

    assert constraints.year_range == (1990, 1999)
    assert np.array_equal(session, ZERO)


def test_runtime_language_and_hard_filters_become_constraints(movie_index):
    intent = Intent.model_validate(
        {
            "runtime": {"max": 120},
            "language": ["fr", "ko"],
            "hard_filters": {
                "exclude": {"genres": ["Horror"], "terms": ["Marvel"]},
                "require": {"genres": ["Comedy"]},
            },
        }
    )

    _, constraints = apply_intent(ZERO, intent, movie_index)

    assert constraints.max_runtime == 120
    assert constraints.min_runtime is None
    assert constraints.languages == {"fr", "ko"}
    assert constraints.excluded_genres == {"Horror"}
    assert constraints.required_genres == {"Comedy"}
    assert constraints.excluded_terms == {"Marvel"}


def test_similar_to_strictly_raises_the_blended_cosine_to_that_movie(movie_index, base):
    alien = movie_index["Alien"]
    before = blend(base, ZERO)

    session, _ = apply_intent(ZERO, Intent(similar_to={"alien"}), movie_index)

    assert _cosine(blend(base, session), alien) > _cosine(before, alien)


def test_unlike_and_a_disliked_reaction_lower_the_blended_cosine(movie_index, base):
    alien = movie_index["Alien"]
    before = _cosine(blend(base, ZERO), alien)
    for intent in (Intent(unlike={"Alien"}), Intent(reaction={"title": "Alien", "liked": False})):
        session, _ = apply_intent(ZERO, intent, movie_index)

        assert _cosine(blend(base, session), alien) < before


def test_titles_outside_the_index_are_ignored_and_the_rest_still_applies(movie_index):
    intent = Intent.model_validate(
        {
            "genre_adjust": {"Horror": 1.0},
            "era": {"from": 1990, "to": 1999},
            "similar_to": ["Dances"],
            "unlike": ["Nope"],
            "reaction": {"title": "Nope", "liked": True},
        }
    )

    session, constraints = apply_intent(ZERO, intent, movie_index)

    expected, _ = apply_intent(ZERO, Intent(genre_adjust={"Horror": 1.0}), movie_index)
    assert np.array_equal(session, expected)
    assert constraints.year_range == (1990, 1999)


@pytest.mark.parametrize(
    ("mood", "raised", "lowered"), [("lighter", "Comedy", "Horror"), ("darker", "Horror", "Comedy")]
)
def test_a_mood_raises_its_genres_and_lowers_their_opposites(movie_index, mood, raised, lowered):
    session, _ = apply_intent(ZERO, Intent(mood_pace=mood), movie_index)

    assert session[_genre_slot(raised)] > 0 > session[_genre_slot(lowered)]


def test_a_reaction_moves_less_than_asking_for_similar(movie_index):
    similar, _ = apply_intent(ZERO, Intent(similar_to={"Alien"}), movie_index)
    reaction, _ = apply_intent(
        ZERO, Intent(reaction={"title": "Alien", "liked": True}), movie_index
    )

    assert 0 < np.linalg.norm(reaction) < np.linalg.norm(similar)
    assert _cosine(reaction, similar) == pytest.approx(1.0)


def test_a_session_vector_of_the_wrong_size_is_an_error(movie_index):
    with pytest.raises(ValueError, match="components"):
        apply_intent(np.zeros(3), Intent(), movie_index)


def test_the_same_intent_from_the_same_start_gives_identical_vectors(movie_index):
    start = np.linspace(0.0, 1.0, FULL_DIMENSION)
    intent = Intent(genre_adjust={"Western": 0.4}, mood_pace="faster", unlike={"Alien"})

    first, first_constraints = apply_intent(start, intent, movie_index)
    second, second_constraints = apply_intent(start, intent, movie_index)

    assert np.array_equal(first, second)
    assert first_constraints == second_constraints


def test_blend_at_the_extremes_returns_either_layer_exactly():
    base = np.linspace(-3.0, 3.0, FULL_DIMENSION)
    session = np.linspace(5.0, -1.0, FULL_DIMENSION)

    assert np.array_equal(blend(base, session, 1.0), base)
    assert np.array_equal(blend(base, session, 0.0), session)
    assert np.allclose(blend(base, session), 0.7 * base + 0.3 * session)


def test_blend_rejects_an_alpha_outside_the_unit_interval():
    with pytest.raises(ValueError, match="alpha"):
        blend(ZERO, ZERO, 1.5)


@pytest.mark.parametrize(
    "payload",
    [
        {"vector": [0.1, 0.2]},
        {"genre_adjust": {"Splatter": 1.0}},
        {"genre_adjust": {"Horror": 3.0}},
        {"era": {"from": 1999, "to": 1990}},
        {"runtime": {"max": 90, "min": 120}},
        {"language": ["English"]},
        {"mood_pace": "sideways"},
        {"hard_filters": {"exclude": {"terms": ["  "]}}},
        {"hard_filters": {"exclude": {"genres": ["Marvel"]}}},
        {"reaction": {"title": "Alien", "liked": True, "user": "bob"}},
    ],
)
def test_invalid_intent_json_is_rejected(payload):
    with pytest.raises(ValidationError):
        Intent.model_validate(payload)
