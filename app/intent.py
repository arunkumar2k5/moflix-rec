"""Chat intent contract and its deterministic session-vector deltas (PRD FR-11, FR-19..24).

The LLM, or `simple_parse` when it is unavailable, only fills in an `Intent`. Everything that
touches a vector happens here, bounded and pure: no network, clock or randomness.
"""

from collections.abc import Mapping
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.recommend import Constraints
from app.vectors import FULL_DIMENSION, GENRES, LAYOUT, Genre

DEFAULT_MAX_DELTA = 0.5
DEFAULT_ALPHA = 0.7
# A reaction to one pick nudges less than an explicit "more like X".
REACTION_WEIGHT = 0.5

MoodPace = Literal["lighter", "darker", "faster", "slower"]
MOOD_PACE_GENRES: dict[MoodPace, dict[Genre, float]] = {
    "lighter": {"Comedy": 1.0, "Family": 1.0, "Animation": 1.0, "Horror": -1.0, "War": -1.0},
    "darker": {"Thriller": 1.0, "Crime": 1.0, "Horror": 1.0, "Comedy": -1.0, "Family": -1.0},
    "faster": {"Action": 1.0, "Thriller": 1.0, "Adventure": 1.0, "Documentary": -1.0},
    "slower": {"Drama": 1.0, "History": 1.0, "Action": -1.0, "Thriller": -1.0},
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_by_name=True)


class Era(_Strict):
    start: int = Field(alias="from", ge=1870, le=2100)
    end: int = Field(alias="to", ge=1870, le=2100)

    @model_validator(mode="after")
    def _ordered(self) -> "Era":
        if self.start > self.end:
            raise ValueError(f"era starts after it ends: {self.start} > {self.end}")
        return self


class Runtime(_Strict):
    min: int | None = Field(default=None, ge=0)  # minutes
    max: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "Runtime":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"runtime minimum exceeds maximum: {self.min} > {self.max}")
        return self


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
LanguageCode = Annotated[str, StringConstraints(pattern=r"^[a-z]{2}$")]  # ISO 639-1
Weight = Annotated[float, Field(ge=-1.0, le=1.0)]


class FilterSet(_Strict):
    genres: frozenset[Genre] = frozenset()
    terms: frozenset[Text] = frozenset()  # matched against titles and keywords


class HardFilters(_Strict):
    exclude: FilterSet = FilterSet()
    require: FilterSet = FilterSet()


class Reaction(_Strict):
    title: Text
    liked: bool


class Intent(_Strict):
    """What one chat message asks for; carries movie titles but never vectors or user data."""

    genre_adjust: dict[Genre, Weight] = Field(default_factory=dict)
    era: Era | None = None
    runtime: Runtime | None = None
    language: frozenset[LanguageCode] = frozenset()  # any of them is acceptable
    mood_pace: MoodPace | None = None
    similar_to: frozenset[Text] = frozenset()  # titles
    unlike: frozenset[Text] = frozenset()
    hard_filters: HardFilters = HardFilters()
    reaction: Reaction | None = None


def _toward(title: str, titles: Mapping[str, np.ndarray], scale: float) -> np.ndarray:
    """`scale` times the movie's vector, shrunk so its largest component is `scale` at most."""
    vector = titles.get(title.casefold())
    if vector is None:
        raise ValueError(f"unknown movie: {title}")
    peak = np.abs(vector).max()
    return np.zeros(FULL_DIMENSION) if peak == 0 else vector * (scale / peak)


def _constraints(intent: Intent) -> Constraints:
    runtime = intent.runtime or Runtime()
    filters = intent.hard_filters
    return Constraints(
        min_runtime=runtime.min,
        max_runtime=runtime.max,
        languages=intent.language or None,
        year_range=None if intent.era is None else (intent.era.start, intent.era.end),
        required_genres=frozenset(filters.require.genres),
        excluded_genres=frozenset(filters.exclude.genres),
        required_terms=filters.require.terms,
        excluded_terms=filters.exclude.terms,
    )


def apply_intent(
    session_vector: np.ndarray,
    intent: Intent,
    movie_index: Mapping[str, np.ndarray],
    max_delta: float = DEFAULT_MAX_DELTA,
) -> tuple[np.ndarray, Constraints]:
    """The session vector moved by the intent, and the hard filters it asks for.

    `movie_index` maps titles to movie vectors and is looked up case-insensitively. Every
    component of the session vector moves by `max_delta` at most; the input is not modified.
    """
    if max_delta <= 0:
        raise ValueError(f"max_delta must be positive, got {max_delta}")
    if session_vector.shape != (FULL_DIMENSION,):
        raise ValueError(f"session_vector must have {FULL_DIMENSION} components")
    titles = {title.casefold(): vector for title, vector in movie_index.items()}
    delta = np.zeros(FULL_DIMENSION)
    genre_weights = dict(intent.genre_adjust)
    if intent.mood_pace is not None:
        for genre, weight in MOOD_PACE_GENRES[intent.mood_pace].items():
            genre_weights[genre] = genre_weights.get(genre, 0.0) + weight
    for genre, weight in genre_weights.items():
        delta[LAYOUT["genres"].start + GENRES.index(genre)] += weight * max_delta
    for title in sorted(intent.similar_to):
        delta += _toward(title, titles, max_delta)
    for title in sorted(intent.unlike):
        delta -= _toward(title, titles, max_delta)
    if intent.reaction is not None:
        sign = 1.0 if intent.reaction.liked else -1.0
        delta += _toward(intent.reaction.title, titles, sign * REACTION_WEIGHT * max_delta)
    return session_vector + np.clip(delta, -max_delta, max_delta), _constraints(intent)


def blend(base: np.ndarray, session: np.ndarray, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """`alpha·base + (1−alpha)·session`: alpha 1 keeps only the long-term taste."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be within [0, 1], got {alpha}")
    if base.shape != session.shape:
        raise ValueError(f"base and session shapes differ: {base.shape} != {session.shape}")
    return alpha * base + (1 - alpha) * session
