"""Structured movie vectors: a fixed-length layout of metadata features."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Bump whenever the layout below changes; stored vectors with another version are stale.
VECTOR_VERSION = 1

GENRES = (
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "Thriller",
    "War",
    "Western",
)
# Lower bound of each decade bucket; years before the first bucket fall into it.
DECADES = (1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020)
# Upper bound (exclusive, minutes) of each runtime bucket but the last.
RUNTIME_LIMITS = (90, 120, 150)
LANGUAGES = ("en", "fr", "es", "de", "it", "ja", "ko", "zh", "hi")
COUNTRIES = ("US", "GB", "FR", "DE", "IT", "ES", "JP", "KR", "CN", "IN")
CERTIFICATIONS = ("G", "PG", "PG-13", "R", "NC-17")
MAX_RATING = 10.0
DIRECTOR_BUCKETS = 16
CAST_BUCKETS = 32
TOP_CAST = 3


@dataclass(frozen=True)
class Movie:
    id: int
    title: str
    year: int
    genres: Sequence[str]
    runtime: int
    language: str
    country: str
    certification: str
    rating: float
    director: str
    cast: Sequence[str]


def _layout() -> dict[str, slice]:
    sizes = {
        "genres": len(GENRES),
        "decade": len(DECADES),
        "runtime": len(RUNTIME_LIMITS) + 1,
        "language": len(LANGUAGES) + 1,
        "country": len(COUNTRIES) + 1,
        "certification": len(CERTIFICATIONS) + 1,
        "rating": 1,
        "director": DIRECTOR_BUCKETS,
        "cast": CAST_BUCKETS,
    }
    layout, start = {}, 0
    for name, size in sizes.items():
        layout[name] = slice(start, start + size)
        start += size
    return layout


LAYOUT = _layout()
DIMENSION = max(s.stop for s in LAYOUT.values())


def _index_or_other(vocabulary: Sequence[str], value: str) -> int:
    """Position of `value` in `vocabulary`, or the trailing "other" slot."""
    return vocabulary.index(value) if value in vocabulary else len(vocabulary)


def _decade_index(year: int) -> int:
    return max(sum(year >= start for start in DECADES) - 1, 0)


def _runtime_index(minutes: int) -> int:
    return sum(minutes >= limit for limit in RUNTIME_LIMITS)


def hash_bucket(name: str, buckets: int) -> int:
    # Python's hash() is salted per process; vectors must be stable across runs.
    digest = hashlib.sha256(name.strip().lower().encode()).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def build_movie_vector(movie: Movie) -> np.ndarray:
    vector = np.zeros(DIMENSION)

    def set_slot(feature: str, index: int, value: float = 1.0) -> None:
        vector[LAYOUT[feature].start + index] += value

    for genre in movie.genres:
        if genre in GENRES:
            set_slot("genres", GENRES.index(genre))
    set_slot("decade", _decade_index(movie.year))
    set_slot("runtime", _runtime_index(movie.runtime))
    set_slot("language", _index_or_other(LANGUAGES, movie.language))
    set_slot("country", _index_or_other(COUNTRIES, movie.country))
    set_slot("certification", _index_or_other(CERTIFICATIONS, movie.certification))
    set_slot("rating", 0, min(max(movie.rating / MAX_RATING, 0.0), 1.0))
    set_slot("director", hash_bucket(movie.director, DIRECTOR_BUCKETS))
    for actor in movie.cast[:TOP_CAST]:
        set_slot("cast", hash_bucket(actor, CAST_BUCKETS))
    return vector
