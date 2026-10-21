"""Weighted base taste vector from one user's watch events (PRD FR-9, FR-10).

Pure: no clock reads, I/O or randomness, so equal events, vectors and `now` give equal bytes.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class TasteParams:
    abandon_threshold: float = 0.2  # completion below this is a negative signal
    half_life_days: float = 180.0
    favourite_weight: float = 2.0  # multiplier for favourites
    rewatch_weight: float = 0.5  # extra weight per natural log of play_count
    negative_weight: float = 1.0  # magnitude of an abandoned or not-interested event


@dataclass(frozen=True)
class WatchEvent:
    movie_id: int
    completion: float = 1.0
    play_count: int = 1
    favourite: bool = False
    not_interested: bool = False
    watched_at: datetime | None = None  # None: unknown, so no recency decay


DEFAULT_PARAMS = TasteParams()


def recency(watched_at: datetime | None, now: datetime, params: TasteParams) -> float:
    if watched_at is None:
        return 1.0
    age_days = max((now - watched_at).total_seconds() / SECONDS_PER_DAY, 0.0)
    return 0.5 ** (age_days / params.half_life_days)


def event_weight(event: WatchEvent, now: datetime, params: TasteParams) -> float:
    """Signed weight: negative for abandoned or not-interested movies, positive otherwise."""
    decay = recency(event.watched_at, now, params)
    if event.not_interested or event.completion < params.abandon_threshold:
        return -params.negative_weight * decay
    rewatch = 1.0 + params.rewatch_weight * math.log(max(event.play_count, 1))
    favourite = params.favourite_weight if event.favourite else 1.0
    return event.completion * rewatch * favourite * decay


def build_base_taste_vector(
    watch_events: Sequence[WatchEvent],
    movie_vectors: Mapping[int, np.ndarray],
    now: datetime,
    params: TasteParams = DEFAULT_PARAMS,
) -> np.ndarray:
    """Weighted average of the watched movies' vectors, negative signals subtracting.

    Normalised by the total absolute weight so the result stays on the scale of a movie vector.
    """
    if not watch_events:
        raise ValueError("a taste vector needs at least one watch event")
    weights = np.array([event_weight(event, now, params) for event in watch_events])
    matrix = np.stack([movie_vectors[event.movie_id] for event in watch_events])
    return weights @ matrix / np.abs(weights).sum()
