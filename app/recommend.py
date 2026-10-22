"""Weighted base taste vector, hard filters, MMR diversity and template reasons (PRD FR-13..17).

Everything here reads SQLite and computes with NumPy only: ranking never touches the network.
"""

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, get_args

import numpy as np

from app.db import decode_vector, load_movies, store_taste_vector
from app.taste import DEFAULT_PARAMS, WatchEvent, build_base_taste_vector, event_weight
from app.vectors import FULL_DIMENSION, GENRES, LAYOUT, VECTOR_VERSION, Movie

Similarity = Literal["dot", "cosine"]


class RecommendError(Exception):
    pass


@dataclass(frozen=True)
class Recommendation:
    movie_id: int
    title: str
    year: int
    score: float  # relevance under the chosen similarity, not the MMR objective
    reason: str


@dataclass(frozen=True, eq=False)
class Candidate:
    movie: Movie
    vector: np.ndarray


@dataclass(frozen=True, eq=False)
class UserContext:
    taste: np.ndarray
    excluded_ids: frozenset[int]  # every movie the user has an event for, liked or not
    liked: tuple[Candidate, ...]  # only this user's positively weighted movies, named in reasons
    passed: tuple[Candidate, ...] = ()  # this user's other movies, named only as passed on


@dataclass(frozen=True)
class Constraints:
    """Hard filters; bounds are inclusive and None or empty means unconstrained."""

    min_runtime: int | None = None
    max_runtime: int | None = None
    languages: frozenset[str] | None = None
    min_rating: float | None = None
    year_range: tuple[int, int] | None = None
    required_genres: frozenset[str] = frozenset()  # a movie must carry all of them
    excluded_genres: frozenset[str] = frozenset()
    # Terms match case-insensitively anywhere in the title or one of the keywords.
    required_terms: frozenset[str] = frozenset()  # a movie must mention all of them
    excluded_terms: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RankParams:
    similarity: Similarity = "dot"
    mmr_lambda: float = 0.7  # 1.0 is pure relevance, 0.0 pure diversity
    top_n: int = 10

    def __post_init__(self) -> None:
        if self.similarity not in get_args(Similarity):
            raise ValueError(f"unknown similarity: {self.similarity}")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError(f"mmr_lambda must be within [0, 1], got {self.mmr_lambda}")
        if self.top_n < 1:
            raise ValueError(f"top_n must be at least 1, got {self.top_n}")


DEFAULT_RANK_PARAMS = RankParams()
NO_CONSTRAINTS = Constraints()


def rank(
    user_ctx: UserContext,
    candidates: Sequence[Candidate],
    constraints: Constraints,
    params: RankParams,
) -> list[Recommendation]:
    """The top candidates the user has no event for and that pass every constraint.

    Picked greedily by MMR; equal objectives go to the higher score, then the lower movie id.
    """
    eligible = [
        candidate
        for candidate in candidates
        if candidate.movie.id not in user_ctx.excluded_ids and _passes(candidate.movie, constraints)
    ]
    if not eligible:
        return []
    ids = np.array([candidate.movie.id for candidate in eligible])
    matrix = np.stack([candidate.vector for candidate in eligible])
    scores = _scores(user_ctx.taste, matrix, params.similarity)
    order = np.lexsort((ids, -scores))
    picks = order[_mmr(scores[order], matrix[order], params.mmr_lambda, params.top_n)]
    return [
        Recommendation(
            movie_id=eligible[i].movie.id,
            title=eligible[i].movie.title,
            year=eligible[i].movie.year,
            score=float(scores[i]),
            reason=_reason(eligible[i], user_ctx),
        )
        for i in picks
    ]


def _passes(movie: Movie, constraints: Constraints) -> bool:
    c = constraints
    return (
        (c.min_runtime is None or movie.runtime >= c.min_runtime)
        and (c.max_runtime is None or movie.runtime <= c.max_runtime)
        and (not c.languages or movie.language in c.languages)
        and (c.min_rating is None or movie.rating >= c.min_rating)
        and (c.year_range is None or c.year_range[0] <= movie.year <= c.year_range[1])
        and c.required_genres.issubset(movie.genres)
        and c.excluded_genres.isdisjoint(movie.genres)
        and all(_mentions(movie, term) for term in c.required_terms)
        and not any(_mentions(movie, term) for term in c.excluded_terms)
    )


def _mentions(movie: Movie, term: str) -> bool:
    needle = term.casefold()
    return any(needle in text.casefold() for text in (movie.title, *movie.keywords))


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length; zero rows stay zero so their cosine is 0, never NaN."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix, dtype=float), where=norms > 0)


def _scores(taste: np.ndarray, matrix: np.ndarray, similarity: Similarity) -> np.ndarray:
    if similarity == "cosine":
        return _unit_rows(matrix) @ _unit_rows(taste)
    return matrix @ taste


def _mmr(scores: np.ndarray, matrix: np.ndarray, lam: float, top_n: int) -> list[int]:
    """Greedy MMR picks as indices into `scores`, which must be sorted best first.

    Relevance is min-max scaled to [0, 1] so lambda trades it against item-item cosine on the
    same scale whatever the similarity. Redundancy is always cosine: it is about direction.
    """
    spread = scores.max() - scores.min()
    relevance = (scores - scores.min()) / spread if spread > 0 else np.ones_like(scores)
    unit = _unit_rows(matrix)
    redundancy = np.zeros_like(scores)
    available = np.ones(len(scores), dtype=bool)
    picks: list[int] = []
    for _ in range(min(top_n, len(scores))):
        objective = np.where(available, lam * relevance - (1 - lam) * redundancy, -np.inf)
        pick = int(np.argmax(objective))
        picks.append(pick)
        available[pick] = False
        redundancy = np.maximum(redundancy, unit @ unit[pick])
    return picks


def _top_genre(candidate: Candidate, taste: np.ndarray) -> str | None:
    """The movie's genre contributing most to its score, if any contributes positively."""
    best, best_contribution = None, 0.0
    for genre in GENRES:
        if genre in candidate.movie.genres:
            slot = LAYOUT["genres"].start + GENRES.index(genre)
            contribution = taste[slot] * candidate.vector[slot]
            if contribution > best_contribution:
                best, best_contribution = genre, contribution
    return best


def _closest_liked(candidate: Candidate, liked: Sequence[Candidate]) -> Movie | None:
    """The liked movie most cosine-similar to the candidate, if any is positively similar."""
    best, best_similarity = None, 0.0
    target = _unit_rows(candidate.vector)
    for other in sorted(liked, key=lambda c: c.movie.id):
        similarity = float(_unit_rows(other.vector) @ target)
        if similarity > best_similarity:
            best, best_similarity = other.movie, similarity
    return best


def _reason(candidate: Candidate, user_ctx: UserContext) -> str:
    genre = _top_genre(candidate, user_ctx.taste)
    liked = _closest_liked(candidate, user_ctx.liked)
    if genre and liked:
        return f"Matches your taste for {genre}, similar to {liked.title}"
    if genre:
        return f"Matches your taste for {genre}"
    if liked:
        return f"Similar to {liked.title}, which you enjoyed"
    if candidate.movie.genres:
        return f"A {candidate.movie.genres[0]} pick close to your overall taste"
    if user_ctx.liked:
        return f"Picked for someone who enjoyed {user_ctx.liked[0].movie.title}"
    if user_ctx.passed:
        return f"A change from {user_ctx.passed[0].movie.title}, which you passed on"
    return "Close to your overall viewing taste"


def _user_id(conn: sqlite3.Connection, user_name: str) -> int:
    row = conn.execute("SELECT id FROM users WHERE name = ?", (user_name,)).fetchone()
    if row is None:
        raise RecommendError(f"unknown user: {user_name}")
    return row[0]


def _watch_events(conn: sqlite3.Connection, user_id: int) -> list[WatchEvent]:
    rows = conn.execute(
        "SELECT movie_id, completion, play_count, favourite, not_interested, watched_at"
        " FROM watch_events WHERE user_id = ? ORDER BY movie_id, id",
        (user_id,),
    )
    return [
        WatchEvent(
            movie_id=movie_id,
            completion=completion,
            play_count=play_count,
            favourite=bool(favourite),
            not_interested=bool(not_interested),
            watched_at=None if watched_at is None else datetime.fromisoformat(watched_at),
        )
        for movie_id, completion, play_count, favourite, not_interested, watched_at in rows
    ]


def _load_vectors(conn: sqlite3.Connection) -> tuple[np.ndarray, np.ndarray]:
    rows = conn.execute(
        "SELECT movie_id, vector_version, dimension, vector FROM movie_vectors ORDER BY movie_id"
    ).fetchall()
    stale = sum(version != VECTOR_VERSION or dim != FULL_DIMENSION for _, version, dim, _ in rows)
    if stale:
        raise RecommendError(
            f"{stale} movie vectors are not version {VECTOR_VERSION}; rebuild them first"
        )
    ids = np.array([row[0] for row in rows], dtype=np.int64)
    matrix = np.array([decode_vector(row[3]) for row in rows]).reshape(len(rows), FULL_DIMENSION)
    return ids, matrix


def recommend(
    conn: sqlite3.Connection,
    user_name: str,
    params: RankParams = DEFAULT_RANK_PARAMS,
    constraints: Constraints = NO_CONSTRAINTS,
    now: datetime | None = None,
) -> list[Recommendation]:
    """Rank the user's unseen movies, recording the taste vector built at `now`."""
    now = now or datetime.now(timezone.utc)
    user_id = _user_id(conn, user_name)
    events = _watch_events(conn, user_id)
    if not events:
        raise RecommendError(f"user {user_name} has no watch history to build a taste vector from")
    ids, matrix = _load_vectors(conn)
    vectors = dict(zip(ids.tolist(), matrix, strict=True))
    taste = build_base_taste_vector(events, vectors, now)
    with conn:
        store_taste_vector(conn, user_id, VECTOR_VERSION, taste, now)
    movies = {movie.id: movie for movie in load_movies(conn)}
    candidates = {movie_id: Candidate(movies[movie_id], v) for movie_id, v in vectors.items()}
    liked_ids = {e.movie_id for e in events if event_weight(e, now, DEFAULT_PARAMS) > 0}
    excluded_ids = frozenset(event.movie_id for event in events)
    user_ctx = UserContext(
        taste=taste,
        excluded_ids=excluded_ids,
        liked=tuple(candidates[movie_id] for movie_id in sorted(liked_ids)),
        passed=tuple(candidates[movie_id] for movie_id in sorted(excluded_ids - liked_ids)),
    )
    return rank(user_ctx, list(candidates.values()), constraints, params)
