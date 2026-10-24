"""Weighted base taste vector, hard filters, MMR diversity and template reasons (PRD FR-13..17).

Everything here reads SQLite and computes with NumPy only: ranking never touches the network.
"""

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Literal, get_args

import numpy as np

from app.db import decode_vector, load_movies, store_taste_vector
from app.taste import DEFAULT_PARAMS, WatchEvent, build_base_taste_vector, event_weight
from app.vectors import FULL_DIMENSION, GENRES, LAYOUT, VECTOR_VERSION, Movie

Similarity = Literal["dot", "cosine"]


class RecommendError(Exception):
    pass


class NoWatchHistory(RecommendError):
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


@dataclass(frozen=True, eq=False)
class Library:
    """Candidate movies with their vectors stacked once, so ranking copies no vectors."""

    candidates: tuple[Candidate, ...]
    by_id: dict[int, Candidate]
    ids: np.ndarray
    matrix: np.ndarray  # one row per candidate, in `candidates` order
    unit: np.ndarray  # `matrix` rows scaled to unit length

    @classmethod
    def of(cls, candidates: Sequence[Candidate]) -> "Library":
        matrix = np.stack([c.vector for c in candidates]) if candidates else np.empty((0, 0))
        # Point each candidate at its row so the library holds one copy of every vector.
        rows = tuple(Candidate(c.movie, row) for c, row in zip(candidates, matrix, strict=True))
        return cls(
            candidates=rows,
            by_id={c.movie.id: c for c in rows},
            ids=np.array([c.movie.id for c in rows], dtype=np.int64),
            matrix=matrix,
            unit=_unit_rows(matrix),
        )

    def vectors(self) -> dict[int, np.ndarray]:
        return {movie_id: c.vector for movie_id, c in self.by_id.items()}


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
    library: Library,
    constraints: Constraints,
    params: RankParams,
) -> list[Recommendation]:
    """The top candidates the user has no event for and that pass every constraint.

    Picked greedily by MMR; equal objectives go to the higher score, then the lower movie id.
    """
    eligible = ~np.isin(library.ids, list(user_ctx.excluded_ids))
    if constraints != NO_CONSTRAINTS:
        eligible &= [_passes(c.movie, constraints) for c in library.candidates]
    rows = np.flatnonzero(eligible)
    if not len(rows):
        return []
    scores = _scores(user_ctx.taste, library, params.similarity)
    order = rows[np.lexsort((library.ids[rows], -scores[rows]))]
    picks = order[_mmr(scores[order], library.unit, order, params.mmr_lambda, params.top_n)]
    return [
        Recommendation(
            movie_id=library.candidates[i].movie.id,
            title=library.candidates[i].movie.title,
            year=library.candidates[i].movie.year,
            score=float(scores[i]),
            reason=_reason(library.candidates[i], user_ctx),
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


def _scores(taste: np.ndarray, library: Library, similarity: Similarity) -> np.ndarray:
    if similarity == "cosine":
        return library.unit @ _unit_rows(taste)
    return library.matrix @ taste


def _mmr(
    scores: np.ndarray, unit: np.ndarray, rows: np.ndarray, lam: float, top_n: int
) -> list[int]:
    """Greedy MMR picks as indices into `scores`, which must be sorted best first.

    `rows` maps each score to its row of `unit`, the unit-length candidate vectors.

    Relevance is min-max scaled to [0, 1] so lambda trades it against item-item cosine on the
    same scale whatever the similarity. Redundancy is always cosine: it is about direction.
    """
    spread = scores.max() - scores.min()
    relevance = (scores - scores.min()) / spread if spread > 0 else np.ones_like(scores)
    redundancy = np.zeros_like(scores)
    available = np.ones(len(scores), dtype=bool)
    picks: list[int] = []
    for _ in range(min(top_n, len(scores))):
        objective = np.where(available, lam * relevance - (1 - lam) * redundancy, -np.inf)
        pick = int(np.argmax(objective))
        picks.append(pick)
        available[pick] = False
        redundancy = np.maximum(redundancy, (unit @ unit[rows[pick]])[rows])
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


def load_library(conn: sqlite3.Connection) -> Library:
    rows = conn.execute(
        "SELECT movie_id, vector_version, dimension, vector FROM movie_vectors ORDER BY movie_id"
    ).fetchall()
    stale = sum(version != VECTOR_VERSION or dim != FULL_DIMENSION for _, version, dim, _ in rows)
    if stale:
        raise RecommendError(
            f"{stale} movie vectors are not version {VECTOR_VERSION}; rebuild them first"
        )
    movies = {movie.id: movie for movie in load_movies(conn)}
    return Library.of([Candidate(movies[row[0]], decode_vector(row[3])) for row in rows])


def _library_fingerprint(conn: sqlite3.Connection) -> tuple:
    return conn.execute(
        "SELECT COUNT(*), MAX(movie_id), TOTAL(vector_version) FROM movie_vectors"
    ).fetchone()


class LibraryCache:
    """Keeps the library in memory, reloading it when movie vectors are added or rebuilt.

    A sync in another process only ever inserts movies with their vectors, and a version bump
    rewrites every vector, so the fingerprint catches both. In-place metadata edits that leave
    the vectors alone (such as a poster backfill) show up only after a restart.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._fingerprint: tuple | None = None
        self._library: Library | None = None

    def get(self, conn: sqlite3.Connection) -> Library:
        fingerprint = _library_fingerprint(conn)
        with self._lock:
            if self._library is None or fingerprint != self._fingerprint:
                self._library = load_library(conn)
                self._fingerprint = fingerprint
            return self._library


def recommend_for_user(
    conn: sqlite3.Connection,
    user_id: int,
    library: Library,
    params: RankParams = DEFAULT_RANK_PARAMS,
    constraints: Constraints = NO_CONSTRAINTS,
    now: datetime | None = None,
) -> list[Recommendation]:
    """Rank the user's unseen movies, recording the taste vector built at `now`."""
    now = now or datetime.now(timezone.utc)
    events = _watch_events(conn, user_id)
    if not events:
        raise NoWatchHistory("no watch history to build a taste vector from")
    taste = build_base_taste_vector(events, library.vectors(), now)
    with conn:
        store_taste_vector(conn, user_id, VECTOR_VERSION, taste, now)
    candidates = library.by_id
    liked_ids = {e.movie_id for e in events if event_weight(e, now, DEFAULT_PARAMS) > 0}
    excluded_ids = frozenset(event.movie_id for event in events)
    user_ctx = UserContext(
        taste=taste,
        excluded_ids=excluded_ids,
        liked=tuple(candidates[movie_id] for movie_id in sorted(liked_ids)),
        passed=tuple(candidates[movie_id] for movie_id in sorted(excluded_ids - liked_ids)),
    )
    return rank(user_ctx, library, constraints, params)


def recommend(
    conn: sqlite3.Connection,
    user_name: str,
    params: RankParams = DEFAULT_RANK_PARAMS,
    constraints: Constraints = NO_CONSTRAINTS,
    now: datetime | None = None,
) -> list[Recommendation]:
    """`recommend_for_user` by name, loading the library from scratch."""
    user_id = _user_id(conn, user_name)
    try:
        return recommend_for_user(conn, user_id, load_library(conn), params, constraints, now)
    except NoWatchHistory as error:
        raise NoWatchHistory(f"user {user_name} has {error}") from None
