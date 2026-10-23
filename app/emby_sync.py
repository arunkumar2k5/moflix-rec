"""Populate moflix-rec database from Emby Playback Reporting CSV exports + TMDB metadata.

Usage:
    export TMDB_API_KEY=your_key_here
    python -m app.cli sync --db moflix.db --data-dir data/emby/

CSV files expected in --data-dir:
    users.csv        - UserName column
    watch_events.csv - UserName, ItemId, ItemName, PlayCount, TotalMinutesWatched
    movies.csv       - ItemId, ItemName, LongestPlayMinutes, ViewerCount

TMDB responses are cached in .tmdb_cache.json so re-runs don't repeat API calls.
"""

import csv
import json
import time
from pathlib import Path

import requests

from app.db import connect, insert_movie, store_vector
from app.vectors import (
    EMBEDDING_MODEL_NAME,
    VECTOR_VERSION,
    Movie,
    _movie_text,
    build_full_vector,
    embed_texts,
)

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_CACHE = Path(".tmdb_cache.json")
# Skip plays under this threshold — accidental starts or trailers
MIN_WATCH_MINUTES = 5


class SyncError(Exception):
    pass


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:  # utf-8-sig strips the BOM
        return list(csv.DictReader(f))


def load_emby_data(data_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    users = _load_csv(data_dir / "users.csv")
    watch_events = _load_csv(data_dir / "watch_events.csv")
    movies = _load_csv(data_dir / "movies.csv")
    return users, watch_events, movies


# ---------------------------------------------------------------------------
# Movie deduplication
# ---------------------------------------------------------------------------


def deduplicate_movies(movies: list[dict]) -> dict[str, dict]:
    """Title → best movie record. Where the same title appears twice (two library
    copies), keep the one with the higher ViewerCount, falling back to the one
    with the longer LongestPlayMinutes."""
    best: dict[str, dict] = {}
    for m in movies:
        title = m["ItemName"].strip()
        if title not in best:
            best[title] = m
        else:
            prev = best[title]
            if int(m.get("ViewerCount") or 0) > int(prev.get("ViewerCount") or 0) or (
                int(m.get("ViewerCount") or 0) == int(prev.get("ViewerCount") or 0)
                and float(m.get("LongestPlayMinutes") or 0)
                > float(prev.get("LongestPlayMinutes") or 0)
            ):
                best[title] = m
    return best


# ---------------------------------------------------------------------------
# TMDB
# ---------------------------------------------------------------------------


def _load_cache() -> dict:
    if TMDB_CACHE.exists():
        return json.loads(TMDB_CACHE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    TMDB_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _tmdb_get(session: requests.Session, path: str, params: dict) -> dict:
    r = session.get(f"{TMDB_BASE}{path}", params=params, timeout=15)
    r.raise_for_status()
    time.sleep(0.1)  # stay well within 40 req/10 s
    return r.json()


def fetch_tmdb_movie(
    title: str, api_key: str, session: requests.Session, cache: dict
) -> dict | None:
    """Return full TMDB movie detail dict (with credits + keywords), or None if not found."""
    cache_key = title.lower().strip()
    if cache_key in cache:
        return cache[cache_key]

    search = _tmdb_get(session, "/search/movie", {"api_key": api_key, "query": title})
    results = search.get("results", [])
    if not results:
        print(f"  [TMDB] not found: {title}")
        cache[cache_key] = None
        return None

    best = max(results, key=lambda x: x.get("popularity", 0))
    detail = _tmdb_get(
        session,
        f"/movie/{best['id']}",
        {"api_key": api_key, "append_to_response": "credits,keywords,release_dates"},
    )
    cache[cache_key] = detail
    return detail


def _certification(tmdb: dict) -> str:
    """Extract US certification from release_dates, fallback to empty string."""
    for entry in tmdb.get("release_dates", {}).get("results", []):
        if entry.get("iso_3166_1") == "US":
            for rd in entry.get("release_dates", []):
                cert = rd.get("certification", "")
                if cert:
                    return cert
    return ""


TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def tmdb_to_movie(item_id: int, tmdb: dict) -> Movie:
    genres = [g["name"] for g in tmdb.get("genres", [])]
    runtime = int(tmdb.get("runtime") or 90)
    credits = tmdb.get("credits", {})
    director = next(
        (p["name"] for p in credits.get("crew", []) if p["job"] == "Director"),
        "Unknown",
    )
    cast = [p["name"] for p in credits.get("cast", [])[:5]]
    keywords = [k["name"] for k in tmdb.get("keywords", {}).get("keywords", [])[:15]]
    year_str = tmdb.get("release_date", "2000")
    year = int(year_str[:4]) if year_str and len(year_str) >= 4 else 2000
    language = tmdb.get("original_language", "en") or "en"
    countries = tmdb.get("production_countries", [])
    country = countries[0]["iso_3166_1"] if countries else "US"
    poster_path = tmdb.get("poster_path") or ""
    poster_url = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else ""
    return Movie(
        id=item_id,
        title=tmdb.get("title") or tmdb.get("original_title", "Unknown"),
        year=year,
        genres=genres,
        runtime=runtime,
        language=language,
        country=country,
        certification=_certification(tmdb),
        rating=float(tmdb.get("vote_average") or 0.0),
        director=director,
        cast=cast,
        overview=tmdb.get("overview", ""),
        keywords=keywords,
        poster_url=poster_url,
    )


def fallback_movie(item_id: int, title: str, longest_play: float) -> Movie:
    """Minimal Movie object for titles TMDB could not find."""
    runtime = max(int(longest_play), 1) if longest_play > 0 else 90
    return Movie(
        id=item_id,
        title=title,
        year=2000,
        genres=[],
        runtime=runtime,
        language="en",
        country="US",
        certification="",
        rating=0.0,
        director="Unknown",
        cast=[],
        overview="",
        keywords=[],
    )


# ---------------------------------------------------------------------------
# Watch event filtering
# ---------------------------------------------------------------------------


def build_watch_map(
    watch_events: list[dict], canonical_ids: dict[str, int]
) -> dict[tuple[str, int], tuple[float, int]]:
    """Return {(username, movie_id): (total_minutes, play_count)} keeping only meaningful plays."""
    result: dict[tuple[str, int], tuple[float, int]] = {}
    for row in watch_events:
        username = row["UserName"].strip()
        title = row["ItemName"].strip()
        movie_id = canonical_ids.get(title)
        if movie_id is None:
            continue
        minutes = float(row.get("TotalMinutesWatched") or 0)
        if minutes < MIN_WATCH_MINUTES:
            continue
        plays = int(row.get("PlayCount") or 0)
        key = (username, movie_id)
        total_minutes, total_plays = result.get(key, (0.0, 0))
        result[key] = (total_minutes + minutes, total_plays + plays)
    return result


# ---------------------------------------------------------------------------
# Main sync entry point
# ---------------------------------------------------------------------------


def sync(
    db_path: str | Path,
    data_dir: str | Path,
    tmdb_api_key: str,
    embedder=None,
) -> None:
    data_dir = Path(data_dir)
    print("Loading CSVs...")
    users_rows, watch_rows, movie_rows = load_emby_data(data_dir)

    # Deduplicate movies by title
    deduped = deduplicate_movies(movie_rows)
    print(f"  {len(movie_rows)} movie rows → {len(deduped)} unique titles after dedup")

    # Fetch TMDB metadata
    print("Fetching TMDB metadata (cached results skip the API)...")
    cache = _load_cache()
    session = requests.Session()
    movies: list[Movie] = []
    for title, row in deduped.items():
        item_id = int(row["ItemId"])
        longest = float(row.get("LongestPlayMinutes") or 0)
        tmdb = fetch_tmdb_movie(title, tmdb_api_key, session, cache)
        if tmdb:
            movies.append(tmdb_to_movie(item_id, tmdb))
        else:
            movies.append(fallback_movie(item_id, title, longest))

    _save_cache(cache)
    found = sum(1 for m in movies if m.director != "Unknown")
    print(f"  TMDB matched {found}/{len(movies)} movies")

    # Build canonical title → movie_id map for watch event linking
    canonical_ids: dict[str, int] = {m.title: m.id for m in movies}
    # Also map by original CSV title (may differ from TMDB title)
    for title, row in deduped.items():
        canonical_ids.setdefault(title, int(row["ItemId"]))

    # Embed all movies
    print(f"Embedding {len(movies)} movies with {EMBEDDING_MODEL_NAME}...")
    if embedder is None:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [_movie_text(m) for m in movies]
    embeddings = embed_texts(texts, model=embedder)

    # Build watch map
    watch_map = build_watch_map(watch_rows, canonical_ids)
    print(f"  {len(watch_map)} meaningful watch events (>{MIN_WATCH_MINUTES} min)")

    # Write to DB
    print(f"Writing to {db_path}...")
    conn = connect(str(db_path))
    with conn:
        # Users
        username_to_id: dict[str, int] = {}
        for row in users_rows:
            name = row["UserName"].strip()
            uid = conn.execute("INSERT OR IGNORE INTO users (name) VALUES (?)", (name,)).lastrowid
            if uid == 0:  # already existed
                uid = conn.execute("SELECT id FROM users WHERE name = ?", (name,)).fetchone()[0]
            username_to_id[name] = uid

        # Movies + vectors — insert new, backfill poster_url on existing ones that lack it
        existing = {
            row[0]: row[1] for row in conn.execute("SELECT id, poster_url FROM movies").fetchall()
        }
        inserted = 0
        poster_updated = 0
        for movie, embedding in zip(movies, embeddings, strict=True):
            if movie.id not in existing:
                insert_movie(conn, movie)
                store_vector(conn, movie.id, VECTOR_VERSION, build_full_vector(movie, embedding))
                inserted += 1
            elif movie.poster_url and not existing[movie.id]:
                conn.execute(
                    "UPDATE movies SET poster_url = ? WHERE id = ?",
                    (movie.poster_url, movie.id),
                )
                poster_updated += 1
        print(f"  Inserted {inserted} new movies, backfilled {poster_updated} poster URLs")

        # Watch events. Completion is total minutes over runtime, capped at 1, not per play:
        # PlayCount also counts partial restarts, which would make a full watch look unfinished.
        runtimes = {m.id: m.runtime for m in movies}
        we_inserted = 0
        we_updated = 0
        for (username, movie_id), (minutes, plays) in watch_map.items():
            user_id = username_to_id.get(username)
            if user_id is None:
                continue
            completion = min(1.0, minutes / runtimes[movie_id])
            play_count = max(1, plays)
            updated = conn.execute(
                "UPDATE watch_events SET completion = ?, play_count = ?"
                " WHERE user_id = ? AND movie_id = ?",
                (completion, play_count, user_id, movie_id),
            ).rowcount
            if updated:
                we_updated += 1
            else:
                conn.execute(
                    "INSERT INTO watch_events (user_id, movie_id, completion, play_count)"
                    " VALUES (?, ?, ?, ?)",
                    (user_id, movie_id, completion, play_count),
                )
                we_inserted += 1
        print(f"  Inserted {we_inserted} watch events, updated {we_updated}")

    conn.close()
    print("Sync complete.")
