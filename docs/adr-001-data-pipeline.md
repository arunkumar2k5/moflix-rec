# ADR-001: Data Pipeline — Vectors, Embeddings, and Emby Sync

**Date:** 2026-10-21
**Status:** Implemented and validated
**Authors:** Arun (ak261), Claude Code

---

## Context

Issue #1 (`core-vectors-scoring`) built the skeleton with a fixture-based test dataset.
Before proceeding with the rest of the backlog, we made several decisions and implemented
a real data pipeline that are not captured in any GitHub issue. This document records what
was built, why, and how to operate it.

---

## Decision 1 — Text embeddings in v1 (resolves MISSION.md Q2)

**Decision:** Use `all-MiniLM-L6-v2` (sentence-transformers, local, CPU-only) for text embeddings.

**Why not API-based embeddings:**
- No per-call cost
- No network dependency at scoring time
- Fully offline after first model download (~80MB, cached in `~/.cache/huggingface/hub`)
- Fast enough on CPU: 736 movies embed in ~90 seconds

**Vector layout (490 dimensions total):**

| Slice | Dims | What it encodes |
|---|---|---|
| `[0:106]` | 106 | Structured metadata (genres, decade, runtime, language, country, certification, rating, director hash, cast hash) |
| `[106:490]` | 384 | `all-MiniLM-L6-v2` embedding of `overview + keywords` (normalised, float32 cast to float64) |

**Key constants in `app/vectors.py`:**
- `VECTOR_VERSION = 2` — bump this whenever the layout or model changes; old DBs will refuse to score
- `DIMENSION = 106` — structured-only
- `FULL_DIMENSION = 490` — what gets stored in `movie_vectors`
- `EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"`

**When to rebuild vectors:**
- Any change to `GENRES`, `DECADES`, `RUNTIME_LIMITS`, `LANGUAGES`, `COUNTRIES`, `CERTIFICATIONS` constants
- Switching embedding model
- Always bump `VECTOR_VERSION` and re-run sync

---

## Decision 2 — Movie metadata source: TMDB API at sync time

**Decision:** Enrich movie metadata from TMDB at sync time, not at scoring time.

**Why TMDB over Emby's own metadata:**
- Emby metadata is not directly queryable from the Playback Reporting plugin
- TMDB provides structured genres, director, cast, keywords, overview, certification, runtime in one API call
- Free tier: 40 req/10s, sufficient for batch sync
- Responses cached in `.tmdb_cache.json` — re-runs are instant, no repeated API calls

**TMDB fields used:**
`genres`, `runtime`, `release_date` (→ year), `original_language`, `production_countries`,
`vote_average`, `credits.crew` (director), `credits.cast` (top 5), `keywords.keywords` (top 15),
`release_dates` (US certification), `overview`

**TMDB hit rate on current library:** 732/736 (99.5%). The 4 misses were:
- `VTS_02_6` — DVD rip artifact, not a real title
- 3 regional titles TMDB couldn't match by title alone

**For misses:** `fallback_movie()` in `app/emby_sync.py` creates a minimal `Movie` with just
title and estimated runtime. These get a structured-zero + embedding-of-empty-string vector —
they'll score low but won't break anything.

---

## Decision 3 — Emby data source: Playback Reporting plugin CSV export

**Decision:** Use manual CSV export from the Emby Playback Reporting plugin as the data source for the initial sync. Automated Emby API sync is Issue #4 in the backlog.

**Why manual CSV for now:**
- No SSH access to the Emby server
- The Playback Reporting plugin has a custom query UI with CSV export
- Sufficient to validate the full pipeline with real data

**Three CSV files required in `data/emby/`:**

| File | Query | Key columns |
|---|---|---|
| `users.csv` | `SELECT DISTINCT UserId FROM PlaybackActivity WHERE ItemType='Movie'` with "Replace UserId with UserName" | `UserName` |
| `watch_events.csv` | Per-user per-movie aggregated play stats | `UserName, ItemId, ItemName, PlayCount, TotalMinutesWatched, LastPlayed, FirstPlayed` |
| `movies.csv` | Distinct watched movies with max play duration | `ItemId, ItemName, LongestPlayMinutes, ViewerCount` |

**Important notes on the CSV export:**
- Files have UTF-8 BOM — code uses `encoding='utf-8-sig'` to strip it
- `LastPlayed`/`FirstPlayed` export with time only (no date) — this is a plugin display bug; sync leaves `watch_events.watched_at` NULL, so recency decay does not apply until a dated source exists. `PlayCount` feeds `play_count`, and `TotalMinutesWatched` over runtime feeds `completion`
- `LongestPlayMinutes` approximates movie runtime for movies not found on TMDB

**Watch event filtering:**
- Plays under 5 minutes are discarded as accidental starts (`MIN_WATCH_MINUTES = 5`)
- Movies with the same title but different Emby ItemIds are deduplicated (keep highest ViewerCount, then longest play)

**Current dataset (as of 2026-10-21):**
- 58 users
- 787 raw movie rows → 736 unique titles after dedup
- 732 TMDB-enriched, 4 fallback
- 1,362 watch events after filtering

---

## How to re-run the sync (with a bigger database)

### Step 1 — Export fresh CSVs from Emby
Run these queries in the Emby Playback Reporting plugin with **"Replace UserId with UserName"** enabled, export as CSV, save to `data/emby/`:

**users.csv:**
```sql
SELECT DISTINCT UserId
FROM PlaybackActivity
WHERE ItemType = 'Movie'
ORDER BY UserId
```

**watch_events.csv:**
```sql
SELECT
    UserId,
    ItemId,
    ItemName,
    COUNT(*)                              AS PlayCount,
    ROUND(SUM(PlayDuration) / 60.0, 1)   AS TotalMinutesWatched,
    MAX(DateCreated)                      AS LastPlayed,
    MIN(DateCreated)                      AS FirstPlayed
FROM PlaybackActivity
WHERE ItemType = 'Movie'
GROUP BY UserId, ItemId
ORDER BY UserId, LastPlayed DESC
```

**movies.csv:**
```sql
SELECT
    ItemId,
    ItemName,
    ROUND(MAX(PlayDuration) / 60.0, 0)   AS LongestPlayMinutes,
    COUNT(DISTINCT UserId)                AS ViewerCount
FROM PlaybackActivity
WHERE ItemType = 'Movie'
GROUP BY ItemId
ORDER BY ItemName
```

### Step 2 — Run the sync

```bash
# Linux (production)
export TMDB_API_KEY=your_key
python3 -m app.cli sync --db moflix.db --data-dir data/emby/

# Windows (development)
python -m app.cli sync --db moflix.db --data-dir data/emby/ --tmdb-key YOUR_KEY
```

The TMDB cache (`.tmdb_cache.json`) is checked first — only new titles hit the API.
Copy it between machines with:
```bash
scp -P 1685 .tmdb_cache.json ak261@192.168.30.9:/home/ak261/Documents/moflix-rec/
```

### Step 3 — If you want to rebuild from scratch

```bash
rm moflix.db
python3 -m app.cli sync --db moflix.db --data-dir data/emby/
```

To also invalidate the TMDB cache (e.g. after a major library change):
```bash
rm .tmdb_cache.json moflix.db
python3 -m app.cli sync --db moflix.db --data-dir data/emby/
```

### Step 4 — Verify

```bash
python3 -m app.cli recommend --db moflix.db --user "ak261" --top 10
```

---

## File map

```
app/vectors.py       — Movie dataclass, build_movie_vector(), embed_texts(), build_full_vector()
app/db.py            — SQLite schema and read/write helpers
app/taste.py         — TasteParams, WatchEvent, build_base_taste_vector() (weighted, pure)
app/recommend.py     — Builds and stores the user's taste vector, ranks unwatched movies
app/seed.py          — JSON fixture → DB (used in tests)
app/emby_sync.py     — Emby CSV + TMDB → DB (used in production)
app/cli.py           — Entry point: seed / recommend / sync
data/emby/           — CSV exports from Emby (gitignored, contains PII)
.tmdb_cache.json     — TMDB API response cache (gitignored)
moflix.db            — Production database (gitignored)
tests/fixtures/      — Synthetic 24-movie fixture for unit tests
```

---

## What comes next (backlog)

| Issue | What it adds |
|---|---|
| ~~#2 weighted-taste-vector~~ | Done (PR #16): completion %, rewatch count, favourite, recency decay and negative signals |
| #3 filters-diversity-explanations | MMR diversity, hard filters, per-recommendation reasons |
| #4 emby-sync | Automated nightly sync via Emby HTTP API (replaces manual CSV) |
| #5 auth-sessions | Login, session cookies, user isolation |
| #6 intent-deltas | LLM intent parsing, session vector, chat tuning |

When Issue #4 lands, this document's "Step 1 — Export fresh CSVs" section becomes obsolete
and should be replaced with `python3 -m app.cli emby-pull --host <ip> --api-key <key>`.
