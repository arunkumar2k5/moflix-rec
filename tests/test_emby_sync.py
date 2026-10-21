import csv

import pytest

from app import emby_sync
from app.db import connect
from app.emby_sync import build_watch_map, sync


def test_watch_map_sums_minutes_and_plays_of_meaningful_rows_only():
    rows = [
        {"UserName": "ann", "ItemName": "Alien", "PlayCount": "2", "TotalMinutesWatched": "200"},
        {"UserName": "ann", "ItemName": "Alien", "PlayCount": "1", "TotalMinutesWatched": "30"},
        {"UserName": "ann", "ItemName": "Alien", "PlayCount": "4", "TotalMinutesWatched": "3"},
        {"UserName": "ann", "ItemName": "Unknown", "PlayCount": "1", "TotalMinutesWatched": "90"},
    ]

    assert build_watch_map(rows, {"Alien": 7}) == {("ann", 7): (230.0, 3)}


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_emby_export(data_dir, minutes, plays):
    _write_csv(data_dir / "users.csv", ["UserName"], [{"UserName": "ann"}])
    _write_csv(
        data_dir / "movies.csv",
        ["ItemId", "ItemName", "LongestPlayMinutes", "ViewerCount"],
        [{"ItemId": "7", "ItemName": "Alien", "LongestPlayMinutes": "100", "ViewerCount": "1"}],
    )
    _write_csv(
        data_dir / "watch_events.csv",
        ["UserName", "ItemId", "ItemName", "PlayCount", "TotalMinutesWatched"],
        [
            {
                "UserName": "ann",
                "ItemId": "7",
                "ItemName": "Alien",
                "PlayCount": str(plays),
                "TotalMinutesWatched": str(minutes),
            }
        ],
    )


@pytest.fixture
def offline_sync(tmp_path, monkeypatch, fake_embedder):
    monkeypatch.setattr(emby_sync, "TMDB_CACHE", tmp_path / "tmdb_cache.json")
    monkeypatch.setattr(emby_sync, "fetch_tmdb_movie", lambda *args: None)
    db_path = tmp_path / "moflix.db"

    def run(minutes, plays):
        _write_emby_export(tmp_path, minutes, plays)
        sync(db_path, tmp_path, "unused-key", embedder=fake_embedder)
        return connect(db_path)

    return run


SIGNALS = "SELECT completion, play_count, favourite, not_interested, watched_at FROM watch_events"


def test_sync_stores_capped_completion_and_play_count(offline_sync):
    conn = offline_sync(minutes=250, plays=0)

    assert conn.execute(SIGNALS).fetchall() == [(1.0, 1, 0, 0, None)]


def test_resync_updates_signals_in_place_and_keeps_user_flags(offline_sync):
    conn = offline_sync(minutes=50, plays=1)
    assert conn.execute(SIGNALS).fetchall() == [(0.5, 1, 0, 0, None)]
    with conn:
        conn.execute(
            "UPDATE watch_events SET favourite = 1, not_interested = 1,"
            " watched_at = '2025-06-01T00:00:00+00:00'"
        )
    conn.close()

    conn = offline_sync(minutes=80, plays=2)

    assert conn.execute(SIGNALS).fetchall() == [(0.8, 2, 1, 1, "2025-06-01T00:00:00+00:00")]
