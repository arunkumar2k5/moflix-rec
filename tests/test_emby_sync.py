from app.emby_sync import build_watch_map


def test_watch_map_sums_minutes_and_plays_of_meaningful_rows_only():
    rows = [
        {"UserName": "ann", "ItemName": "Alien", "PlayCount": "2", "TotalMinutesWatched": "200"},
        {"UserName": "ann", "ItemName": "Alien", "PlayCount": "1", "TotalMinutesWatched": "30"},
        {"UserName": "ann", "ItemName": "Alien", "PlayCount": "4", "TotalMinutesWatched": "3"},
        {"UserName": "ann", "ItemName": "Unknown", "PlayCount": "1", "TotalMinutesWatched": "90"},
    ]

    assert build_watch_map(rows, {"Alien": 7}) == {("ann", 7): (230.0, 3)}
