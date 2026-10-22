"""Command line entry point: `python -m app.cli seed|recommend|sync`."""

import argparse
import os
import sys
from pathlib import Path
from typing import get_args

from app.db import connect
from app.recommend import DEFAULT_RANK_PARAMS, RankParams, RecommendError, Similarity, recommend
from app.seed import SeedError, load_fixture, seed


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _unit_float(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    seed_cmd = commands.add_parser("seed", help="load a JSON library fixture into a new database")
    seed_cmd.add_argument("--db", required=True)
    seed_cmd.add_argument("--fixture", required=True)

    rec_cmd = commands.add_parser("recommend", help="print a user's top unwatched movies")
    rec_cmd.add_argument("--db", required=True)
    rec_cmd.add_argument("--user", required=True)
    rec_cmd.add_argument("--top", type=_positive_int, default=10)
    rec_cmd.add_argument(
        "--similarity", choices=get_args(Similarity), default=DEFAULT_RANK_PARAMS.similarity
    )
    rec_cmd.add_argument(
        "--mmr-lambda",
        type=_unit_float,
        default=DEFAULT_RANK_PARAMS.mmr_lambda,
        help="1 ranks by relevance alone, lower values favour diverse picks",
    )

    sync_cmd = commands.add_parser("sync", help="populate DB from Emby CSV exports + TMDB")
    sync_cmd.add_argument("--db", required=True)
    sync_cmd.add_argument("--data-dir", required=True)
    sync_cmd.add_argument(
        "--tmdb-key",
        default=None,
        help="TMDB API key (defaults to TMDB_API_KEY env var)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "recommend" and not Path(args.db).is_file():
        print(f"error: database not found: {args.db}", file=sys.stderr)
        return 1
    if args.command == "sync":
        from app.emby_sync import SyncError
        from app.emby_sync import sync as emby_sync

        api_key = args.tmdb_key or os.environ.get("TMDB_API_KEY", "")
        if not api_key:
            print("error: TMDB API key required (--tmdb-key or TMDB_API_KEY)", file=sys.stderr)
            return 1
        try:
            emby_sync(args.db, args.data_dir, api_key)
        except SyncError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        return 0

    conn = connect(args.db)
    try:
        if args.command == "seed":
            seed(conn, load_fixture(args.fixture))
        else:
            params = RankParams(
                similarity=args.similarity, mmr_lambda=args.mmr_lambda, top_n=args.top
            )
            for rec in recommend(conn, args.user, params):
                print(f"{rec.title} ({rec.year}) {rec.score:.4f} | {rec.reason}")
    except (SeedError, RecommendError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
