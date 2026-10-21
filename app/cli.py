"""Command line entry point: `python -m app.cli seed|recommend`."""

import argparse
import sys
from pathlib import Path

from app.db import connect
from app.recommend import RecommendError, recommend
from app.seed import SeedError, load_fixture, seed


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "recommend" and not Path(args.db).is_file():
        print(f"error: database not found: {args.db}", file=sys.stderr)
        return 1
    conn = connect(args.db)
    try:
        if args.command == "seed":
            seed(conn, load_fixture(args.fixture))
        else:
            for rec in recommend(conn, args.user, args.top):
                print(f"{rec.title} ({rec.year}) {rec.score:.4f}")
    except (SeedError, RecommendError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
