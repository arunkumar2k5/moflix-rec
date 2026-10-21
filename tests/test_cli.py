import re
import subprocess
import sys
from pathlib import Path

from tests.conftest import FIXTURE_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent
LINE = re.compile(r"^.+ \(\d{4}\) -?\d+\.\d{4}$")


def run_cli(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "app.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_seed_then_recommend_prints_top_n_lines(tmp_path):
    db = str(tmp_path / "cli.db")
    # seed embeds 24 movies with the local model; allow extra time on first run.
    seeded = run_cli("seed", "--db", db, "--fixture", str(FIXTURE_PATH), timeout=300)
    assert seeded.returncode == 0, seeded.stderr

    result = run_cli("recommend", "--db", db, "--user", "alice", "--top", "3")

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 3
    assert all(LINE.match(line) for line in lines)


def test_recommend_for_unknown_user_exits_nonzero(tmp_path):
    db = str(tmp_path / "cli.db")
    run_cli("seed", "--db", db, "--fixture", str(FIXTURE_PATH), timeout=300)

    result = run_cli("recommend", "--db", db, "--user", "mallory", "--top", "3")

    assert result.returncode == 1
    assert "unknown user" in result.stderr


def test_recommend_against_missing_db_reports_it_and_creates_nothing(tmp_path):
    db = tmp_path / "missing.db"

    result = run_cli("recommend", "--db", str(db), "--user", "alice", "--top", "3")

    assert result.returncode == 1
    assert "database not found" in result.stderr
    assert not db.exists()
