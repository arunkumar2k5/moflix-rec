import re
import subprocess
import sys
from pathlib import Path

from app.auth import verify_password
from app.db import connect
from tests.conftest import FIXTURE_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent
LINE = re.compile(r"^.+ \(\d{4}\) -?\d+\.\d{4} \| .{10,}$")


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

    tuned = run_cli(
        "recommend",
        "--db",
        db,
        "--user",
        "alice",
        "--top",
        "3",
        "--similarity",
        "cosine",
        "--mmr-lambda",
        "1.0",
    )

    assert tuned.returncode == 0, tuned.stderr
    tuned_lines = tuned.stdout.splitlines()
    assert len(tuned_lines) == 3
    assert all(LINE.match(line) for line in tuned_lines)


def test_recommend_rejects_mmr_lambda_outside_unit_interval(tmp_path):
    result = run_cli(
        "recommend", "--db", str(tmp_path / "x.db"), "--user", "a", "--mmr-lambda", "2"
    )

    assert result.returncode == 2
    assert "must be between 0 and 1" in result.stderr


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


def test_add_user_gives_an_existing_user_a_password(tmp_path):
    db = tmp_path / "cli.db"
    conn = connect(db)
    with conn:
        synced_id = conn.execute("INSERT INTO users (name) VALUES ('alice')").lastrowid
    conn.close()

    result = subprocess.run(
        [sys.executable, "-m", "app.cli", "add-user", "--db", str(db), "--user", "alice"],
        cwd=REPO_ROOT,
        input="s3cret-pass\n",
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "s3cret-pass" not in result.stdout + result.stderr
    conn = connect(db)
    row = conn.execute("SELECT id, role, password_hash FROM users WHERE name = 'alice'").fetchone()
    conn.close()
    assert row[:2] == (synced_id, "user")
    assert verify_password("s3cret-pass", row[2])


def test_add_user_rejects_an_empty_password(tmp_path):
    db = tmp_path / "cli.db"

    result = subprocess.run(
        [sys.executable, "-m", "app.cli", "add-user", "--db", str(db), "--user", "bob"],
        cwd=REPO_ROOT,
        input="\n",
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1
    assert "must not be empty" in result.stderr
    assert not db.exists()
