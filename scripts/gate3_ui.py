"""MISSION Gate 3 steps 2-7 through the phone page, in headless Chromium at 390x844.

    pip install -e '.[e2e]' && playwright install --with-deps chromium
    python3 scripts/gate3_ui.py --log gate3-ui.log

Seeds a fresh database from tests/fixtures/library.json, logs in as alice (who has watch
history) and serves it with uvicorn. Every *_API_KEY variable is dropped from the server's
environment, so chat runs in simple mode: deterministic and free, through the same UI path.
Writes one `STEP n PASS|FAIL` line per step and exits 1 on the first failure.
"""

import argparse
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.auth import set_password  # noqa: E402
from app.db import connect  # noqa: E402
from app.seed import load_fixture, seed  # noqa: E402

USER = "alice"
CHAT_MESSAGE = "something from the 90s"
PAGE_BUDGET_MS = 2000
CHAT_BUDGET_MS = 4000
STARTUP_TIMEOUT_S = 60


class StepFailed(Exception):
    pass


def seed_database(path: Path, password: str) -> None:
    conn = connect(path)
    try:
        seed(conn, load_fixture(ROOT / "tests" / "fixtures" / "library.json"))
        set_password(conn, USER, password, "user")
    finally:
        conn.close()


def start_server(db_path: Path, port: int, output: TextIO) -> subprocess.Popen:
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env["MOFLIX_DB"] = str(db_path)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-config",
            "app/log_config.json",
            # The app has no WebSocket routes; skipping them avoids importing `websockets`.
            "--ws",
            "none",
        ],
        cwd=ROOT,
        env=env,
        stdout=output,
        stderr=subprocess.STDOUT,
    )


def wait_healthy(base_url: str, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise StepFailed(f"server exited with code {server.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.5)
    raise StepFailed(f"server not healthy after {STARTUP_TIMEOUT_S}s")


def card_titles(page: Page) -> list[str]:
    return page.locator("#picks .card-title").all_inner_texts()


def wait_for_explained_cards(page: Page) -> None:
    page.wait_for_function(
        """() => [...document.querySelectorAll('#picks .card')]
            .filter(card => card.querySelector('.card-reason').textContent.trim()).length >= 2"""
    )


def run_steps(page: Page, password: str, log: Callable[[str], None]) -> None:
    baseline: list[str] = []
    chat_started = 0.0

    def step(number: int, run: Callable[[], str]) -> None:
        started = time.perf_counter()
        try:
            detail = run()
        except Exception as error:
            log(f"STEP {number} FAIL {error} ({elapsed_ms(started)} ms)")
            raise StepFailed(f"step {number}") from error
        log(f"STEP {number} PASS {detail} ({elapsed_ms(started)} ms)")

    def log_in() -> str:
        page.goto("/")
        page.fill("#username", USER)
        page.fill("#password", password)
        with page.expect_navigation():
            page.click("#login-form button[type=submit]")
        if page.locator("#login-form").count():
            raise StepFailed(f"still on the login form: {page.inner_text('#login-error')!r}")
        return f"logged in as {USER}"

    def picks_load() -> str:
        started = time.perf_counter()
        page.goto("/")
        wait_for_explained_cards(page)
        took = elapsed_ms(started)
        if took > PAGE_BUDGET_MS:
            raise StepFailed(f"page took {took} ms, budget {PAGE_BUDGET_MS} ms")
        baseline.extend(card_titles(page))
        return f"{len(baseline)} picks with reasons in {took} ms: {baseline}"

    def send_chat() -> str:
        nonlocal chat_started
        page.fill("#chat-input", CHAT_MESSAGE)
        chat_started = time.perf_counter()
        page.click("#chat-form button[type=submit]")
        return f"sent {CHAT_MESSAGE!r}"

    def picks_update() -> str:
        page.wait_for_function("() => document.getElementById('chat-reply').textContent.trim()")
        took = elapsed_ms(chat_started)
        if took > CHAT_BUDGET_MS:
            raise StepFailed(f"picks took {took} ms, budget {CHAT_BUDGET_MS} ms")
        error = page.inner_text("#chat-error")
        if error:
            raise StepFailed(f"chat error shown: {error!r}")
        years = [int(y) for y in page.locator("#picks .card-year").all_inner_texts()]
        if not any(1990 <= year <= 1999 for year in years):
            raise StepFailed(f"no pick from the 1990s: {years}")
        understood = page.inner_text("#understood")
        if not understood.startswith("I understood: ") or re.search(r"[{}\[\]]", understood):
            raise StepFailed(f"echo is not plain prose: {understood!r}")
        notice = "shown" if page.is_visible("#simple-notice") else "hidden"
        return (
            f"picks updated in {took} ms, years {years}; {understood!r};"
            f" reply {page.inner_text('#chat-reply')!r}; simple-mode notice {notice}"
        )

    def watch_tonight_persists() -> str:
        card = page.locator("#picks .card").first
        title = card.locator(".card-title").inner_text()
        with page.expect_response(lambda r: r.url.endswith("/feedback")) as response:
            card.locator("[data-action=watch_tonight]").click()
        if response.value.status != 204:
            raise StepFailed(f"POST /feedback returned {response.value.status}")
        page.reload()
        saved = page.locator("#watch-tonight li").all_inner_texts()
        if title not in saved:
            raise StepFailed(f"{title!r} not in the saved list after reload: {saved}")
        return f"{title!r} still saved for tonight after reload"

    def reset_reverts() -> str:
        with page.expect_navigation():
            page.click("#reset-tuning")
        wait_for_explained_cards(page)
        titles = card_titles(page)
        if titles != baseline:
            raise StepFailed(f"picks {titles} differ from pre-chat {baseline}")
        return f"picks back to the pre-chat set {titles}"

    step(2, log_in)
    step(3, picks_load)
    step(4, send_chat)
    step(5, picks_update)
    step(6, watch_tonight_persists)
    step(7, reset_reverts)


def elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, default=Path("gate3-ui.log"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    # Chromium treats localhost, unlike 127.0.0.1, as a secure context, so it keeps the
    # Secure session and __Host-csrf cookies over plain http.
    base_url = f"http://localhost:{args.port}"
    password = secrets.token_urlsafe(16)

    with (
        tempfile.TemporaryDirectory() as tmp,
        args.log.open("w", encoding="utf-8") as log_file,
    ):

        def log(line: str) -> None:
            print(line)
            log_file.write(line + "\n")
            log_file.flush()

        log("Gate 3 steps 2-7, headless Chromium, viewport 390x844, simple mode (no API keys)")
        db_path = Path(tmp) / "gate3.db"
        seed_database(db_path, password)
        server_log = Path(tmp) / "server.log"
        with server_log.open("w", encoding="utf-8") as server_output:
            server = start_server(db_path, args.port, server_output)
        try:
            wait_healthy(base_url, server)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                context = browser.new_context(
                    base_url=base_url, viewport={"width": 390, "height": 844}
                )
                run_steps(context.new_page(), password, log)
                browser.close()
        except StepFailed as error:
            log(f"FAILED: {error}")
            log("--- server output (tail) ---")
            for line in server_log.read_text(encoding="utf-8").splitlines()[-40:]:
                log(line)
            return 1
        finally:
            server.terminate()
            server.wait(timeout=10)
        log("Gate 3 PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
