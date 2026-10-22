"""Local accounts: scrypt password hashes, login rate limiting and server-side sessions."""

import base64
import hashlib
import hmac
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

Role = Literal["user", "admin"]

SESSION_LIFETIME = timedelta(days=7)
MAX_FAILED_LOGINS = 5
FAILED_LOGIN_WINDOW = timedelta(minutes=15)

# scrypt cost parameters; they are stored in each hash so they can be raised later.
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_KEY_LENGTH = 64


@dataclass(frozen=True)
class User:
    id: int
    name: str
    role: Role


@dataclass(frozen=True)
class Session:
    token: str
    expires_at: datetime


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    key = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64(salt)}${_b64(key)}"


def verify_password(password: str, stored: str) -> bool:
    _, n, r, p, salt, key = stored.split("$")
    candidate = _scrypt(password, base64.b64decode(salt), int(n), int(r), int(p))
    return hmac.compare_digest(candidate, base64.b64decode(key))


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        password.encode(), salt=salt, n=n, r=r, p=p, dklen=SCRYPT_KEY_LENGTH, maxmem=2**26
    )


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# Checked against when the username is unknown, so a miss costs as much as a wrong password
# and response time does not reveal which accounts exist.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def set_password(conn: sqlite3.Connection, name: str, password: str, role: Role) -> None:
    """Create the account, or give an existing (e.g. synced) user a password and role.

    Ends the user's existing sessions so a reset password locks out whoever held the old one.
    """
    with conn:
        conn.execute(
            "INSERT INTO users (name, password_hash, role) VALUES (?, ?, ?)"
            " ON CONFLICT (name) DO UPDATE SET"
            " password_hash = excluded.password_hash, role = excluded.role",
            (name, hash_password(password), role),
        )
        conn.execute(
            "DELETE FROM sessions WHERE user_id = (SELECT id FROM users WHERE name = ?)", (name,)
        )


def login_rate_limited(conn: sqlite3.Connection, username: str, now: datetime) -> bool:
    (failures,) = conn.execute(
        "SELECT COUNT(*) FROM login_failures WHERE username = ? AND failed_at > ?",
        (username, (now - FAILED_LOGIN_WINDOW).isoformat()),
    ).fetchone()
    return failures >= MAX_FAILED_LOGINS


def authenticate(
    conn: sqlite3.Connection, username: str, password: str, now: datetime
) -> User | None:
    """Return the user for valid credentials; otherwise record a failed attempt."""
    row = conn.execute(
        "SELECT id, name, role, password_hash FROM users WHERE name = ?", (username,)
    ).fetchone()
    stored = row[3] if row else None
    password_ok = verify_password(password, stored or _DUMMY_HASH)
    if stored and password_ok:
        with conn:
            conn.execute("DELETE FROM login_failures WHERE username = ?", (username,))
        return User(id=row[0], name=row[1], role=row[2])
    with conn:
        conn.execute(
            "INSERT INTO login_failures (username, failed_at) VALUES (?, ?)",
            (username, now.isoformat()),
        )
    return None


def create_session(conn: sqlite3.Connection, user: User, now: datetime) -> Session:
    token = secrets.token_urlsafe(32)
    expires_at = now + SESSION_LIFETIME
    with conn:
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (_token_hash(token), user.id, expires_at.isoformat()),
        )
    return Session(token=token, expires_at=expires_at)


def session_user(conn: sqlite3.Connection, token: str, now: datetime) -> User | None:
    row = conn.execute(
        "SELECT users.id, users.name, users.role, sessions.expires_at"
        " FROM sessions JOIN users ON users.id = sessions.user_id"
        " WHERE sessions.token_hash = ?",
        (_token_hash(token),),
    ).fetchone()
    if row is None:
        return None
    if datetime.fromisoformat(row[3]) <= now:
        end_session(conn, token)
        return None
    return User(id=row[0], name=row[1], role=row[2])


def end_session(conn: sqlite3.Connection, token: str) -> None:
    with conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


def _token_hash(token: str) -> str:
    """Sessions are stored by token digest so a leaked database holds no usable cookies."""
    return hashlib.sha256(token.encode()).hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
