# ADR-002: UI Design Decisions from Wireframe

**Date:** 2026-10-22
**Status:** Active — governs issue #12 and backend API shape
**Source files:** `data/GUI/index.html`, `data/GUI/moflix-home-requirements.md`

---

## What the wireframe establishes

The UI is a **TV zoom intro** (SVG room scene, couple on sofa, TV on the wall) that zooms
into the TV screen to reveal the home page. The home page has:
- A sticky header with logo, nav, and a login widget
- Two personalised content rails (horizontal scroll-snap)
- Placeholder "Movie night" and "Start a series" rows for later

The full `index.html` is the base file. All changes extend it — no framework, no bundler,
plain HTML/CSS/JS.

---

## Decision 1 — Authentication model: login form embedded in the TV screen

**Decision:** Password-based login. The login form lives **inside the TV screen** on the
landing page — top-right of the TV while the room intro plays.

**How it works:**
- Username = exact DB username (e.g. "ak261", "Vimalnath")
- All accounts start with the default password **`Test123`**
- After first login the user can change their password (stored hashed in the DB)
- Session via a secure HTTP-only cookie with expiry and logout
- The home page only renders after a valid session exists
- `GET /api/users` returns `[{ "id": "u1", "name": "Arun" }]` (names only, no hashes)
  so the login form can populate a username dropdown

**Login UI placement:**
- The TV screen shows a compact login widget in the top-right corner while the intro plays
- Username dropdown (populated from `/api/users`) + password field + Login button
- On successful login the TV zooms into the home page as normal
- On a returning valid session cookie the intro skips straight to the home page

**Password storage:**
- `users` table gets a `password_hash TEXT` column (bcrypt)
- Default hash for `Test123` written at sync time for any user with no password set yet
- Endpoints:
  - `POST /api/auth/login` — validates credentials, sets HTTP-only session cookie
  - `POST /api/auth/logout` — clears session cookie
  - `POST /api/auth/change-password` — validates current password, stores new bcrypt hash
  - `GET /api/auth/me` — returns current user from session (used on page load to restore state)

**Hard invariant:** session cookie validated on every request; cross-user access → 403.

---

## Decision 2 — API contracts (what the backend must return)

### `GET /api/users`
```json
[ { "id": "u1", "name": "Arun" } ]
```
- Names only — no password hashes ever leave the backend
- Used to populate the login username dropdown
- Dev fallback: `data/users.json`

### `GET /api/new-releases`
```json
[
  {
    "id": "m101", "title": "Example Title", "year": 2026, "type": "movie",
    "language": "Tamil", "runtime_min": 138, "genres": ["Drama"],
    "poster_url": "https://image.tmdb.org/t/p/w500/abc.jpg",
    "overview": "...", "added_on": "2026-10-20"
  }
]
```
- Content is **curated** — admin adds entries; backend auto-fetches TMDB metadata by title
- Dev fallback: `data/new_releases.json`
- Empty list → section hidden entirely

### `GET /api/picks?user_id=<id>&limit=12`
```json
[
  {
    "id": "m42", "title": "Kara", "year": 2026, "type": "movie",
    "language": "Tamil", "runtime_min": 142, "genres": ["Action", "Thriller"],
    "poster_url": "https://image.tmdb.org/t/p/w500/xyz.jpg",
    "overview": "...", "score": 0.87,
    "reason": "Matches your taste for Action, similar to Vada Chennai"
  }
]
```
- `score` is **cosine similarity in [0, 1]** — UI shows `round(score * 100)%`
- `poster_url` now stored in movies table (721/736 backfilled from TMDB)
- Dev fallback: `data/picks_<userId>.json`

### `GET /api/auth/me`
```json
{ "id": "u1", "name": "Arun" }
```
Returns 401 if no valid session. Used on page load.

---

## Decision 3 — Score normalisation: cosine, not dot product

`RankParams(similarity="cosine")` must be the API default. Raw dot products (3–5 range)
would show nonsensical match percentages.

---

## Decision 4 — Poster images

- `poster_url` stored in `movies` table as `https://image.tmdb.org/t/p/w500{poster_path}`
- 721/736 movies backfilled as of 2026-10-22 (commit `ac3d489`)
- Re-sync automatically backfills any movies with missing poster URLs
- For new releases: admin POSTs a title → backend fetches TMDB → stores poster + metadata
- Frontend CSS gradient placeholder shown if `poster_url` is null or image 404s

---

## Decision 5 — TV zoom: portrait vs landscape behaviour

| Mode | Trigger | Zoom end | Transition |
|---|---|---|---|
| Landscape | aspect ratio ≥ 1.6 | TV covers full viewport | Crossfade (current behaviour) |
| Portrait/tablet | aspect ratio < 1.6 | TV width = viewport width | Clip-expand ~500ms to full mobile layout |

Portrait intro shows simplified content (logo + "Friday night, sorted." only).
On portrait phones the TV is cropped tighter in the opening shot.

---

## New endpoints summary

| Endpoint | Issue | Status |
|---|---|---|
| `POST /api/auth/login` | #5 | PR #19 open |
| `POST /api/auth/logout` | #5 | PR #19 open |
| `POST /api/auth/change-password` | #5 | PR #19 open |
| `GET /api/auth/me` | #5 | PR #19 open |
| `GET /api/users` | #5 | PR #19 open |
| `GET /api/picks` | #7 | Re-run after #5 merges |
| `GET /api/health` | #8 | Re-run after #5+#6 merge |
| `GET /api/new-releases` | Manual/small patch | Needed before #12 |

---

## What issue #12 (web-ui) will build

Starting from `data/GUI/index.html`:
1. Login widget in TV screen top-right (username dropdown + password + Login button)
2. On valid session → TV zooms to home page
3. Two content rails with real API data
4. Password change modal after first login
5. Mobile + portrait/landscape zoom modes
6. Reduced-motion support, error/retry states, dev JSON fallbacks

**Do not start #12 until:** #5 and #7 are merged, poster_url is confirmed populated.

---

## Files to keep out of git

`data/GUI/` — wireframe source files (gitignored).
