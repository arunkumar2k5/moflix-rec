# PRD: Chat-Based Movie Recommender (working title: "Weekend Picker")

**Version:** 0.1 | **Owner:** Arun | **Status:** Draft | **Date:** 20 Oct 2026

## 1. Overview
A self-hosted web app that recommends movies to each user from their watch history and lets them refine the picks by chatting. Recommendations come from a per-user taste vector scored against movie metadata vectors by dot product. Chat feedback updates the taste vector in real time.

It runs entirely on my own Linux server as an independent application. Cloudflare handles the domain and static IP mapping, and I'll configure that separately. No local LLM is used. Any AI comes from external APIs.

## 2. Goals and Non-Goals
**Goals**
- Remove "what do we watch tonight?" decision fatigue for my existing users.
- Personalised top-N recommendations immediately after login.
- Chat-based tuning, e.g. "less violent", "something older", "like movie X but funnier".
- Fully self-hosted, testable locally first, then launched on the domain.

**Non-goals (v1)**
- No running or hosting of local LLMs or GPU inference.
- No video streaming or playback (the media server handles that).
- No public sign-ups. Users are the existing media-server users.
- No social features (shared lists, ratings feed).

## 3. Users
- **End user:** an existing media-server friend or family member. Non-technical, mostly on mobile, wants picks in under 10 seconds.
- **Admin (me):** manages users, triggers data sync, monitors API cost and logs.

## 4. User Flow
1. The user opens the site and logs in.
2. The server loads their taste vector and computes the top-N picks.
3. The chat window opens with the assistant presenting 2–3 movies (poster, year, genres, one-line reason).
4. The user chats ("too slow-paced", "I want something with a twist", "not another Marvel").
5. The system interprets the message, updates the session taste vector, re-scores, and replies with a refreshed set.
6. The user can mark Watch tonight / Not interested / Already seen. These signals persist.

## 5. Functional Requirements

### 5.1 Authentication
- FR-1: Login using the existing media-server user identity (via its API) or a local user table mapped to media-server user IDs. Decision needed, see Open Questions.
- FR-2: Session via secure HTTP-only cookie, with expiry and logout.
- FR-3: Users can only see their own data. Admin role is separate.

### 5.2 Data Ingestion
- FR-4: Nightly and on-demand sync from the media server: library items (movies), per-user watch history (played, play count, last played, completion %, favourite flag).
- FR-5: Metadata enrichment per movie: title, year, genres, runtime, language, country, certification, overview, cast, director, keywords, community rating. Source: media-server metadata first, with an optional TMDB API fallback for gaps.
- FR-6: Store everything in a local database. Sync is idempotent and logged.

### 5.3 Movie Vector
- FR-7: Each movie gets a fixed-length vector built from:
  - Multi-hot genres
  - Decade/era bucket
  - Runtime bucket
  - Language and country
  - Certification
  - Normalised rating/popularity
  - Top director and cast (hashed or one-hot over frequent entities)
  - Optional text embedding of overview + keywords (via embedding API, reduced to a small dimension)
- FR-8: Vectors are computed at sync time, versioned, and cached. Scoring never calls an API.

### 5.4 Taste Vector
- FR-9: Each user's base taste vector = weighted average of the vectors of movies they watched. Weights: completion %, rewatch count, favourite flag, and recency decay.
- FR-10: Negative signals (abandoned early, "not interested") subtract from the vector.
- FR-11: Two layers are kept: a **base vector** (long-term, from history) and a **session vector** (short-term, from chat). Final score uses a blend: `score = (α·base + (1−α)·session) · movie_vector`.
- FR-12: Option to persist stable chat-derived preferences into the base vector with a small learning rate.

### 5.5 Recommendation Engine
- FR-13: Score all unwatched movies by dot product (cosine as a configurable alternative, since raw dot product favours high-magnitude vectors).
- FR-14: Hard filters: exclude already-watched, "not interested", and anything violating chat constraints (runtime, language, rating).
- FR-15: Diversity pass (e.g. MMR) so the top 3 are not near-duplicates.
- FR-16: Cold start: with fewer than N watched movies, fall back to popularity plus a short onboarding chat ("pick genres you like").
- FR-17: Every recommendation carries a short explanation (e.g. "matches your love of thrillers, similar to *X*").

### 5.6 Chat Interface and Tuning
- FR-18: Chat UI per the supplied PNG design.
- FR-19: Each user message goes to an external LLM API with a strict JSON output schema, returning intents such as:
  - `genre_adjust` (+/− weights)
  - `era`, `runtime`, `language`, `mood/pace` adjustments
  - `similar_to` / `unlike` (referencing a movie)
  - `hard_filters` (exclude or require)
  - `reaction` to a specific recommended movie
- FR-20: The backend maps that JSON to deterministic vector deltas. The LLM never edits vectors directly and never ranks. It only parses intent and writes friendly replies.
- FR-21: Re-score after each message and respond within the same chat turn.
- FR-22: The LLM sees only the user's message, the last few turns, and the current picks (titles/metadata), never other users' data.
- FR-23: A "Reset tuning" button clears the session vector.
- FR-24: Graceful fallback if the API fails: basic keyword/genre parser plus a "AI is unavailable, using simple mode" notice.

### 5.7 Feedback Loop
- FR-25: Log every recommendation shown and every action (click, watch-tonight, dismiss, chat-adjust) for later tuning of weights.

### 5.8 Admin
- FR-26: Admin page: user list, last sync time, manual sync, API usage and cost counters, error log, configurable parameters (α, decay, N).

## 6. Non-Functional Requirements
- **Performance:** Scoring under 200 ms for a library of up to ~20k movies. Chat turn under 4 s including the LLM call.
- **Deployment:** Single Linux host, Docker Compose preferred (app, DB, reverse proxy). Runs on a modest CPU-only machine.
- **Security:** HTTPS via Cloudflare (Tunnel or proxied A record, with origin locked down). Secrets in env/secret files. Rate-limit login and chat endpoints. CSRF protection. No API keys in the frontend.
- **Privacy:** Watch history stays on my server. Only the minimal chat text and movie metadata leave to the AI API. Provide per-user data deletion.
- **Cost control:** Per-user daily chat cap, a small and cheap model for intent parsing, response caching, a monthly budget alarm.
- **Observability:** Structured logs, health endpoint, simple metrics.
- **Backups:** Nightly DB backup.
- **Responsive:** Mobile-first layout.

## 7. Proposed Architecture
- **Frontend:** Lightweight SPA (React or plain HTMX/JS), chat-style UI from the PNG. Poster images from the media server or TMDB.
- **Backend:** Python + FastAPI (fits my stack).
- **Math:** NumPy for vectors and scoring.
- **DB:** SQLite for v1 (PostgreSQL if users or data grow). Vectors stored as blobs and held in memory at runtime.
- **Jobs:** Scheduler (APScheduler or cron) for sync and vector builds.
- **AI:** One provider abstraction layer so I can swap APIs (Anthropic, OpenAI, Groq, etc.) via config.
- **Edge:** Nginx or Caddy, with Cloudflare in front.

**Data model (core tables):** `users`, `movies`, `movie_vectors`, `watch_events`, `taste_vectors` (base), `chat_sessions`, `chat_messages`, `recommendation_log`, `feedback_events`, `sync_runs`.

**Request flow:** Login → load base vector → score → UI shows the picks → user message → LLM API (intent JSON) → vector delta → re-score → reply.

## 8. Success Metrics
- ≥60% of sessions end with a Watch tonight click.
- Median time to a chosen movie under 2 minutes.
- ≥30% of users use chat tuning at least once.
- API cost under a set monthly cap (decide the figure).
- Zero cross-user data leakage in tests.

## 9. Milestones
1. **M0:** Access to the media-server API, data export, a test user set.
2. **M1:** Sync and DB, movie vectors.
3. **M2:** Taste vector and scoring engine, tested offline (CLI).
4. **M3:** Backend API plus the UI from the PNG, without chat tuning.
5. **M4:** Chat tuning with the LLM intent parser.
6. **M5:** Local testing with 2–3 real users, parameter tuning.
7. **M6:** Hardening (auth, rate limits, backups), Cloudflare mapping, launch.

## 10. Testing
- Unit tests for vector building and scoring (known fixtures).
- Replay test: hide the last 20% of a user's history and check hit rate.
- Chat intent tests: a table of phrases mapped to expected JSON.
- Load and security checks before launch.

## 11. Risks
- **Sparse history:** weak taste vectors, mitigated by cold-start flow and popularity blend.
- **LLM misreads intent:** strict schema, bounded deltas, and a visible "I understood: …" echo.
- **Media-server API limits or schema changes:** abstract the source behind an adapter.
- **API cost or outage:** caps and a fallback mode.
- **Taste drift from over-tuning chat:** a small learning rate and a reset option.

## 12. Open Questions
1. Login method: reuse media-server credentials, or a separate account table?
2. Who runs the media server and does it expose a user-history API I can read? (My notes say a friend runs it.)
3. Is a text embedding (API) worth adding for v1, or ship with structured features only and add it later? I'd suggest structured first.
4. Which AI provider and model for intent parsing, and the monthly budget?
5. Do users need a Watch tonight list persisted across sessions?
6. Group mode ("what should we all watch?") in v1 or later? I'd suggest later.
