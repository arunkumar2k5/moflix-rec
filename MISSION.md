# Mission

**Derived from:** PRD.md (Weekend Picker v0.1)
**Last reconciled with it:** 2026-10-20

## What moflix-rec is

moflix-rec is a self-hosted chat-based movie recommender for a small, closed group of existing media-server users. It removes "what do we watch tonight?" friction by computing personalised top-N picks the moment a user logs in, using a per-user taste vector scored against movie metadata vectors by dot product. The user can then refine the picks through plain-English chat; the system parses intent via an external LLM API, applies deterministic vector deltas, and responds with a refreshed set — all within a single chat turn.

It runs as a standalone service on one Linux machine alongside the Emby media server. Cloudflare sits in front. No local GPU inference runs anywhere; all AI is external API calls.

## Who it is for

- **Media-server friends and family:** non-technical, mostly mobile, want a pick in under 10 seconds without scrolling the full library.
- **Admin (Arun):** manages users, triggers syncs, monitors API cost and logs.

moflix-rec is not a streaming player, a social platform, or a public discovery service.

## Core capabilities (in scope)

**Recommendations**
- Per-user top-N picks scored by dot product of taste vector and movie vectors
- Diversity pass (MMR) so the top results are not near-duplicates
- Short plain-English explanation with every recommendation
- Cold-start flow (popularity + onboarding chat) for users with sparse history

**Chat tuning**
- Natural-language refinement ("less violent", "like Parasite but funnier", "nothing over 2 hours")
- LLM parses message → strict JSON intent → deterministic vector delta → immediate re-score
- Session vector resets with a single button
- Graceful fallback to keyword/genre parser when the LLM API is unavailable

**Data and sync**
- Nightly + on-demand sync from the Emby media server (library, per-user watch history)
- Metadata enrichment from Emby first, TMDB API as fallback
- Structured movie vectors built at sync time; scoring never calls an external API
- Taste vector: weighted average of watched-movie vectors, adjusted by completion %, recency, rewatches, favourites, negative signals

**Feedback**
- Watch tonight / Not interested / Already seen actions persisted and fed back into the taste vector
- Full recommendation and action log for weight tuning

**Admin**
- User list, last sync time, manual sync trigger, API cost counters, error log
- Configurable parameters: blend weight α, recency decay, top-N count

## Out of scope — the factory must never build this

**Local AI / GPU**
- Running, fine-tuning, or hosting any LLM or embedding model on this machine
- Any GPU inference, quantised model serving, or llama.cpp integration

**Media playback**
- Video streaming, transcoding, or playback controls of any kind
- Deep-linking into Emby beyond poster/metadata fetches

**Public access**
- Open registration, invite links, or any path for a non-admin to create an account
- OAuth with a third-party provider (Google, GitHub, etc.)

**Social features**
- Shared watchlists, friend ratings feeds, comments, or collaborative filtering visible to other users
- Group "what should we all watch?" mode (revisit post-v1)

**External LLM ranking**
- Letting the LLM score, rank, or directly select movies — it parses intent only
- Sending watch history or taste vectors to any external API

**Data export / portability**
- Public data-download endpoints or API access for end users
- Integration with other recommendation platforms (Trakt, Letterboxd, etc.)

**Per-user UI customisation**
- Themes, layout preferences, custom lists beyond the built-in feedback actions

## Hard invariants — not tunable by any issue

1. **User data is strictly isolated.** One user's watch history, taste vector, and chat session are never readable by another user, including through timing channels or shared caches. Any issue that would share data across users is an automatic reject.
2. **The LLM never touches vectors or rankings directly.** The LLM produces a structured intent JSON. The backend applies bounded, deterministic deltas. The LLM never receives taste vectors or other users' data.
3. **Scoring is fully local.** The dot-product scoring pass never makes a network call. Only chat messages and minimal movie metadata (titles, genres) may leave the server.
4. **No playback.** This service never streams, transcodes, or proxies media files.
5. **The factory cannot modify governance files.** `MISSION.md`, `FACTORY.md`, `FACTORY_RULES.md`, and `engineering.md` are the constitution. A PR touching any of them is an automatic reject.
6. **The factory cannot modify its own judge.** `harness/` and `.factory/holdout/` define what "working" means. Adding assertions is always welcome; removing or loosening one is a human decision.

## Allowed evolutions

- Swapping SQLite for PostgreSQL if the dataset or user count grows — as long as the data model and API contracts are unchanged.
- Adding a text-embedding dimension to movie vectors (via an optional embedding API call at sync time) — structured-only is the default; embedding is an opt-in config flag.
- Replacing or adding an AI provider in the abstraction layer without changing the intent schema.
- Improving the cold-start or diversity algorithm.

## Definition of done

Every change the factory ships clears all three gates.

**Gate 1 — static checks and tests pass.**
```
pytest tests/ -x -q
```

**Gate 2 — no cross-user data leakage, scoring is deterministic, and the LLM path is bounded.**
Any new user-facing feature must be usable on mobile without documentation.

**Gate 3 — the end-to-end path passes as a real user.**

1. Start the app: `docker compose up -d` (or `uvicorn app.main:app --reload` for dev).
2. Log in as a test user with watch history.
3. The recommendation page loads within 2 seconds showing at least 2 movies with explanations.
4. Send a chat message: "something from the 90s".
5. The picks update within 4 seconds and at least one result is from the 1990s.
6. Mark one movie "Watch tonight"; reload; the mark persists.
7. Click "Reset tuning"; the picks revert to the pre-chat set.

This runs on every change that touches runnable code.

## Open questions — decisions nobody has made yet

These are undecided, not forbidden. The factory may propose an answer, build against it, and record what it assumed — the merge is held for a human.

- **Q1** Login method: reuse Emby credentials via its API, or a separate local account table mapped to Emby user IDs?
- **Q2** Text embeddings in v1: ship with structured features only (faster, no embedding cost) or include an optional embedding dimension from the start?
- **Q3** Which AI provider and model for intent parsing, and what is the monthly spend cap?
- **Q4** Does "Watch tonight" need to persist across sessions as a saved list, or is it session-scoped?

**These stop the factory** — they are irreversible:

- Any change to the user identity or authentication model (who may act as whom).
- Any schema migration that drops or transforms stored watch history or taste vectors.

## Decisions

- **2026-10-21** **Text embeddings in v1 (Q2)?** Yes, using `all-MiniLM-L6-v2` (sentence-transformers, local, CPU-only, no API cost). Movie vectors are 490-dim: 106 structured + 384 embedding. Embedding runs at sync time only; scoring is still fully local.
- **2026-10-20** **Group mode in v1?** No. Deferred to post-v1.
- **2026-10-20** **Local LLM?** Never. External API only, provider-agnostic abstraction layer.

## What the factory does NOT own — permanently human

- Does it FEEL right — pacing of the chat, tone of the movie blurbs, whether the "reason" line is useful or annoying.
- Does it LOOK right — poster layout, card hierarchy, whether "Watch tonight" reads as distinct from "Not interested" on a small screen.
- Is it UNDERSTANDABLE — can a non-technical family member work it out without being told what a "taste vector" is.
- Is the cold-start onboarding friendly — the factory checks it runs; a human checks it feels welcoming.

The factory owns the data model, the vector math, the scoring pipeline, the intent-parsing contract, and the sync logic: the layer whose correctness can be asserted. The list above is reviewed by a human, forever.
