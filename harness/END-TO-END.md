# Runtime scenarios

Journeys for the moflix-rec chat-based movie recommender.
Since this repo starts from a PRD (no running code yet), these journeys describe the
declared behaviour the first skeleton must satisfy before anything can merge.
The stack is FastAPI + SQLite, started with `docker compose up -d` or
`uvicorn app.main:app --reload` for local dev.

---

## 1. A user logs in and sees personalised picks immediately

1. Start the app with at least one seeded test user who has ≥5 watch events.
2. POST `/auth/login` with that user's credentials.
3. GET `/recommendations` for that user.
4. The response returns at least 2 movies, each with: title, year, genres, and a non-empty `reason` string.
5. All returned movies are ones the user has NOT watched (not in their watch history).
6. Response time is under 2 seconds from a cold scoring pass.

**What would make this fail:** a watched movie appears in results; the `reason` field is missing or empty; the response takes more than 2 seconds; the endpoint is reachable without a valid session cookie.

---

## 2. Chat tuning updates the picks in the same turn

1. Log in as a test user who has watched a mix of genres (e.g. action and drama).
2. GET `/recommendations` — note the current top picks.
3. POST `/chat` with the message `"something from the 1990s"`.
4. The response includes an updated recommendation list.
5. At least one returned movie has a release year between 1990 and 1999.
6. The response is delivered within 4 seconds (including the LLM API call).
7. The assistant reply contains a short natural-language acknowledgement of the constraint.

**What would make this fail:** picks do not change; no 1990s movie appears; response exceeds 4 s; the reply contains raw JSON visible to the user.

---

## 3. Feedback signals persist and affect future recommendations

1. Log in as a test user.
2. GET `/recommendations` — note movie A in the list.
3. POST `/feedback` marking movie A as `not_interested`.
4. GET `/recommendations` again (same session).
5. Movie A does not appear in the new results.
6. Log out and log back in.
7. GET `/recommendations` — movie A is still absent (signal persisted across sessions).

**What would make this fail:** movie A reappears after the feedback call; the `not_interested` flag is lost after logout; another user's recommendations are affected.

---

## 4. Session vector resets cleanly

1. Log in as a test user.
2. GET `/recommendations` — record the result set (call it "baseline").
3. POST `/chat` with `"only horror movies please"`.
4. Verify the updated picks are weighted toward horror.
5. POST `/chat/reset` (or equivalent reset endpoint).
6. GET `/recommendations` — the result set matches the baseline from step 2 (within the same session; base vector unchanged).

**What would make this fail:** the reset changes the base taste vector; the post-reset picks differ from the pre-chat baseline; the reset affects another user's session.

---

## 5. Cold-start user sees a usable onboarding flow, not an empty page

1. Create a new test user with zero watch history.
2. Log in as that user.
3. GET `/recommendations`.
4. The response returns at least 2 movies (popularity-based fallback), each with title, year, genres, and reason.
5. The response includes a flag or message indicating cold-start mode (e.g. `"cold_start": true` in the payload, or an onboarding prompt in the chat).
6. POST `/chat` with `"I like sci-fi"`.
7. The returned picks include at least one sci-fi title.

**What would make this fail:** an empty list or an error is returned; no onboarding signal is present; the sci-fi constraint is ignored; a cold-start user can see another user's history.
