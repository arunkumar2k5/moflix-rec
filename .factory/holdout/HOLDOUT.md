# Holdout scenarios

Independent edge-case and combination scenarios for moflix-rec.
These are verified separately from the builder's view.
Start from a clean database state for each scenario.

---

## H1. Cross-user data isolation — two users never see each other's data

1. Seed user A with a history of sci-fi films and user B with a history of romantic comedies.
2. Log in as user A. GET `/recommendations`. Record the returned movie IDs.
3. Log out. Log in as user B. GET `/recommendations`. Record the returned movie IDs.
4. The two result sets share no movies (their histories are fully disjoint genres).
5. Log in as user A. POST `/chat` with `"I want something longer"`. Verify the updated picks belong only to user A's taste vector.
6. Log in as user B. Verify user B's picks are unchanged from step 3.
7. Attempt to GET `/recommendations?user_id=<user_A_id>` while authenticated as user B. The server returns 403 or ignores the parameter and returns user B's own data.

**What would make this fail:** any of user A's movies appear in user B's results; user B can read user A's session or taste vector; the explicit `user_id` override succeeds.

---

## H2. Scoring is deterministic and never calls an external API

1. Seed 500 movies with known vectors and one user with a fixed taste vector.
2. Block all outbound network calls (mock or firewall rule).
3. GET `/recommendations` three times in succession.
4. All three responses return identical ordered lists.
5. No network call is logged or attempted during any of the three scoring passes.

**What would make this fail:** the order differs between calls; a network call is made during scoring; the endpoint errors when the network is blocked.

---

## H3. LLM API failure falls back gracefully without breaking the session

1. Configure the LLM endpoint to return a 500 error (mock).
2. Log in as a test user with watch history.
3. POST `/chat` with `"something funny"`.
4. The response completes successfully (no 500 to the client).
5. The reply contains a user-visible "AI is unavailable, using simple mode" notice (or equivalent).
6. The returned picks are updated using the keyword/genre fallback parser: at least one comedy or comedy-adjacent result appears.
7. POST `/chat` again with `"actually make it scary"` — fallback still works.
8. Restore the LLM endpoint. POST `/chat` — full LLM path resumes without restarting the app.

**What would make this fail:** the client receives a 500; the picks do not update at all; the session is invalidated; recovering the LLM requires a restart.

---

## H4. Compound hard filter — runtime AND language exclusion

1. Log in as a test user whose library contains English, French, and Korean films of varying runtimes.
2. POST `/chat` with `"only films under 90 minutes and not in English"`.
3. GET `/recommendations`.
4. Every returned movie has runtime < 90 minutes AND language ≠ English.
5. POST `/chat` with `"actually, English is fine"` — language filter lifted.
6. The updated picks may include English films again, but still respect the runtime filter.
7. POST `/chat/reset` — both filters are cleared; English films and films over 90 minutes return.

**What would make this fail:** any result violates an active filter; lifting one filter silently lifts both; reset fails to clear the runtime filter.

---

## H5. Feedback signals survive restart and do not contaminate other users

1. Seed user A and user B with overlapping libraries (share at least 10 movies).
2. Log in as user A. Mark movies X, Y, Z as `not_interested`.
3. Mark movie W as `watch_tonight`.
4. Restart the app process completely.
5. Log in as user A. GET `/recommendations` — X, Y, Z absent; W (if unwatched) remains excluded from re-recommendation (or the watch-tonight list shows it).
6. Log in as user B. GET `/recommendations` — X, Y, Z may appear normally (user B gave no feedback on them).

**What would make this fail:** X, Y, or Z appears for user A after restart; user B's recommendations are suppressed for X, Y, or Z; W's feedback is lost.
