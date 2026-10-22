"""Keyword parser that stands in for the LLM when its API is unavailable (PRD FR-24).

It recognises only a handful of phrasings; anything else yields an empty `Intent`, which leaves
the picks unchanged rather than guessing.
"""

import re

from app.intent import Era, FilterSet, HardFilters, Intent, MoodPace, Runtime
from app.vectors import Genre

GENRE_WORDS: dict[str, Genre] = {
    r"action": "Action",
    r"adventures?": "Adventure",
    r"animat(?:ed|ion)|cartoons?": "Animation",
    r"comed(?:y|ies)|funn(?:y|ier)": "Comedy",
    r"crime": "Crime",
    r"documentar(?:y|ies)": "Documentary",
    r"dramas?": "Drama",
    r"family": "Family",
    r"fantasy": "Fantasy",
    r"histor(?:y|ical)": "History",
    r"horror|scary": "Horror",
    r"musicals?": "Music",
    r"myster(?:y|ies)": "Mystery",
    r"romance|romantic": "Romance",
    r"sci-?fi|science fiction": "Science Fiction",
    r"thrillers?": "Thriller",
    r"war": "War",
    r"westerns?": "Western",
}
GENRE_PATTERN = re.compile(
    r"\b(?:" + "|".join(f"(?P<g{i}>{w})" for i, w in enumerate(GENRE_WORDS)) + r")\b"
)
EXCLUDE_BEFORE = re.compile(r"\b(?:no|without)\s+(?:more\s+)?$")
REQUIRE_BEFORE = re.compile(r"\bonly\s+$")
LESS_BEFORE = re.compile(
    r"\b(?:less|fewer|not|not into|hate|dislike|don't like|do not like|tired of|sick of)\s+$"
)

DECADE = re.compile(r"\b(19|20)?(\d)0'?s\b")
SHORTER_THAN = re.compile(
    r"\b(?:under|less than|shorter than|at most|no more than|no longer than|nothing over"
    r"|nothing longer than)\s+(\d+(?:\.\d+)?)\s*(h|hrs?|hours?|m|mins?|minutes?)\b"
)
NOT_ANOTHER = re.compile(r"\b(?:not|no)\s+(?:another|more)\s+([a-z][\w-]*)")
NOT_A_TERM = re.compile(rf"(?:{'|'.join(GENRE_WORDS)})s?|movies?|films?|of")
SIMILAR_TO = re.compile(r"\b(?:similar to|something like|movies? like)\s+([^,.;!?]+)", re.I)
TITLE_END = re.compile(r"\s+(?:but|and|with|please)\b.*$", re.I)

LANGUAGE_WORDS = {
    "english": "en",
    "french": "fr",
    "spanish": "es",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "hindi": "hi",
    "bollywood": "hi",
}
LANGUAGE_PATTERN = re.compile(r"\b(" + "|".join(LANGUAGE_WORDS) + r")\b")

MOOD_WORDS: dict[str, MoodPace] = {
    r"light(?:er|-hearted)?|feel-good|cheer(?:ful| me up)": "lighter",
    r"dark(?:er)?|gritty|bleak": "darker",
    r"fast(?:er|-paced)?|exciting|thrilling": "faster",
    r"slow(?:er)?(?: burn)?|calm(?:er)?|relaxing": "slower",
}


def _genres(text: str) -> tuple[dict[Genre, float], set[Genre], set[Genre]]:
    adjust: dict[Genre, float] = {}
    excluded: set[Genre] = set()
    required: set[Genre] = set()
    genres = list(GENRE_WORDS.values())
    for match in GENRE_PATTERN.finditer(text):
        genre = genres[int(match.lastgroup[1:])]
        before = text[: match.start()]
        if EXCLUDE_BEFORE.search(before):
            excluded.add(genre)
        elif REQUIRE_BEFORE.search(before):
            required.add(genre)
        else:
            adjust[genre] = -1.0 if LESS_BEFORE.search(before) else 1.0
    return adjust, excluded, required


def _era(text: str) -> Era | None:
    match = DECADE.search(text)
    if match is None:
        return None
    century, decade = match.groups()
    if century is None:
        century = "19" if int(decade) >= 3 else "20"
    start = int(f"{century}{decade}0")
    return Era(start=start, end=start + 9)


def _runtime(text: str) -> Runtime | None:
    match = SHORTER_THAN.search(text)
    if match is None:
        return None
    amount, unit = float(match.group(1)), match.group(2)
    return Runtime(max=round(amount * 60 if unit.startswith("h") else amount))


def _mood_pace(text: str) -> MoodPace | None:
    for words, mood in MOOD_WORDS.items():
        if re.search(rf"\b(?:{words})\b", text):
            return mood
    return None


def simple_parse(text: str) -> Intent:
    lowered = text.lower()
    adjust, excluded_genres, required_genres = _genres(lowered)
    excluded_terms = {
        word for word in NOT_ANOTHER.findall(lowered) if not NOT_A_TERM.fullmatch(word)
    }
    similar_to = {TITLE_END.sub("", match.group(1)).strip() for match in SIMILAR_TO.finditer(text)}
    return Intent(
        genre_adjust=adjust,
        era=_era(lowered),
        runtime=_runtime(lowered),
        language=frozenset(LANGUAGE_WORDS[word] for word in LANGUAGE_PATTERN.findall(lowered)),
        mood_pace=_mood_pace(lowered),
        similar_to=frozenset(title for title in similar_to if title),
        hard_filters=HardFilters(
            exclude=FilterSet(genres=frozenset(excluded_genres), terms=frozenset(excluded_terms)),
            require=FilterSet(genres=frozenset(required_genres)),
        ),
    )
