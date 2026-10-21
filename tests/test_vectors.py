import numpy as np

from app.vectors import (
    DIMENSION,
    EMBEDDING_DIM,
    FULL_DIMENSION,
    LAYOUT,
    Movie,
    _movie_text,
    build_full_vector,
    build_movie_vector,
    embed_texts,
)


def test_fixture_meets_coverage_requirements(library):
    movies = library["movies"]
    assert len(movies) >= 20
    assert len({g for m in movies for g in m["genres"]}) >= 5
    assert len({m["year"] // 10 for m in movies}) >= 3
    assert len(library["users"]) >= 2
    assert all(len(user["watched"]) >= 5 for user in library["users"])


def test_known_fixture_movie_yields_exact_structured_vector(library):
    godfather = Movie(**library["movies"][0])
    assert godfather.title == "The Godfather"

    expected = np.zeros(106)
    expected[4] = 1.0  # genre Crime
    expected[6] = 1.0  # genre Drama
    expected[20] = 1.0  # decade 1970s
    expected[29] = 1.0  # runtime >= 150 min
    expected[30] = 1.0  # language en
    expected[40] = 1.0  # country US
    expected[54] = 1.0  # certification R
    expected[57] = 0.87  # rating 8.7 / 10
    expected[58 + 8] = 1.0  # director hash bucket
    expected[74 + 1] = 1.0  # cast hash buckets
    expected[74 + 6] = 1.0
    expected[74 + 31] = 1.0

    np.testing.assert_allclose(build_movie_vector(godfather), expected, rtol=0, atol=1e-12)


def test_unknown_values_fall_into_other_slots():
    movie = Movie(
        id=99,
        title="Obscure",
        year=1931,
        genres=["Silent"],
        runtime=60,
        language="sv",
        country="SE",
        certification="Unrated",
        rating=12.0,
        director="Nobody",
        cast=[],
    )
    vector = build_movie_vector(movie)

    assert not vector[LAYOUT["genres"]].any()
    assert vector[LAYOUT["decade"]][0] == 1.0
    assert vector[LAYOUT["runtime"]][0] == 1.0
    assert vector[LAYOUT["language"]][-1] == 1.0
    assert vector[LAYOUT["country"]][-1] == 1.0
    assert vector[LAYOUT["certification"]][-1] == 1.0
    assert vector[LAYOUT["rating"]][0] == 1.0
    assert not vector[LAYOUT["cast"]].any()


def test_every_vector_has_declared_dimension(library):
    assert all(build_movie_vector(Movie(**m)).shape == (DIMENSION,) for m in library["movies"])


def test_full_dimension_constant():
    assert FULL_DIMENSION == DIMENSION + EMBEDDING_DIM
    assert FULL_DIMENSION == 490


def test_movie_text_combines_overview_and_keywords():
    movie = Movie(
        id=1,
        title="X",
        year=2000,
        genres=[],
        runtime=90,
        language="en",
        country="US",
        certification="PG",
        rating=7.0,
        director="D",
        cast=[],
        overview="A hero saves the world.",
        keywords=["action", "hero"],
    )
    text = _movie_text(movie)
    assert "hero saves the world" in text
    assert "action" in text
    assert "hero" in text


def test_movie_text_empty_overview_and_keywords():
    movie = Movie(
        id=2,
        title="Y",
        year=2000,
        genres=[],
        runtime=90,
        language="en",
        country="US",
        certification="PG",
        rating=7.0,
        director="D",
        cast=[],
    )
    assert _movie_text(movie) == ""


def test_embed_texts_uses_provided_model(fake_embedder):
    texts = ["A thriller set in space.", "A romantic comedy in Paris."]
    result = embed_texts(texts, model=fake_embedder)
    assert result.shape == (2, EMBEDDING_DIM)
    assert result.dtype == np.float32
    # Fake embedder returns normalised vectors — norm should be ~1.
    np.testing.assert_allclose(np.linalg.norm(result[0]), 1.0, atol=1e-5)


def test_build_full_vector_shape_and_structure(fake_embedder):
    movie = Movie(
        id=1,
        title="The Godfather",
        year=1972,
        genres=["Crime", "Drama"],
        runtime=175,
        language="en",
        country="US",
        certification="R",
        rating=8.7,
        director="Francis Ford Coppola",
        cast=["Marlon Brando", "Al Pacino", "James Caan"],
        overview="A crime family drama.",
        keywords=["mafia"],
    )
    texts = [_movie_text(movie)]
    embedding = embed_texts(texts, model=fake_embedder)[0]
    full = build_full_vector(movie, embedding)

    assert full.shape == (FULL_DIMENSION,)
    # Structured prefix must equal build_movie_vector output.
    np.testing.assert_array_equal(full[:DIMENSION], build_movie_vector(movie))
    # Embedding suffix must equal the input embedding (cast to float64).
    np.testing.assert_allclose(full[DIMENSION:], embedding.astype(np.float64), atol=1e-6)
