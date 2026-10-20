import numpy as np

from app.vectors import DIMENSION, LAYOUT, Movie, build_movie_vector


def test_fixture_meets_coverage_requirements(library):
    movies = library["movies"]
    assert len(movies) >= 20
    assert len({g for m in movies for g in m["genres"]}) >= 5
    assert len({m["year"] // 10 for m in movies}) >= 3
    assert len(library["users"]) >= 2
    assert all(len(user["watched"]) >= 5 for user in library["users"])


def test_known_fixture_movie_yields_exact_vector(library):
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
