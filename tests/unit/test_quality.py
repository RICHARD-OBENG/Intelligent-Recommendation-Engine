import polars as pl

from src.data.quality import validate_ratings


def create_test_data():

    ratings = pl.DataFrame({
        "userId": [1, 1, 2, 3],
        "movieId": [10, 20, 10, 30],
        "rating": [5.0, 4.0, 3.0, 2.5],
        "timestamp": [1000, 2000, 3000, 4000],
    }).lazy()

    movies = pl.DataFrame({
        "movieId": [10, 20, 30],
        "title": ["Movie A", "Movie B", "Movie C"],
        "genres": ["Action", "Drama", "Comedy"],
    }).lazy()

    return ratings, movies


def test_valid_dataset():

    ratings, movies = create_test_data()

    result = validate_ratings(ratings, movies)

    assert result["invalid_ratings"] == 0
    assert result["invalid_ids"] == 0
    assert result["invalid_timestamps"] == 0
    assert result["duplicate_user_movie_pairs"] == 0
    assert result["unknown_movie_references"] == 0


def test_duplicate_interactions():

    ratings = pl.DataFrame({
        "userId": [1, 1],
        "movieId": [10, 10],
        "rating": [5.0, 4.0],
        "timestamp": [1000, 2000],
    }).lazy()

    movies = pl.DataFrame({
        "movieId": [10],
    }).lazy()

    result = validate_ratings(ratings, movies)

    assert result["duplicate_user_movie_pairs"] == 1


def test_invalid_rating():

    ratings = pl.DataFrame({
        "userId": [1],
        "movieId": [10],
        "rating": [6.0],
        "timestamp": [1000],
    }).lazy()

    movies = pl.DataFrame({
        "movieId": [10],
    }).lazy()

    result = validate_ratings(ratings, movies)

    assert result["invalid_ratings"] == 1