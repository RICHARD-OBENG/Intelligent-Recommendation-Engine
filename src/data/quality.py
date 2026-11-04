import polars as pl


def validate_ratings(
    ratings: pl.LazyFrame,
    movies: pl.LazyFrame,
) -> dict:

    results = {}

    # 1. Missing values
    missing = (
        ratings.select([
            pl.col(column)
            .null_count()
            .alias(column)
            for column in [
                "userId",
                "movieId",
                "rating",
                "timestamp",
            ]
        ])
        .collect()
        .to_dicts()[0]
    )

    results["missing_values"] = missing

    # 2. Invalid ratings
    invalid_ratings = (
        ratings.filter(
            (pl.col("rating") < 0.5)
            | (pl.col("rating") > 5.0)
        )
        .select(pl.len())
        .collect()
        .item()
    )

    results["invalid_ratings"] = invalid_ratings

    # 3. Invalid identifiers
    invalid_ids = (
        ratings.filter(
            (pl.col("userId") <= 0)
            | (pl.col("movieId") <= 0)
        )
        .select(pl.len())
        .collect()
        .item()
    )

    results["invalid_ids"] = invalid_ids

    # 4. Invalid timestamps
    invalid_timestamps = (
        ratings.filter(pl.col("timestamp") <= 0)
        .select(pl.len())
        .collect()
        .item()
    )

    results["invalid_timestamps"] = invalid_timestamps

    # 5. Duplicate user-item interactions
    duplicate_interactions = (
        ratings.group_by(["userId", "movieId"])
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") > 1)
        .select(pl.len())
        .collect()
        .item()
    )

    results["duplicate_user_movie_pairs"] = (
        duplicate_interactions
    )

    # 6. Ratings referring to nonexistent movies
    invalid_movie_references = (
        ratings.select("movieId")
        .unique()
        .join(
            movies.select("movieId").unique(),
            on="movieId",
            how="anti",
        )
        .select(pl.len())
        .collect()
        .item()
    )

    results["unknown_movie_references"] = (
        invalid_movie_references
    )

    return results