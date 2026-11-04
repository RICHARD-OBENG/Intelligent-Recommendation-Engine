import json
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ANALYTICS_DIR = PROCESSED_DIR / "analytics"

ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, filename: str) -> None:

    output_path = ANALYTICS_DIR / filename

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_data():

    ratings = pl.scan_parquet(
        PROCESSED_DIR / "ratings" / "ratings.parquet"
    )

    movies = pl.scan_parquet(
        PROCESSED_DIR / "movies" / "movies.parquet"
    )

    return ratings, movies


def analyze_interactions():

    ratings, movies = load_data()

    # Overall dataset statistics
    summary = (
        ratings.select(
            pl.len().alias("total_interactions"),
            pl.col("userId").n_unique().alias("unique_users"),
            pl.col("movieId").n_unique().alias("unique_movies"),
            pl.col("rating").mean().alias("average_rating"),
            pl.col("rating").min().alias("minimum_rating"),
            pl.col("rating").max().alias("maximum_rating"),
            pl.col("timestamp").min().alias("first_timestamp"),
            pl.col("timestamp").max().alias("last_timestamp"),
        )
        .collect()
        .to_dicts()[0]
    )

    interactions = summary["total_interactions"]
    users = summary["unique_users"]
    items = summary["unique_movies"]

    possible_interactions = users * items

    density = interactions / possible_interactions

    sparsity = 1 - density

    summary["interaction_density"] = density
    summary["interaction_sparsity"] = sparsity

    save_json(summary, "dataset_summary.json")

    return summary


def analyze_rating_distribution():

    ratings, _ = load_data()

    distribution = (
        ratings.group_by("rating")
        .agg(pl.len().alias("count"))
        .sort("rating")
        .with_columns(
            (
                pl.col("count")
                / pl.col("count").sum()
                * 100
            ).alias("percentage")
        )
    )

    distribution.sink_parquet(
        ANALYTICS_DIR / "rating_distribution.parquet",
        compression="zstd",
    )


def analyze_user_activity():

    ratings, _ = load_data()

    activity = (
        ratings.group_by("userId")
        .agg(
            pl.len().alias("interaction_count"),
            pl.col("movieId").n_unique().alias("unique_movies"),
            pl.col("rating").mean().alias("average_rating"),
        )
        .sort(
            "interaction_count",
            descending=True,
        )
    )

    activity.sink_parquet(
        ANALYTICS_DIR / "user_activity.parquet",
        compression="zstd",
    )


def analyze_item_popularity():

    ratings, movies = load_data()

    popularity = (
        ratings.group_by("movieId")
        .agg(
            pl.len().alias("interaction_count"),
            pl.col("userId").n_unique().alias("unique_users"),
            pl.col("rating").mean().alias("average_rating"),
        )
        .join(
            movies.select(["movieId", "title", "genres"]),
            on="movieId",
            how="left",
        )
        .sort(
            "interaction_count",
            descending=True,
        )
    )

    popularity.sink_parquet(
        ANALYTICS_DIR / "item_popularity.parquet",
        compression="zstd",
    )


def analyze_temporal_patterns():

    ratings, _ = load_data()

    temporal = (
        ratings
        .with_columns(
            pl.from_epoch(
                pl.col("timestamp"),
                time_unit="s",
            ).alias("event_time")
        )
        .with_columns(
            pl.col("event_time").dt.date().alias("event_date")
        )
        .group_by("event_date")
        .agg(
            pl.len().alias("interaction_count"),
            pl.col("userId").n_unique().alias("active_users"),
            pl.col("movieId").n_unique().alias("unique_movies"),
            pl.col("rating").mean().alias("average_rating"),
        )
        .sort("event_date")
    )

    temporal.sink_parquet(
        ANALYTICS_DIR / "temporal_activity.parquet",
        compression="zstd",
    )


def analyze_positive_interactions():

    ratings, _ = load_data()

    positive = (
        ratings.filter(
            pl.col("rating") >= 4.0
        )
        .select(
            "userId",
            "movieId",
            "rating",
            "timestamp",
        )
    )

    positive.sink_parquet(
        ANALYTICS_DIR / "positive_interactions.parquet",
        compression="zstd",
    )


def run_analytics():

    print("Starting recommendation analytics...")

    analyze_interactions()

    analyze_rating_distribution()

    analyze_user_activity()

    analyze_item_popularity()

    analyze_temporal_patterns()

    analyze_positive_interactions()

    print("Recommendation analytics completed.")