from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "movielens"
    / "ml-32m"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


TABLES = {
    "ratings": {
        "schema": {
            "userId": pl.Int32,
            "movieId": pl.Int32,
            "rating": pl.Float32,
            "timestamp": pl.Int64,
        }
    },
    "movies": {
        "schema": {
            "movieId": pl.Int32,
            "title": pl.String,
            "genres": pl.String,
        }
    },
    "tags": {
        "schema": {
            "userId": pl.Int32,
            "movieId": pl.Int32,
            "tag": pl.String,
            "timestamp": pl.Int64,
        }
    },
    "links": {
        "schema": {
            "movieId": pl.Int32,
            "imdbId": pl.String,
            "tmdbId": pl.String,
        }
    },
}


def convert_to_parquet(
    table_name: str,
) -> Path:

    csv_path = RAW_DIR / f"{table_name}.csv"

    output_dir = PROCESSED_DIR / table_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / f"{table_name}.parquet"

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    schema = TABLES[table_name]["schema"]

    # Lazy CSV scanning with explicit data types
    lazy_df = pl.scan_csv(
        csv_path,
        schema_overrides=schema,
        infer_schema_length=10000,
        null_values=["", "NA", "N/A", "null"],
    )

    # Streaming-capable Parquet writer
    lazy_df.sink_parquet(
        output_path,
        compression="zstd",
        row_group_size=100000,
        maintain_order=True,
    )

    print(
        f"{table_name} converted successfully: "
        f"{output_path}"
    )

    return output_path


def convert_all_tables() -> None:

    for table_name in TABLES:

        convert_to_parquet(table_name)