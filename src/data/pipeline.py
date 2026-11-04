import json
import logging
from pathlib import Path

import polars as pl

from src.data.analytics import run_analytics
from src.data.processing import (
    RAW_DIR,
    TABLES,
    convert_all_tables,
)
from src.data.profiling import profile_csv
from src.data.quality import validate_ratings

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT / "reports"

PROFILE_DIR = REPORT_DIR / "profiling"

QUALITY_DIR = REPORT_DIR / "data_quality"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def run_pipeline():

    logger.info("PHASE 02 DATA PROCESSING STARTED")

    # STEP 1: Profile source datasets
    logger.info("Running data profiling...")

    for table_name in TABLES:

        profile_csv(
            RAW_DIR / f"{table_name}.csv",
            table_name,
            PROFILE_DIR,
        )

    # STEP 2: Validate ratings and movie references
    logger.info("Running data quality validation...")

    ratings = pl.scan_csv(
        RAW_DIR / "ratings.csv",
        schema_overrides=TABLES["ratings"]["schema"],
        infer_schema_length=10000,
    )

    movies = pl.scan_csv(
        RAW_DIR / "movies.csv",
        schema_overrides=TABLES["movies"]["schema"],
        infer_schema_length=10000,
    )

    quality_results = validate_ratings(
        ratings,
        movies,
    )

    QUALITY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        QUALITY_DIR / "quality_report.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            quality_results,
            file,
            indent=4,
        )

    # Fail when critical quality constraints are violated.
    critical_failures = {
        "missing_values": any(
            count > 0
            for count in quality_results["missing_values"].values()
        ),
        "invalid_ratings": quality_results["invalid_ratings"] > 0,
        "invalid_ids": quality_results["invalid_ids"] > 0,
        "invalid_timestamps": (
            quality_results["invalid_timestamps"] > 0
        ),
        "duplicate_user_movie_pairs": (
            quality_results["duplicate_user_movie_pairs"] > 0
        ),
        "unknown_movie_references": (
            quality_results["unknown_movie_references"] > 0
        ),
    }

    failures = [
        name
        for name, failed in critical_failures.items()
        if failed
    ]

    if failures:
        raise ValueError(
            f"Data quality checks failed: {failures}"
        )

    logger.info("Data quality checks passed.")

    # STEP 3: Convert CSV to Parquet
    logger.info("Converting CSV datasets to Parquet...")

    convert_all_tables()

    # STEP 4: Recommendation analytics
    logger.info("Running recommendation analytics...")

    run_analytics()

    logger.info("PHASE 02 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_pipeline()