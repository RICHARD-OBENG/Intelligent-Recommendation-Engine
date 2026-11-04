from __future__ import annotations

import csv
import logging
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_URL = (
    "https://files.grouplens.org/"
    "datasets/movielens/ml-32m.zip"
)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "movielens"
ARCHIVE_PATH = RAW_DIR / "ml-32m.zip"
DATASET_DIR = RAW_DIR / "ml-32m"

REQUIRED_COLUMNS = {
    "ratings.csv": [
        "userId", "movieId", "rating", "timestamp"
    ],
    "movies.csv": [
        "movieId", "title", "genres"
    ],
    "tags.csv": [
        "userId", "movieId", "tag", "timestamp"
    ],
    "links.csv": [
        "movieId", "imdbId", "tmdbId"
    ],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def download_dataset() -> None:
    """Download MovieLens dataset without loading it into memory."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists() and zipfile.is_zipfile(ARCHIVE_PATH):
        logger.info("Dataset archive already exists.")
        return

    logger.info("Downloading MovieLens 32M...")

    temporary_path = ARCHIVE_PATH.with_suffix(".tmp")

    try:
        request = urllib.request.Request(
            DATA_URL,
            headers={"User-Agent": "RecommendationEngine/1.0"},
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            total_size = int(
                response.headers.get("Content-Length", 0)
            )

            downloaded = 0

            with temporary_path.open("wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    file.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        percentage = downloaded / total_size * 100
                        logger.info(
                            "Downloaded: %.1f%%",
                            percentage,
                        )

        if not zipfile.is_zipfile(temporary_path):
            raise ValueError("Downloaded file is not a valid ZIP archive.")

        temporary_path.replace(ARCHIVE_PATH)

        logger.info("Dataset downloaded successfully.")

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def extract_dataset() -> None:
    """Extract archive after validating its contents."""

    if DATASET_DIR.exists() and all(
        (DATASET_DIR / filename).is_file()
        for filename in REQUIRED_COLUMNS
    ):
        logger.info("Dataset already extracted.")
        return

    logger.info("Checking ZIP archive...")

    with zipfile.ZipFile(ARCHIVE_PATH, "r") as archive:

        corrupted_file = archive.testzip()

        if corrupted_file:
            raise ValueError(
                f"Corrupted archive member: {corrupted_file}"
            )

        root = RAW_DIR.resolve()

        for member in archive.infolist():
            member_path = PurePosixPath(member.filename)

            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(
                    f"Unsafe archive path: {member.filename}"
                )

            destination = (RAW_DIR / Path(*member_path.parts)).resolve()

            if destination != root and root not in destination.parents:
                raise ValueError(
                    f"Archive path escapes destination: {member.filename}"
                )

        archive.extractall(RAW_DIR)

    missing_files = [
        filename
        for filename in REQUIRED_COLUMNS
        if not (DATASET_DIR / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Extracted dataset is missing required files: "
            + ", ".join(missing_files)
        )

    logger.info("Dataset extracted successfully.")


def validate_dataset() -> dict[str, int]:
    """Validate dataset schemas and representative records."""

    results = {}

    for filename, expected_columns in REQUIRED_COLUMNS.items():

        file_path = DATASET_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required dataset file missing: {filename}"
            )

        logger.info("Validating %s", filename)

        row_count = 0

        with file_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            if reader.fieldnames != expected_columns:
                raise ValueError(
                    f"Unexpected columns in {filename}: "
                    f"{reader.fieldnames}"
                )

            for row in reader:
                row_count += 1

                if row_count <= 1000:
                    if None in row or any(
                        value is None for value in row.values()
                    ):
                        raise ValueError(
                            f"Malformed record in {filename} at row "
                            f"{row_count + 1}"
                        )

                    if filename == "ratings.csv":
                        try:
                            rating = float(row["rating"])
                            user_id = int(row["userId"])
                            movie_id = int(row["movieId"])
                            timestamp = int(row["timestamp"])
                        except (TypeError, ValueError) as error:
                            raise ValueError(
                                f"Invalid rating data in {filename} at "
                                f"row {row_count + 1}"
                            ) from error

                        if not 0.5 <= rating <= 5.0:
                            raise ValueError(
                                f"Invalid rating: {rating}"
                            )

                        if user_id <= 0 or movie_id <= 0:
                            raise ValueError(
                                "Invalid user or movie identifier."
                            )

                        if timestamp <= 0:
                            raise ValueError("Invalid timestamp.")

        if row_count == 0:
            raise ValueError(f"{filename} contains no records.")

        results[filename] = row_count

    return results


def run_pipeline() -> None:
    logger.info("Starting MovieLens ingestion pipeline.")

    download_dataset()
    extract_dataset()

    results = validate_dataset()

    logger.info("INGESTION COMPLETED SUCCESSFULLY")

    for filename, count in results.items():
        logger.info("%s: %s records", filename, f"{count:,}")


if __name__ == "__main__":
    run_pipeline()