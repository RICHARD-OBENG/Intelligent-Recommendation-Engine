import json
from pathlib import Path

import polars as pl


def profile_csv(
    file_path: Path,
    table_name: str,
    output_dir: Path,
) -> dict:

    df = pl.scan_csv(
        file_path,
        infer_schema_length=10000,
        null_values=["", "NA", "N/A", "null"],
    )

    schema = df.collect_schema()
    columns = schema.names()

    # Total records
    total_rows = df.select(pl.len()).collect().item()

    # Missing values
    null_counts = (
        df.select([
            pl.col(column)
            .null_count()
            .alias(column)
            for column in columns
        ])
        .collect()
        .to_dicts()[0]
    )

    missing_report = {
        column: {
            "missing_count": int(count),
            "missing_percentage": (
                round(count / total_rows * 100, 4)
                if total_rows else 0
            ),
        }
        for column, count in null_counts.items()
    }

    report = {
        "table": table_name,
        "total_rows": total_rows,
        "total_columns": len(columns),
        "columns": columns,
        "schema": {
            name: str(dtype)
            for name, dtype in schema.items()
        },
        "missing_values": missing_report,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{table_name}_profile.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return report