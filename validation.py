from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd


REQUIRED_COLUMNS = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
)

NUMERIC_COLUMNS = (
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)

STRING_COLUMNS = (
    "type",
    "nameOrig",
    "nameDest",
)

BINARY_COLUMNS = (
    "isFraud",
    "isFlaggedFraud",
)


@dataclass(slots=True)
class CSVValidationResult:
    """Stores the validation outcome in the exact shape expected by Sprint 1."""

    is_valid: bool
    rows_count: int
    message: str

    def to_dict(self) -> dict[str, bool | int | str]:
        """Converts the result to a plain dict for easy FastAPI responses later."""

        return asdict(self)


def _read_csv(csv_file: str | Path | BinaryIO) -> pd.DataFrame:
    """Loads the CSV into a DataFrame so the ingestion flow can validate its shape."""

    if hasattr(csv_file, "seek"):
        csv_file.seek(0)

    return pd.read_csv(csv_file)


def _find_invalid_type_columns(dataframe: pd.DataFrame) -> list[str]:
    """Checks that each required column contains values compatible with the expected type."""

    invalid_columns: list[str] = []

    for column in NUMERIC_COLUMNS:
        non_null_values = dataframe[column].dropna()
        converted_values = pd.to_numeric(non_null_values, errors="coerce")
        if converted_values.isna().any():
            invalid_columns.append(column)

    for column in STRING_COLUMNS:
        non_null_values = dataframe[column].dropna()
        if not non_null_values.map(lambda value: isinstance(value, str)).all():
            invalid_columns.append(column)

    for column in BINARY_COLUMNS:
        non_null_values = dataframe[column].dropna()
        converted_values = pd.to_numeric(non_null_values, errors="coerce")
        if converted_values.isna().any() or not converted_values.isin((0, 1)).all():
            invalid_columns.append(column)

    return invalid_columns


def _find_empty_required_columns(dataframe: pd.DataFrame) -> list[str]:
    """Flags required columns that are completely empty and unusable for ingestion."""

    return [
        column
        for column in REQUIRED_COLUMNS
        if column in dataframe.columns and dataframe[column].isna().all()
    ]


def validate_csv(csv_file: str | Path | BinaryIO) -> CSVValidationResult:
    """Validates a fraud-detection CSV before it is uploaded into the platform."""

    try:
        df = _read_csv(csv_file)
    except Exception as error:
        return CSVValidationResult(
            is_valid=False,
            rows_count=0,
            message=f"Failed to read CSV file: {error}",
        )

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        return CSVValidationResult(
            is_valid=False,
            rows_count=0,
            message=f"Missing required columns: {', '.join(missing_columns)}",
        )

    if df.empty:
        return CSVValidationResult(
            is_valid=False,
            rows_count=0,
            message="CSV file contains no rows.",
        )

    empty_required_columns = _find_empty_required_columns(df)
    if empty_required_columns:
        return CSVValidationResult(
            is_valid=False,
            rows_count=len(df),
            message=(
                "Required columns contain only empty values: "
                f"{', '.join(empty_required_columns)}"
            ),
        )

    invalid_type_columns = _find_invalid_type_columns(df)
    if invalid_type_columns:
        return CSVValidationResult(
            is_valid=False,
            rows_count=len(df),
            message=(
                "Columns contain values with unexpected types: "
                f"{', '.join(invalid_type_columns)}"
            ),
        )

    return CSVValidationResult(
        is_valid=True,
        rows_count=len(df),
        message="CSV file is valid.",
    )
