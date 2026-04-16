from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.services.metadata import (
    BatchMetadata,
    get_batch_metadata,
    get_latest_batch_metadata,
)
from app.services.storage import download_object_to_buffer


TARGET_COLUMN = "isFraud"
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

BASE_NUMERIC_FEATURES = (
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
)

BASE_CATEGORICAL_FEATURES = ("type",)

ENGINEERED_NUMERIC_FEATURES = (
    "origin_balance_delta",
    "destination_balance_delta",
    "origin_balance_error",
    "destination_balance_error",
)


@dataclass(slots=True)
class TrainingDatasetMetadata:
    """Describes how the training-ready dataset was built from one raw batch."""

    batch_id: int
    dataset_version: str
    file_name: str
    storage_path: str
    target_column: str
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    raw_rows_count: int
    train_rows_count: int
    validation_rows_count: int
    train_positive_rate: float
    validation_positive_rate: float


@dataclass(slots=True)
class PreparedTrainingDataset:
    """Stores preprocessed train/validation data together with the fitted pipeline."""

    X_train: Any
    X_validation: Any
    y_train: pd.Series
    y_validation: pd.Series
    preprocessor: ColumnTransformer
    metadata: TrainingDatasetMetadata
    raw_dataframe: pd.DataFrame | None = field(default=None, repr=False)


def _resolve_batch_metadata(
    batch_id: int | None = None,
    dataset_version: str | None = None,
) -> BatchMetadata:
    """Selects the requested batch explicitly or falls back to the newest uploaded one."""

    if batch_id is not None:
        return get_batch_metadata(batch_id)

    return get_latest_batch_metadata(dataset_version=dataset_version)


def _load_raw_dataframe(storage_path: str, nrows: int | None = None) -> pd.DataFrame:
    """Downloads one raw CSV from MinIO and loads it into a DataFrame."""

    buffer = download_object_to_buffer(storage_path)
    return pd.read_csv(buffer, nrows=nrows)


def _validate_training_columns(dataframe: pd.DataFrame) -> None:
    """Ensures the raw dataset contains the columns needed for baseline training."""

    required_columns = {
        TARGET_COLUMN,
        *BASE_NUMERIC_FEATURES,
        *BASE_CATEGORICAL_FEATURES,
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "amount",
    }
    missing_columns = sorted(required_columns.difference(dataframe.columns))

    if missing_columns:
        raise ValueError(
            "The raw dataset is missing columns required for training: "
            f"{', '.join(missing_columns)}"
        )


def _add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Adds simple balance-based features."""

    dataframe = dataframe.copy()
    dataframe["origin_balance_delta"] = (
        dataframe["oldbalanceOrg"] - dataframe["newbalanceOrig"]
    )
    dataframe["destination_balance_delta"] = (
        dataframe["newbalanceDest"] - dataframe["oldbalanceDest"]
    )
    dataframe["origin_balance_error"] = (
        dataframe["oldbalanceOrg"] - dataframe["amount"] - dataframe["newbalanceOrig"]
    )
    dataframe["destination_balance_error"] = (
        dataframe["oldbalanceDest"] + dataframe["amount"] - dataframe["newbalanceDest"]
    )
    return dataframe


def _select_training_columns(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Builds the baseline feature frame and target vector for model training."""

    _validate_training_columns(dataframe)
    dataframe = _add_engineered_features(dataframe)

    feature_columns = [
        *BASE_NUMERIC_FEATURES,
        *ENGINEERED_NUMERIC_FEATURES,
        *BASE_CATEGORICAL_FEATURES,
    ]

    # We intentionally exclude nameOrig and nameDest here:
    # they are anonymized high-cardinality IDs, so naive one-hot encoding would
    # explode the feature space and add little value to a baseline model.
    X = dataframe.loc[:, feature_columns].copy()
    y = dataframe[TARGET_COLUMN].astype(int)
    return X, y


def _build_preprocessor() -> ColumnTransformer:
    """Builds a simple preprocessing pipeline for the baseline fraud model."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, [*BASE_NUMERIC_FEATURES, *ENGINEERED_NUMERIC_FEATURES]),
            ("categorical", categorical_pipeline, list(BASE_CATEGORICAL_FEATURES)),
        ]
    )


def prepare_training_dataset(
    batch_id: int | None = None,
    dataset_version: str | None = None,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    nrows: int | None = None,
    include_raw_dataframe: bool = False,
) -> PreparedTrainingDataset:
    """Downloads one uploaded batch and turns it into train/validation datasets.

    Args:
        batch_id: Exact uploaded batch ID from PostgreSQL metadata.
        dataset_version: Optional dataset version when the newest batch should be selected.
        validation_size: Fraction reserved for validation split.
        random_state: Random seed for reproducible train/validation split.
        nrows: Optional cap for local experiments on very large raw CSV files.
        include_raw_dataframe: Returns the raw DataFrame when downstream code needs it.
    """

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    batch_metadata = _resolve_batch_metadata(
        batch_id=batch_id,
        dataset_version=dataset_version,
    )
    raw_dataframe = _load_raw_dataframe(batch_metadata.storage_path, nrows=nrows)

    if raw_dataframe.empty:
        raise ValueError("The selected raw dataset is empty and cannot be used for training.")

    X, y = _select_training_columns(raw_dataframe)

    if y.nunique() < 2:
        raise ValueError(
            "The selected raw dataset must contain at least two target classes for training."
        )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = _build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_validation_processed = preprocessor.transform(X_validation)

    metadata = TrainingDatasetMetadata(
        batch_id=batch_metadata.id,
        dataset_version=batch_metadata.dataset_version,
        file_name=batch_metadata.file_name,
        storage_path=batch_metadata.storage_path,
        target_column=TARGET_COLUMN,
        feature_columns=list(X.columns),
        numeric_features=[*BASE_NUMERIC_FEATURES, *ENGINEERED_NUMERIC_FEATURES],
        categorical_features=list(BASE_CATEGORICAL_FEATURES),
        raw_rows_count=len(raw_dataframe),
        train_rows_count=len(X_train),
        validation_rows_count=len(X_validation),
        train_positive_rate=float(y_train.mean()),
        validation_positive_rate=float(y_validation.mean()),
    )

    return PreparedTrainingDataset(
        X_train=X_train_processed,
        X_validation=X_validation_processed,
        y_train=y_train.reset_index(drop=True),
        y_validation=y_validation.reset_index(drop=True),
        preprocessor=preprocessor,
        metadata=metadata,
        raw_dataframe=raw_dataframe if include_raw_dataframe else None,
    )
