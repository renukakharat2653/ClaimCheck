"""
Tests for ClaimCheck data preprocessing and persistence.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from backend.ml.preprocess_data import (
    ClaimDataPreprocessor,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
)


def test_schema_columns_defined():
    assert TARGET_COLUMN == "FraudFound_P"
    assert "PolicyNumber" in IDENTIFIER_COLUMNS
    assert len(NUMERICAL_COLUMNS) >= 6
    assert len(CATEGORICAL_COLUMNS) >= 20


def test_preprocessor_fitting_and_transformation():
    # Use synthetic fixture
    fixture_path = "data/processed/synthetic_fixture.csv"
    assert os.path.exists(fixture_path), "Synthetic fixture must exist."

    df = pd.read_csv(fixture_path)
    assert TARGET_COLUMN in df.columns

    X = df.drop(columns=[c for c in IDENTIFIER_COLUMNS if c in df.columns] + [TARGET_COLUMN])
    preprocessor = ClaimDataPreprocessor()
    preprocessor.fit(X)

    assert preprocessor.median_age is not None
    assert preprocessor.mode_day_claimed is not None

    transformed = preprocessor.transform(X)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] > 50


def test_feature_schema_persistence():
    schema_path = "backend/models/feature_schema.json"
    assert os.path.exists(schema_path), "Feature schema JSON must be generated."

    with open(schema_path, "r") as f:
        schema = json.load(f)

    assert schema["target_column"] == "FraudFound_P"
    assert "categorical_options" in schema
    assert "Make" in schema["categorical_options"]
    assert "numerical_ranges" in schema
