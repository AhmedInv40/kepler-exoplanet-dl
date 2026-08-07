from pathlib import Path

import pandas as pd


PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed"


def test_processed_data_exists():
    assert (PROCESSED_PATH / "train.parquet").exists()
    assert (PROCESSED_PATH / "val.parquet").exists()
    assert (PROCESSED_PATH / "test.parquet").exists()
    assert (PROCESSED_PATH / "feature_sets.json").exists()


def test_splits_have_correct_shape():
    train = pd.read_parquet(PROCESSED_PATH / "train.parquet")
    val = pd.read_parquet(PROCESSED_PATH / "val.parquet")
    test = pd.read_parquet(PROCESSED_PATH / "test.parquet")

    assert train.shape[0] > val.shape[0]
    assert "target" in train.columns
    assert "target" in val.columns
    assert "target" in test.columns
    assert train.shape[1] == val.shape[1] == test.shape[1]
