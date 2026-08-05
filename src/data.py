import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


def load_processed_data(path: Path) -> tuple:
    train = pd.read_parquet(path / "train.parquet")
    val = pd.read_parquet(path / "val.parquet")
    test = pd.read_parquet(path / "test.parquet")

    with open(path / "feature_sets.json", "r") as f:
        feature_sets = json.load(f)

    return train, val, test, feature_sets


def split_features_target(df: pd.DataFrame) -> tuple:
    return df.drop(columns=["target"]), df["target"]


def prepare_dataloaders(
    features: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    batch_size: int = 256,
) -> tuple:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train[features])
    X_va = scaler.transform(X_val[features])
    X_te = scaler.transform(X_test[features])

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_te, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler
