import json
from pathlib import Path


PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed"

LEAKY_COLUMNS = [
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_pdisposition",
]


def test_setup_c_excludes_leaky_features():
    with open(PROCESSED_PATH / "feature_sets.json", "r") as f:
        feature_sets = json.load(f)

    leak_free = feature_sets["setup_c_leak_free"]

    for col in LEAKY_COLUMNS:
        assert col not in leak_free, f"Leaky column {col} found in leak-free feature set"
