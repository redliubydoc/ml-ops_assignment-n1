import pandas as pd
import pytest

from src.pipeline import FEATURES, build_preprocessor, load_dataset


def test_preprocessor_handles_missing_values() -> None:
    frame = pd.DataFrame([{feature: 1 for feature in FEATURES} for _ in range(2)])
    frame.loc[0, "chol"] = None
    transformed = build_preprocessor().fit_transform(frame)
    assert transformed.shape[0] == 2
    assert transformed.shape[1] >= len(FEATURES)


def test_load_dataset_rejects_missing_columns(tmp_path) -> None:
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"age": [50], "target": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        load_dataset(path)
