from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def main() -> None:
    output = Path("data/heart_disease.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(URL, header=None, names=COLUMNS, na_values="?")
    frame["target"] = (frame.pop("num") > 0).astype(int)
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame)} rows to {output}")


if __name__ == "__main__":
    main()

