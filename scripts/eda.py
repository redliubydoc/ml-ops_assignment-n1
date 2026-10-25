from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.pipeline import load_dataset


def main() -> None:
    X, y = load_dataset("data/heart_disease.csv")
    output = Path("artifacts/eda")
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    X.hist(figsize=(12, 8), bins=20)
    plt.tight_layout()
    plt.savefig(output / "feature_histograms.png", dpi=160)
    plt.close()
    plt.figure(figsize=(5, 4))
    y.value_counts().sort_index().plot.bar(color=["#4c72b0", "#dd8452"])
    plt.xticks([0, 1], ["No disease", "Disease"], rotation=0)
    plt.ylabel("Patients")
    plt.title("Target class balance")
    plt.tight_layout()
    plt.savefig(output / "class_balance.png", dpi=160)
    plt.close()
    plt.figure(figsize=(10, 8))
    sns.heatmap(X.assign(target=y).corr(numeric_only=True), cmap="vlag", center=0)
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(output / "correlation_heatmap.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()

