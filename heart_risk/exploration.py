"""Professional, labelled plots; descriptive EDA does not fit transformers."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from heart_risk.data import acquire_data
from heart_risk.settings import NUMERIC_FEATURES, ROOT


def create_eda():
    data = acquire_data()
    folder = ROOT / "reports/figures"
    folder.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="colorblind")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, name in zip(axes.flat, NUMERIC_FEATURES):
        sns.histplot(data=data, x=name, hue="heart_disease", bins=15, ax=axis,
                     element="step", common_norm=False)
        axis.set_title(name.replace("_", " ").title())
        axis.set_xlabel(name.replace("_", " "))
    axes.flat[-1].axis("off")
    fig.suptitle("Cleveland cohort: continuous feature distributions", fontsize=16)
    fig.tight_layout()
    fig.savefig(folder / "feature_distributions.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), cmap="vlag", center=0, vmin=-1, vmax=1, ax=axis,
                xticklabels=[c.replace("_", " ") for c in data.columns],
                yticklabels=[c.replace("_", " ") for c in data.columns])
    axis.set_title("Pearson correlations (categorical codes are descriptive only)")
    fig.tight_layout()
    fig.savefig(folder / "correlation_heatmap.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = data.heart_disease.value_counts().sort_index()
    axes[0].bar(["Absent (0)", "Present (1)"], counts.values, color=["#246a82", "#d57b30"])
    for i, count in enumerate(counts.values):
        axes[0].text(i, count + 2, str(count), ha="center")
    axes[0].set(title="Target class balance", ylabel="Patients", ylim=(0, counts.max() * 1.2))
    missing = data.isna().sum()
    missing[missing > 0].plot.bar(ax=axes[1], color="#246a82")
    axes[1].set(title="Missing feature values", ylabel="Missing records")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(folder / "class_balance_and_missingness.png", dpi=160)
    plt.close(fig)
    summary = {"rows": len(data), "features": len(data.columns)-1,
               "class_counts": counts.to_dict(), "missing_values": missing.to_dict(),
               "duplicate_rows": int(data.duplicated().sum())}
    (ROOT / "reports/eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
