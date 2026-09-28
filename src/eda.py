"""Exploratory data analysis and feature signal ranking."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from config import RAW_CSV, FIGURES, PROCESSED, RANDOM_STATE
from features import engineer, OPTIMIZED_FEATURES

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
RED, BLUE = "#C0392B", "#2E86C1"


def feature_signal(d: pd.DataFrame) -> pd.DataFrame:
    numeric = d.select_dtypes(include=[np.number]).drop(columns=["persisted_next_term", "at_risk"])
    corr = numeric.corrwith(d["at_risk"])
    mi = mutual_info_classif(numeric.fillna(0), d["at_risk"], random_state=RANDOM_STATE)
    out = pd.DataFrame({
        "feature": numeric.columns,
        "correlation_with_risk": corr.values.round(4),
        "abs_correlation": corr.abs().values.round(4),
        "mutual_information": mi.round(4),
    }).sort_values("mutual_information", ascending=False)
    out["selected_for_model"] = out["feature"].isin(OPTIMIZED_FEATURES)
    return out


def plot_distributions(d: pd.DataFrame) -> None:
    cols = ["credit_completion_ratio", "term_gpa", "assignment_submission_rate",
            "login_trend", "days_since_last_login", "engagement_index"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, col in zip(axes.ravel(), cols):
        lo, hi = d[col].quantile([0.01, 0.99])
        bins = np.linspace(lo, hi, 35)
        ax.hist(d.loc[d.at_risk == 0, col].clip(lo, hi), bins=bins, alpha=0.6, color=BLUE, label="Persisted", density=True)
        ax.hist(d.loc[d.at_risk == 1, col].clip(lo, hi), bins=bins, alpha=0.6, color=RED, label="At Risk", density=True)
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
    axes[0, 0].legend()
    fig.suptitle("Key Feature Distributions: Persisted vs At-Risk Learners", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_feature_distributions.png")
    plt.close(fig)


def plot_signal(signal: pd.DataFrame) -> None:
    top = signal.head(20).iloc[::-1]
    colors = [RED if s else "#95A5A6" for s in top["selected_for_model"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["mutual_information"], color=colors)
    ax.set_xlabel("Mutual information with at-risk outcome")
    ax.set_title("Feature Signal Ranking (red = selected for optimized model)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_feature_signal.png")
    plt.close(fig)


def plot_rates(d: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, ["enrollment_status", "modality", "college"]):
        rate = d.groupby(col)["at_risk"].mean().sort_values(ascending=False) * 100
        ax.bar(rate.index, rate.values, color=RED)
        ax.set_title(f"At-Risk Rate by {col.replace('_', ' ').title()}")
        ax.set_ylabel("% at risk")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_risk_rates.png")
    plt.close(fig)


def main():
    d = engineer(pd.read_csv(RAW_CSV))
    print(f"Records: {len(d):,} | At risk: {d.at_risk.sum():,} ({d.at_risk.mean():.1%})")
    print(f"Missing values: {int(d.isna().sum().sum())}")
    signal = feature_signal(d)
    signal.to_csv(PROCESSED / "eda_feature_signal.csv", index=False)
    plot_distributions(d)
    plot_signal(signal)
    plot_rates(d)
    print("Top 8 signals:")
    print(signal.head(8)[["feature", "correlation_with_risk", "mutual_information"]].to_string(index=False))
    dropped = signal[~signal.selected_for_model]["feature"].tolist()
    print(f"Fields excluded from the optimized model: {dropped}")


if __name__ == "__main__":
    main()
