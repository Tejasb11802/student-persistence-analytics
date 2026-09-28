"""
Train Random Forest persistence models.

1. Baseline: all raw exported fields, no feature engineering.
2. Optimized: EDA-driven engineered and selected features.

The decision threshold is tuned on out-of-fold training predictions to reach
the target recall (early-warning use case), then evaluated on a held-out test set.
"""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from config import RAW_CSV, FIGURES, MODELS, REPORTS, RANDOM_STATE, TARGET_RECALL
from features import engineer, baseline_matrix, OPTIMIZED_FEATURES


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500, min_samples_leaf=8, max_features="sqrt",
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)


def threshold_for_recall(y_true, proba, target_recall: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = np.where(recall[:-1] >= target_recall)[0]
    return float(thresholds[ok[-1]]) if len(ok) else 0.5


def fit_and_evaluate(name, X, y, idx_train, idx_test):
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(make_model(), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = threshold_for_recall(y_train, oof, TARGET_RECALL)

    model = make_model().fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "model": name,
        "n_features": X.shape[1],
        "threshold": round(threshold, 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
    }
    return model, metrics, proba, pred, y_test


def plot_roc(results) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, y_test, proba, auc in results:
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve: Held-Out Test Set", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_roc_curve.png")
    plt.close(fig)


def plot_confusion(y_test, pred) -> None:
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Reds")
    labels = ["Persisted", "At Risk"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Optimized Model: Confusion Matrix", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_confusion_matrix.png")
    plt.close(fig)


def plot_importance(importance: pd.DataFrame) -> None:
    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(top["feature"], top["importance"], color="#C0392B")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Top Drivers of At-Risk Prediction", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_feature_importance.png")
    plt.close(fig)


def main():
    raw = pd.read_csv(RAW_CSV)
    d = engineer(raw)
    y = d["at_risk"]
    idx = np.arange(len(d))
    idx_train, idx_test = train_test_split(idx, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    X_base = baseline_matrix(raw)
    X_opt = d[OPTIMIZED_FEATURES]

    _, m_base, p_base, _, y_test = fit_and_evaluate("Baseline (raw fields)", X_base, y, idx_train, idx_test)
    model, m_opt, p_opt, pred_opt, _ = fit_and_evaluate("Optimized (EDA features)", X_opt, y, idx_train, idx_test)

    comparison = pd.DataFrame([m_base, m_opt])
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    print(comparison.to_string(index=False))
    lift = (m_opt["precision"] - m_base["precision"]) * 100
    print(f"\nPrecision improvement from EDA feature optimization: +{lift:.1f} pts at {TARGET_RECALL:.0%} target recall")

    importance = pd.DataFrame({"feature": OPTIMIZED_FEATURES, "importance": model.feature_importances_}) \
        .sort_values("importance", ascending=False)
    importance.to_csv(REPORTS / "feature_importance.csv", index=False)

    plot_roc([("Baseline", y_test, p_base, m_base["roc_auc"]), ("Optimized", y_test, p_opt, m_opt["roc_auc"])])
    plot_confusion(y_test, pred_opt)
    plot_importance(importance)

    final = make_model().fit(X_opt, y)  # refit on all data for scoring and deployment
    joblib.dump({"model": final, "features": OPTIMIZED_FEATURES, "threshold": m_opt["threshold"]},
                MODELS / "rf_persistence_model.joblib", compress=3)
    summary = {"test_metrics": m_opt, "baseline_metrics": m_base,
               "precision_lift_pts": round(lift, 2), "n_records": int(len(d)),
               "n_at_risk": int(y.sum()), "target_recall": TARGET_RECALL}
    (REPORTS / "model_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved model to {MODELS / 'rf_persistence_model.joblib'}")


if __name__ == "__main__":
    main()
