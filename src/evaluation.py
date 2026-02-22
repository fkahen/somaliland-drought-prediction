"""
src/evaluation.py
=================
Model evaluation, metric computation, and visualization for the
Somaliland Drought Prediction project.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate_regression(
    name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    verbose: bool = True,
) -> dict:
    """Compute regression evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)

    metrics = dict(model=name, rmse=rmse, r2=r2, mae=mae, bias=bias)

    if verbose:
        print(
            f"  {name:<15s} "
            f"RMSE={rmse:.4f}  R²={r2:.4f}  MAE={mae:.4f}  Bias={bias:+.4f}"
        )

    return metrics


def evaluate_classification(
    name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    verbose: bool = True,
) -> dict:
    """Compute classification evaluation metrics."""
    auc = (
        roc_auc_score(y_true, y_prob)
        if len(np.unique(y_true)) > 1
        else float("nan")
    )
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = (y_true == y_pred).mean()

    metrics = dict(
        model=name, auc=auc, f1=f1,
        precision=prec, recall=rec, accuracy=acc,
    )

    if verbose:
        print(
            f"  {name:<15s} "
            f"AUC={auc:.4f}  F1={f1:.4f}  "
            f"Prec={prec:.4f}  Rec={rec:.4f}  Acc={acc:.4f}"
        )

    return metrics


def run_regression_evaluation(
    reg_pipes: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate all regression models and return results DataFrame."""
    print("REGRESSION EVALUATION — SPI-3 (3-month ahead)")
    print("-" * 65)
    results = []
    preds = {}
    for name, pipe in reg_pipes.items():
        pred = pipe.predict(X_test)
        preds[name] = pred
        results.append(evaluate_regression(name, y_test, pred))
    return pd.DataFrame(results), preds


def run_classification_evaluation(
    cls_pipes: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple:
    """Evaluate all classification models and return results + predictions."""
    print("\nCLASSIFICATION EVALUATION — Drought binary (3-month ahead)")
    print("-" * 65)
    results = []
    preds, probs = {}, {}
    for name, pipe in cls_pipes.items():
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        preds[name] = pred
        probs[name] = prob
        results.append(evaluate_classification(name, y_test, pred, prob))
    return pd.DataFrame(results), preds, probs


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_model_comparison(
    reg_df: pd.DataFrame,
    cls_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Bar chart comparison of all models for regression and classification."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Model Comparison — Somaliland Drought Prediction",
        fontsize=15, fontweight="bold",
    )
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    def bar(ax, df, col, title, ylabel, higher_better=True):
        bars = ax.bar(df["model"], df[col], color=colors, edgecolor="black", alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        for bar_, val in zip(bars, df[col]):
            ax.text(
                bar_.get_x() + bar_.get_width() / 2,
                bar_.get_height() + (0.001 if higher_better else 0.01),
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9,
            )
        if not higher_better:
            ax.invert_yaxis()

    bar(axes[0, 0], reg_df, "rmse",  "Regression: RMSE ↓", "RMSE",      higher_better=False)
    bar(axes[0, 1], reg_df, "r2",    "Regression: R² ↑",   "R²",        higher_better=True)
    bar(axes[1, 0], cls_df, "auc",   "Classification: AUC ↑", "AUC",    higher_better=True)
    bar(axes[1, 1], cls_df, "f1",    "Classification: F1 ↑",  "F1",     higher_better=True)

    axes[1, 0].axhline(0.5, color="red", ls="--", lw=1, label="Random baseline")
    axes[1, 0].legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix_and_roc(
    cls_preds: dict,
    cls_probs: dict,
    y_true: pd.Series,
    best_model: str,
    save_path: Optional[Path] = None,
) -> None:
    """Confusion matrix for best model + ROC curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Classification Evaluation — {best_model}",
        fontsize=14, fontweight="bold",
    )

    cm = confusion_matrix(y_true, cls_preds[best_model])
    ConfusionMatrixDisplay(
        cm, display_labels=["No Drought", "Drought"]
    ).plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix")

    for name, prob in cls_probs.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        axes[1].plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curves — All Models")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nClassification Report — {best_model}:")
    print(
        classification_report(
            y_true, cls_preds[best_model],
            target_names=["No Drought", "Drought"],
        )
    )


def plot_forecast_timeseries(
    test_df: pd.DataFrame,
    pipe,
    feature_cols: list,
    district: str,
    model_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot observed vs. predicted SPI-3 time series for a district.
    Highlights drought episodes (SPI < -1) in red.
    """
    d = test_df[test_df["district"] == district].sort_values("date")
    y_obs  = d["target_spi3"].values
    y_pred = pipe.predict(d[feature_cols])
    dates  = d["date"].values

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(
        f"3-Month Ahead Drought Forecast — {district} | {model_name}\n"
        f"Test period: {pd.Timestamp(dates[0]).date()} → {pd.Timestamp(dates[-1]).date()}",
        fontsize=13, fontweight="bold",
    )

    ax1 = axes[0]
    ax1.plot(dates, y_obs,  color="steelblue", lw=2, label="Observed SPI-3", zorder=3)
    ax1.plot(dates, y_pred, color="orangered", lw=2, ls="--", label="Predicted SPI-3", zorder=3)
    ax1.axhline(-1, color="red", ls=":", lw=1.5, label="Drought threshold (SPI=-1)")
    ax1.axhline(0,  color="gray", lw=0.5)
    ax1.fill_between(dates, -4, 4, where=(y_obs < -1), color="red", alpha=0.15, label="Drought")
    ax1.set_ylim(-3.5, 3.5)
    ax1.set_ylabel("SPI-3")
    ax1.legend(fontsize=9)

    resid = y_obs - y_pred
    ax2 = axes[1]
    ax2.bar(dates, resid, color=np.where(resid >= 0, "steelblue", "tomato"), alpha=0.75)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Residual (Obs − Pred)")
    rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
    ax2.set_title(f"Residuals | RMSE = {rmse:.4f}", fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(
    pipe,
    feature_names: list,
    title: str,
    top_n: int = 25,
    save_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Horizontal bar chart of top feature importances."""
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        logger.warning(f"{type(model).__name__} has no feature_importances_.")
        return None

    fi = (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(fi)))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(fi["feature"][::-1], fi["importance"][::-1], color=colors[::-1])
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.set_xlabel("Feature Importance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fi


def plot_shap_summary(
    pipe,
    X: pd.DataFrame,
    feature_names: list,
    title: str,
    max_samples: int = 500,
    save_path: Optional[Path] = None,
) -> None:
    """SHAP summary plot for tree-based models."""
    try:
        import shap
        X_t = pipe[:-1].transform(X.head(max_samples))
        explainer = shap.TreeExplainer(pipe.named_steps["model"])
        sv = explainer.shap_values(X_t)
        if isinstance(sv, list):
            sv = sv[1]  # positive class for classifiers

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            sv, X_t, feature_names=feature_names,
            show=False, max_display=20,
        )
        plt.title(title, fontweight="bold", fontsize=13)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
    except ImportError:
        logger.warning("shap not installed — skipping SHAP analysis.")
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: list,
    save_path: Optional[Path] = None,
) -> None:
    """Lower-triangle correlation heatmap."""
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_seasonal_analysis(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Four-panel seasonal analysis plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Somaliland Seasonal Climate Analysis\n"
        "Gu (Apr–Jun) | Deyr (Oct–Nov) | Hagaa (Jul–Sep) | Jilaal (Dec–Mar)",
        fontsize=13, fontweight="bold",
    )
    s_colors = {"Gu": "#27AE60", "Deyr": "#2980B9", "Hagaa": "#E67E22", "Jilaal": "#E74C3C"}
    s_order  = ["Gu", "Deyr", "Hagaa", "Jilaal"]

    # Seasonal boxplot
    data_s = [df[df["season"] == s]["rainfall_mm"].dropna() for s in s_order]
    bp = axes[0, 0].boxplot(data_s, labels=s_order, patch_artist=True)
    for patch, s in zip(bp["boxes"], s_order):
        patch.set_facecolor(s_colors[s]); patch.set_alpha(0.75)
    axes[0, 0].set_title("Rainfall by Season"); axes[0, 0].set_ylabel("mm/month")

    # Monthly climatology
    mc = df.groupby("month")["rainfall_mm"].mean()
    m_colors = [
        "#27AE60" if m in [4, 5, 6] else "#2980B9" if m in [10, 11] else "#E74C3C"
        for m in range(1, 13)
    ]
    axes[0, 1].bar(range(1, 13), mc.values, color=m_colors, edgecolor="white")
    axes[0, 1].set_xticks(range(1, 13))
    axes[0, 1].set_xticklabels(list("JFMAMJJASOND"))
    axes[0, 1].set_title("Monthly Climatology"); axes[0, 1].set_ylabel("Mean Rainfall (mm)")

    # Drought frequency by district
    df_by_d = (
        df.groupby("district").apply(lambda g: (g["spi_3"] < -1).mean() * 100)
        .sort_values()
    )
    c_d = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(df_by_d)))
    axes[1, 0].barh(df_by_d.index, df_by_d.values, color=c_d)
    axes[1, 0].set_title("Drought Frequency by District")
    axes[1, 0].set_xlabel("Months with SPI-3 < -1 (%)")

    # Annual trend
    ann = df.groupby("year").apply(lambda g: (g["spi_3"] < -1).mean() * 100)
    axes[1, 1].plot(ann.index, ann.values, color="tomato", lw=2, marker="o", ms=3)
    axes[1, 1].fill_between(ann.index, 0, ann.values, color="tomato", alpha=0.2)
    z = np.polyfit(ann.index, ann.values, 1)
    axes[1, 1].plot(ann.index, np.poly1d(z)(ann.index), "k--", lw=1.5,
                    label=f"Trend: {z[0]:+.2f}%/yr")
    axes[1, 1].set_title("Annual Drought Trend")
    axes[1, 1].set_ylabel("Districts in Drought (%)")
    axes[1, 1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
