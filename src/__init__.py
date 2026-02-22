"""
Somaliland Drought Prediction — Source Package
"""

from .data_loader import load_all_data
from .feature_engineering import build_feature_matrix, get_feature_columns
from .modeling import (
    train_regression_models,
    train_classification_models,
    temporal_split,
    save_all_models,
    load_model_and_predict,
)
from .evaluation import (
    run_regression_evaluation,
    run_classification_evaluation,
    plot_model_comparison,
    plot_confusion_matrix_and_roc,
    plot_forecast_timeseries,
    plot_feature_importance,
    plot_shap_summary,
)
from .utils import setup_logging, set_seed, ensure_dirs, check_data_leakage

__all__ = [
    "load_all_data",
    "build_feature_matrix",
    "get_feature_columns",
    "train_regression_models",
    "train_classification_models",
    "temporal_split",
    "save_all_models",
    "load_model_and_predict",
    "run_regression_evaluation",
    "run_classification_evaluation",
    "plot_model_comparison",
    "plot_confusion_matrix_and_roc",
    "plot_forecast_timeseries",
    "plot_feature_importance",
    "plot_shap_summary",
    "setup_logging",
    "set_seed",
    "ensure_dirs",
    "check_data_leakage",
]
