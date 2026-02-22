"""
src/modeling.py
===============
Model training, pipeline construction, and serialization for the
Somaliland Drought Prediction project.

Models:
    Regression    : RandomForest, XGBoost, LightGBM → SPI-3
    Classification: RandomForest, XGBoost, LightGBM → Binary drought
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


# ── Pipeline factory ──────────────────────────────────────────────────────────

def make_pipeline(model: Any) -> Pipeline:
    """
    Wrap a scikit-learn compatible model in a preprocessing pipeline.

    Steps:
        1. SimpleImputer  — median imputation for remaining NaNs
        2. StandardScaler — zero-mean / unit-variance scaling
        3. model          — the estimator

    Tree-based models (RF, XGB, LGBM) are invariant to monotonic
    feature scaling, but scaling is included for consistency and
    to enable comparison with future linear/kernel models.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])


# ── Model definitions ─────────────────────────────────────────────────────────

def get_regression_models(random_state: int = RANDOM_SEED) -> dict:
    """
    Return a dictionary of untrained regression models.

    Hyperparameters are sensible defaults for medium-sized tabular datasets.
    For production: tune with Optuna or GridSearchCV.
    """
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=random_state,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        ),
    }


def get_classification_models(
    pos_weight: float = 1.0,
    random_state: int = RANDOM_SEED,
) -> dict:
    """
    Return a dictionary of untrained classification models.

    class_weight='balanced' / scale_pos_weight compensate for
    class imbalance (drought months are a minority class).

    Parameters
    ----------
    pos_weight : ratio of negative to positive samples (for XGBoost)
    """
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=random_state,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        ),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_SEED,
) -> dict:
    """
    Train all regression models and return fitted pipelines.

    Returns
    -------
    dict : {model_name: fitted Pipeline}
    """
    models = get_regression_models(random_state)
    pipelines = {}

    for name, model in models.items():
        logger.info(f"Training regression: {name} ...")
        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
        logger.info(f"  {name} — done.")

    return pipelines


def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_SEED,
) -> dict:
    """
    Train all classification models and return fitted pipelines.

    Returns
    -------
    dict : {model_name: fitted Pipeline}
    """
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = get_classification_models(pos_weight=pos_weight, random_state=random_state)
    pipelines = {}

    for name, model in models.items():
        logger.info(f"Training classification: {name} ...")
        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
        logger.info(f"  {name} — done.")

    return pipelines


# ── Train / Test Split ────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    feature_cols: list,
    train_end: str = "2015-12-31",
    test_start: str = "2016-01-01",
) -> dict:
    """
    Time-based train/test split.

    CRITICAL: Random split MUST NOT be used for time-series data —
    it leaks future information into the training set.

    Parameters
    ----------
    df : feature DataFrame with 'date', 'target_spi3', 'target_drought'
    feature_cols : list of feature column names
    train_end : last date for training (inclusive)
    test_start : first date for testing (inclusive)

    Returns
    -------
    dict with keys: X_train, X_test, y_train_reg, y_test_reg,
                    y_train_cls, y_test_cls, train_df, test_df
    """
    df_clean = df.dropna(subset=feature_cols + ["target_spi3"])

    train_df = df_clean[df_clean["date"] <= train_end].copy()
    test_df  = df_clean[df_clean["date"] >= test_start].copy()

    result = {
        "X_train":      train_df[feature_cols],
        "X_test":       test_df[feature_cols],
        "y_train_reg":  train_df["target_spi3"],
        "y_test_reg":   test_df["target_spi3"],
        "y_train_cls":  train_df["target_drought"],
        "y_test_cls":   test_df["target_drought"],
        "train_df":     train_df,
        "test_df":      test_df,
    }

    logger.info(
        f"Temporal split: train={len(train_df):,} "
        f"({train_df['date'].min().date()} → {train_df['date'].max().date()}), "
        f"test={len(test_df):,} "
        f"({test_df['date'].min().date()} → {test_df['date'].max().date()})"
    )

    return result


# ── Model Persistence ─────────────────────────────────────────────────────────

def save_pipeline(
    pipeline: Pipeline,
    name: str,
    task: str,
    save_dir: Path,
    compress: int = 3,
) -> Path:
    """
    Serialize a fitted pipeline to disk using joblib.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    name : model name (e.g., 'XGBoost')
    task : 'regression' or 'classification'
    save_dir : target directory
    compress : joblib compression level (0-9)

    Returns
    -------
    Path to saved file
    """
    fn = f"{task[:3]}_{name.lower()}_pipeline.pkl"
    dest = save_dir / fn
    joblib.dump(pipeline, dest, compress=compress)
    size_kb = dest.stat().st_size / 1024
    logger.info(f"Saved: {fn} ({size_kb:.0f} KB)")
    return dest


def load_pipeline(model_path: Path) -> Pipeline:
    """Load a serialized pipeline from disk."""
    pipe = joblib.load(model_path)
    logger.info(f"Loaded: {model_path.name}")
    return pipe


def save_all_models(
    reg_pipes: dict,
    cls_pipes: dict,
    feature_cols: list,
    save_dir: Path,
) -> list:
    """
    Save all regression and classification pipelines plus feature column list.

    Returns
    -------
    list of saved Path objects
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for name, pipe in reg_pipes.items():
        saved.append(save_pipeline(pipe, name, "regression", save_dir))

    for name, pipe in cls_pipes.items():
        saved.append(save_pipeline(pipe, name, "classification", save_dir))

    fc_path = save_dir / "feature_columns.pkl"
    joblib.dump(feature_cols, fc_path)
    saved.append(fc_path)
    logger.info(f"Feature columns saved: {fc_path.name}")

    return saved


def load_model_and_predict(
    model_path: Path,
    feature_cols_path: Path,
    new_data: pd.DataFrame,
    task: str = "regression",
) -> np.ndarray:
    """
    Load a saved pipeline and generate predictions on new data.

    Parameters
    ----------
    model_path : path to .pkl pipeline file
    feature_cols_path : path to feature_columns.pkl
    new_data : DataFrame with raw features (pre-engineering not required
               if the pipeline includes transformers)
    task : 'regression' or 'classification'

    Returns
    -------
    np.ndarray of predictions
    """
    pipe = load_pipeline(model_path)
    feat_cols = joblib.load(feature_cols_path)

    missing = set(feat_cols) - set(new_data.columns)
    if missing:
        raise ValueError(f"Missing features in new_data: {missing}")

    X = new_data[feat_cols]

    if task == "classification":
        return pipe.predict_proba(X)[:, 1]  # return probability

    return pipe.predict(X)
