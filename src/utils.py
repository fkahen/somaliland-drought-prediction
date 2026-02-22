"""
src/utils.py
============
Shared utilities for the Somaliland Drought Prediction project.
"""

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure root logger with console (and optional file) handlers.

    Parameters
    ----------
    level : logging level (default: INFO)
    log_file : optional path to write log file

    Returns
    -------
    logging.Logger
    """
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy and random."""
    np.random.seed(seed)
    random.seed(seed)


# ── Path helpers ──────────────────────────────────────────────────────────────

def ensure_dirs(*paths: Path) -> None:
    """Create directories if they do not exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def data_paths(base: Optional[Path] = None) -> dict:
    """Return standard project data paths."""
    root = base or project_root()
    return {
        "raw":      root / "data" / "raw",
        "processed": root / "data" / "processed",
        "models":   root / "models",
        "figures":  root / "reports" / "figures",
        "notebooks": root / "notebooks",
    }


# ── Data helpers ──────────────────────────────────────────────────────────────

def summarize_df(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print a concise summary of a DataFrame."""
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Shape      : {df.shape}")
    if "date" in df.columns:
        print(f"  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    if "district" in df.columns:
        print(f"  Districts  : {sorted(df['district'].unique())}")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if len(miss) > 0:
        print(f"  Missing:\n{miss.to_string()}")
    else:
        print("  Missing    : none")
    print(f"{'='*55}\n")


def check_data_leakage(
    feature_cols: list,
    target_cols: list = ("target_spi3", "target_drought"),
    forbidden: tuple = ("spi_3", "spi_6"),
) -> None:
    """
    Sanity check: raise ValueError if any forbidden future-leaking
    columns appear in the feature set.
    """
    leakers = [c for c in feature_cols if c in target_cols or c in forbidden]
    if leakers:
        raise ValueError(
            f"Potential data leakage detected in feature columns: {leakers}"
        )
    logging.getLogger(__name__).info("✅ No data leakage detected in feature columns.")


def describe_targets(df: pd.DataFrame) -> None:
    """Print target variable statistics."""
    print("\nTarget: SPI-3 (regression)")
    print(df["target_spi3"].describe().round(4).to_string())
    print("\nTarget: Drought (classification)")
    counts = df["target_drought"].value_counts().rename({0: "No Drought", 1: "Drought"})
    print(counts.to_string())
    print(f"Drought prevalence: {df['target_drought'].mean()*100:.1f}%")


# ── Feature helpers ───────────────────────────────────────────────────────────

def feature_summary(
    feature_cols: list,
    df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Return a DataFrame summarising each feature.
    If df is provided, includes mean, std, min, max.
    """
    summary = pd.DataFrame({"feature": feature_cols})
    summary["category"] = summary["feature"].apply(_categorise_feature)
    if df is not None:
        for stat in ["mean", "std", "min", "max"]:
            summary[stat] = summary["feature"].map(
                df[feature_cols].agg(stat).round(4)
            )
    return summary


def _categorise_feature(col: str) -> str:
    """Assign a human-readable category label to a feature column name."""
    if "spi" in col:          return "SPI"
    if "rain" in col:         return "Rainfall"
    if "temp" in col:         return "Temperature"
    if "soil" in col or "sm" in col: return "Soil moisture"
    if "oni" in col:          return "ENSO"
    if "pet" in col:          return "Evapotranspiration"
    if "asis" in col:         return "ASIS"
    if "lag" in col:          return "Lag feature"
    if "roll" in col:         return "Rolling mean"
    if "anom" in col:         return "Anomaly"
    if col in ["year", "month", "month_sin", "month_cos"]: return "Calendar"
    if col in ["is_gu", "is_deyr", "season"]:  return "Season"
    if col in ["lon", "lat", "district_code"]: return "Spatial"
    if "gdp" in col or "rural" in col or "agr" in col: return "Socioeconomic"
    return "Derived"


# ── Formatting ────────────────────────────────────────────────────────────────

def print_header(text: str, width: int = 60, char: str = "=") -> None:
    """Print a formatted section header."""
    border = char * width
    print(f"\n{border}")
    print(f"  {text}")
    print(f"{border}\n")


def format_bytes(n_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"
