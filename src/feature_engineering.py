"""
src/feature_engineering.py
===========================
Feature engineering pipeline for Somaliland drought prediction.

Produces:
    - SPI-1, SPI-3, SPI-6  (WMO gamma-distribution method)
    - Standardized anomalies (z-score vs 1981-2010 climatology)
    - Lag features (1, 3, 6 months)
    - Rolling means (3, 6, 12 months)
    - Derived / interaction features
    - Calendar & seasonal indicators
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
DISTRICT_COORDS = {
    "Hargeisa": (44.065, 9.560),  "Berbera": (45.014, 10.439),
    "Borama":   (43.183, 9.935),  "Burao":   (45.533,  9.517),
    "Erigavo":  (47.367, 10.617), "Las_Anod":(47.367,  8.483),
    "Gabiley":  (43.467, 9.983),  "Sheikh":  (45.200,  9.933),
    "Odweyne":  (45.067, 9.417),  "Zeila":   (43.483, 11.350),
}

REF_START = 1981
REF_END   = 2010

DEFAULT_LAGS    = (1, 3, 6)
DEFAULT_WINDOWS = (3, 6, 12)

DEFAULT_LAG_VARS = [
    "rainfall_mm", "spi_1", "spi_3",
    "temp_2m_c", "soil_moisture", "oni",
    "pet_mm", "rain_anom", "sm_anom", "asis_index",
]


# ── SPI ───────────────────────────────────────────────────────────────────────

def compute_spi(
    rain_series: pd.Series,
    scale: int = 3,
    ref_start: int = REF_START,
    ref_end: int = REF_END,
) -> pd.Series:
    """
    Standardized Precipitation Index (WMO standard gamma-distribution method).

    Algorithm
    ---------
    1. Compute rolling sum over `scale` months.
    2. Per calendar month: fit gamma distribution to 1981-2010 reference period.
    3. CDF → standard normal PPF = SPI.

    SPI classes (McKee et al. 1993):
        ≥ +2.0 : Extremely wet
        +1.0 → +1.99 : Moderately wet
        -0.99 → +0.99 : Near normal
        -1.0 → -1.49 : Moderate drought
        -1.5 → -1.99 : Severe drought
        ≤ -2.0 : Extreme drought

    Parameters
    ----------
    rain_series : pd.Series, datetime-indexed monthly rainfall (mm)
    scale : accumulation period in months
    ref_start, ref_end : reference period for gamma fitting

    Returns
    -------
    pd.Series of SPI values
    """
    rolled = rain_series.rolling(window=scale, min_periods=scale).sum()
    spi = pd.Series(np.nan, index=rain_series.index, name=f"spi_{scale}")

    if not hasattr(rolled.index, "month"):
        logger.warning("SPI: index has no month attribute — skipping.")
        return spi

    ref_mask = (
        (rolled.index.year >= ref_start) & (rolled.index.year <= ref_end)
    )

    for m in range(1, 13):
        m_mask  = rolled.index.month == m
        ref_dat = rolled[m_mask & ref_mask].dropna()
        all_dat = rolled[m_mask].dropna()

        if len(ref_dat) < 8 or len(all_dat) == 0:
            continue

        # Probability of zero (P0)
        p0 = (ref_dat == 0).mean()
        nz = ref_dat[ref_dat > 0]

        if len(nz) < 5:
            # Fall back to z-score
            if ref_dat.std() > 0:
                spi[all_dat.index] = (all_dat - ref_dat.mean()) / ref_dat.std()
            continue

        try:
            shape, _, scale_param = stats.gamma.fit(nz, floc=0)
            cdf = p0 + (1 - p0) * stats.gamma.cdf(
                all_dat, shape, scale=scale_param
            )
            spi[all_dat.index] = stats.norm.ppf(cdf.clip(0.001, 0.999))
        except Exception as e:
            logger.debug(f"Gamma fit failed month={m} scale={scale}: {e}")

    return spi


def add_spi_features(
    df: pd.DataFrame,
    scales: tuple = (1, 3, 6),
) -> pd.DataFrame:
    """Apply SPI computation across all districts and accumulation scales."""
    df = df.copy()
    for scale in scales:
        col = f"spi_{scale}"
        parts = []
        for district, grp in df.groupby("district"):
            rain = grp.set_index("date")["rainfall_mm"].sort_index()
            s = compute_spi(rain, scale=scale).reset_index()
            s.columns = ["date", col]
            s["district"] = district
            parts.append(s)
        spi_df = pd.concat(parts, ignore_index=True)
        df = df.merge(spi_df, on=["date", "district"], how="left")
        logger.info(
            f"SPI-{scale} computed: {df[col].notna().sum()} valid values"
        )
    return df


# ── Anomalies ─────────────────────────────────────────────────────────────────

def compute_anomaly(
    series: pd.Series,
    month_series: pd.Series,
    ref_years: tuple = (REF_START, REF_END),
) -> pd.Series:
    """
    Standardized anomaly (z-score) per calendar month.

    anomaly = (x - monthly_climatological_mean) / monthly_climatological_std

    Reference period follows WMO standard: 1981-2010.
    """
    anom = pd.Series(np.nan, index=series.index)
    if hasattr(series.index, "year"):
        ref_mask = (series.index.year >= ref_years[0]) & (series.index.year <= ref_years[1])
    else:
        ref_mask = pd.Series(True, index=series.index)

    for m in range(1, 13):
        mm = month_series == m
        ref_vals = series[mm & ref_mask]
        all_vals = series[mm]
        if len(ref_vals) > 2 and ref_vals.std() > 0:
            anom[mm] = (all_vals - ref_vals.mean()) / ref_vals.std()

    return anom


def add_anomaly_features(
    df: pd.DataFrame,
    variables: dict = None,
) -> pd.DataFrame:
    """
    Add standardized anomaly features for key climate variables.

    Parameters
    ----------
    df : master DataFrame (must have 'district', 'date', 'month' columns)
    variables : dict mapping {source_col: anomaly_col_name}
    """
    if variables is None:
        variables = {
            "rainfall_mm":   "rain_anom",
            "temp_2m_c":     "temp_anom",
            "soil_moisture": "sm_anom",
            "pet_mm":        "pet_anom",
        }

    df = df.copy()
    parts = []

    for district, grp in df.groupby("district"):
        grp = grp.sort_values("date").set_index("date").copy()
        months = grp["month"]

        for var, name in variables.items():
            if var in grp.columns:
                grp[name] = compute_anomaly(grp[var], months)

        parts.append(grp.reset_index())

    result = pd.concat(parts, ignore_index=True)
    logger.info(f"Anomaly features added: {list(variables.values())}")
    return result


# ── Lag & Rolling Features ────────────────────────────────────────────────────

def add_lag_rolling_features(
    df: pd.DataFrame,
    lag_vars: list = None,
    lags: tuple = DEFAULT_LAGS,
    windows: tuple = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """
    Add lag and rolling mean features per district.

    IMPORTANT: all operations are within-district to prevent
    spatial data leakage.

    Parameters
    ----------
    df : DataFrame sorted by (district, date)
    lag_vars : variables to lag; defaults to DEFAULT_LAG_VARS
    lags : lag periods in months (default: 1, 3, 6)
    windows : rolling window sizes in months (default: 3, 6, 12)
    """
    lag_vars = [v for v in (lag_vars or DEFAULT_LAG_VARS) if v in df.columns]
    parts = []

    for district, grp in df.groupby("district"):
        grp = grp.sort_values("date").copy()

        for var in lag_vars:
            for lag in lags:
                grp[f"{var}_lag{lag}"] = grp[var].shift(lag)
            for w in windows:
                grp[f"{var}_roll{w}m"] = (
                    grp[var].rolling(w, min_periods=max(1, w // 2)).mean()
                )

        parts.append(grp)

    result = pd.concat(parts, ignore_index=True)
    n_new = sum(
        1 for c in result.columns
        if any(f"_lag{l}" in c for l in lags)
        or any(f"_roll{w}m" in c for w in windows)
    )
    logger.info(f"Lag/rolling features added: {n_new} columns")
    return result


# ── Derived Features ──────────────────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physically-motivated composite and interaction features.

    Features
    --------
    aridity_idx   : P/PET ratio (Aridity Index — low = arid)
    water_deficit : PET − P (mm; positive = water stressed)
    drought_stress: temp_anom × (−sm_anom) — joint heat+moisture stress
    oni_gu        : ENSO × Gu season interaction
    oni_deyr      : ENSO × Deyr season interaction
    rain_yoy      : Year-on-year % change in rainfall
    """
    df = df.copy()

    df["aridity_idx"]   = df["rainfall_mm"] / (df["pet_mm"].clip(lower=1))
    df["water_deficit"] = (df["pet_mm"] - df["rainfall_mm"]).clip(lower=0)

    if "temp_anom" in df.columns and "sm_anom" in df.columns:
        df["drought_stress"] = df["temp_anom"] * (-df["sm_anom"].clip(upper=0))

    if "oni" in df.columns:
        df["oni_gu"]   = df["oni"] * df.get("is_gu", 0)
        df["oni_deyr"] = df["oni"] * df.get("is_deyr", 0)

    df = df.sort_values(["district", "date"])
    df["rain_yoy"] = (
        df.groupby("district")["rainfall_mm"].pct_change(12) * 100
    )

    logger.info("Derived features added.")
    return df


# ── Calendar & Spatial Features ───────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, season, and spatial features."""
    df = df.copy()
    df["year"]      = df["date"].dt.year
    df["month"]     = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    def season(m):
        return (
            "Gu" if m in [4, 5, 6]
            else "Deyr" if m in [10, 11]
            else "Hagaa" if m in [7, 8, 9]
            else "Jilaal"
        )

    df["season"]  = df["month"].apply(season)
    df["is_gu"]   = (df["season"] == "Gu").astype(int)
    df["is_deyr"] = (df["season"] == "Deyr").astype(int)

    district_codes = {
        d: i for i, d in enumerate(sorted(df["district"].unique()))
    }
    df["district_code"] = df["district"].map(district_codes)
    df["lon"] = df["district"].map(lambda d: DISTRICT_COORDS.get(d, (45, 9.5))[0])
    df["lat"] = df["district"].map(lambda d: DISTRICT_COORDS.get(d, (45, 9.5))[1])

    return df


# ── Target Variable ───────────────────────────────────────────────────────────

def create_targets(
    df: pd.DataFrame,
    horizon: int = 3,
    threshold: float = -1.0,
) -> pd.DataFrame:
    """
    Create forecast targets shifted `horizon` months into the future.

    IMPORTANT: Targets are created WITHIN each district group to
    prevent cross-district temporal leakage.

    Parameters
    ----------
    df : feature DataFrame (must have 'spi_3' and 'district' columns)
    horizon : forecast horizon in months (default: 3)
    threshold : SPI threshold for drought classification (default: -1.0)

    Returns
    -------
    DataFrame with added columns:
        target_spi3     : SPI-3 at t + horizon (regression)
        target_drought  : 1 if SPI-3 < threshold at t + horizon (classification)
    """
    parts = []
    for district, grp in df.groupby("district"):
        grp = grp.sort_values("date").copy()
        grp["target_spi3"]    = grp["spi_3"].shift(-horizon)
        grp["target_drought"] = (grp["target_spi3"] < threshold).astype(float)
        parts.append(grp)

    result = pd.concat(parts, ignore_index=True)
    n0 = len(result)
    result = result.dropna(subset=["target_spi3"])
    result["target_drought"] = result["target_drought"].astype(int)

    logger.info(
        f"Targets created (horizon={horizon}m). "
        f"Dropped {n0 - len(result)} rows. "
        f"Drought prevalence: {result['target_drought'].mean()*100:.1f}%"
    )
    return result


# ── Master Feature Pipeline ───────────────────────────────────────────────────

def build_feature_matrix(
    master_df: pd.DataFrame,
    spi_scales: tuple = (1, 3, 6),
    lag_vars: list = None,
    lags: tuple = DEFAULT_LAGS,
    windows: tuple = DEFAULT_WINDOWS,
    forecast_horizon: int = 3,
    drought_threshold: float = -1.0,
    min_year: int = 1985,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline applied in correct dependency order.

    Order
    -----
    1. Calendar / seasonal features
    2. SPI calculation
    3. Anomaly features
    4. Lag + rolling features
    5. Derived / interaction features
    6. Target variable creation

    Parameters
    ----------
    master_df : cleaned merged DataFrame from data_loader
    spi_scales : SPI accumulation periods to compute
    lag_vars : variables to lag (None = use defaults)
    lags : lag steps in months
    windows : rolling window sizes in months
    forecast_horizon : months ahead to predict
    drought_threshold : SPI threshold for binary drought label
    min_year : drop rows before this year (allow lag feature warm-up)

    Returns
    -------
    DataFrame ready for train/test split
    """
    logger.info("Starting feature engineering pipeline ...")

    df = add_calendar_features(master_df)
    df = add_spi_features(df, scales=spi_scales)
    df = add_anomaly_features(df)
    df = add_lag_rolling_features(df, lag_vars=lag_vars, lags=lags, windows=windows)
    df = add_derived_features(df)
    df = create_targets(df, horizon=forecast_horizon, threshold=drought_threshold)

    df = df[df["year"] >= min_year].reset_index(drop=True)

    logger.info(
        f"Feature matrix built: {df.shape[0]:,} rows × {df.shape[1]} columns"
    )
    return df


def get_feature_columns(
    df: pd.DataFrame,
    exclude: set = None,
) -> list:
    """
    Return model-ready feature columns (numeric, excluding targets/identifiers).

    Parameters
    ----------
    df : feature DataFrame
    exclude : additional column names to exclude

    Returns
    -------
    list of feature column names
    """
    default_exclude = {
        "date", "district", "season",
        "target_spi3", "target_drought",
        "spi_3", "spi_6",  # prevent current → future leakage (use lagged versions)
    }
    if exclude:
        default_exclude |= set(exclude)

    return [
        c for c in df.select_dtypes(include=[float, int]).columns
        if c not in default_exclude and not c.startswith("target_")
    ]


def merge_worldbank(
    df: pd.DataFrame,
    wb_df: pd.DataFrame,
) -> pd.DataFrame:
    """Expand World Bank annual data to monthly and merge into master DataFrame."""
    wb = wb_df.copy()
    if hasattr(wb.index, "year"):
        wb.index = pd.to_datetime(wb.index.astype(str), format="%Y")
    else:
        wb.index = pd.to_datetime(wb.index, format="%Y")

    all_months = pd.date_range(
        f"{wb.index.min().year}-01",
        f"{wb.index.max().year}-12",
        freq="MS",
    )
    wb = wb.reindex(all_months).ffill().bfill()
    wb.index.name = "date"
    return df.merge(wb.reset_index(), on="date", how="left")
