"""
streamlit_app/app.py
====================
Interactive Drought Early Warning Dashboard for Somaliland.

Run:
    streamlit run streamlit_app/app.py

Features:
    - District selector
    - 3-month ahead drought probability forecast
    - Historical SPI-3 time series
    - Real-time prediction from saved model
    - Downloadable forecast CSV
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Somaliland Drought Early Warning",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────
DISTRICTS = [
    "Hargeisa", "Berbera", "Borama", "Burao", "Erigavo",
    "Las_Anod", "Gabiley", "Sheikh", "Odweyne", "Zeila",
]

DISTRICT_COORDS = {
    "Hargeisa": (44.065, 9.560),  "Berbera":  (45.014, 10.439),
    "Borama":   (43.183, 9.935),  "Burao":    (45.533,  9.517),
    "Erigavo":  (47.367, 10.617), "Las_Anod": (47.367,  8.483),
    "Gabiley":  (43.467, 9.983),  "Sheikh":   (45.200,  9.933),
    "Odweyne":  (45.067, 9.417),  "Zeila":    (43.483, 11.350),
}

SPI_CATEGORIES = [
    (-99, -2.0, "Extreme Drought", "#8B0000"),
    (-2.0, -1.5, "Severe Drought",   "#DC143C"),
    (-1.5, -1.0, "Moderate Drought", "#FF6347"),
    (-1.0,  0.0, "Mild Dry",         "#FFA500"),
    ( 0.0,  1.0, "Near Normal",      "#90EE90"),
    ( 1.0,  1.5, "Moderately Wet",   "#4169E1"),
    ( 1.5,  99,  "Very Wet",         "#00008B"),
]

MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data" / "processed"


def spi_to_category(spi: float) -> tuple:
    """Return (label, color) for a given SPI value."""
    for lo, hi, label, color in SPI_CATEGORIES:
        if lo <= spi < hi:
            return label, color
    return "Near Normal", "#90EE90"


def drought_probability_label(prob: float) -> tuple:
    """Return (label, color) for a drought probability."""
    if prob >= 0.7:   return "High Risk",   "#DC143C"
    elif prob >= 0.4: return "Moderate Risk","#FFA500"
    else:             return "Low Risk",     "#27AE60"


# ── Data / model loading ──────────────────────────────────────
@st.cache_data(ttl=3600)
def load_processed_data() -> pd.DataFrame | None:
    """Load processed feature data if available."""
    p = DATA_DIR / "features_full.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"])
        return df
    return None


@st.cache_resource
def load_models() -> tuple:
    """Load best available regression and classification pipelines."""
    reg_pipe, cls_pipe, feature_cols = None, None, None

    fc_path = MODELS_DIR / "feature_columns.pkl"
    if fc_path.exists():
        feature_cols = joblib.load(fc_path)

    # Try LightGBM first (usually best), then XGBoost, then RF
    for name in ["lightgbm", "xgboost", "randomforest"]:
        reg_path = MODELS_DIR / f"reg_{name}_pipeline.pkl"
        cls_path = MODELS_DIR / f"cls_{name}_pipeline.pkl"
        if reg_path.exists() and reg_pipe is None:
            reg_pipe = joblib.load(reg_path)
        if cls_path.exists() and cls_pipe is None:
            cls_pipe = joblib.load(cls_path)

    return reg_pipe, cls_pipe, feature_cols


# ── Sidebar ───────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/4/40/Flag_of_Somaliland.svg",
        width=220,
    )
    st.sidebar.title("🌍 Drought EWS Settings")

    district = st.sidebar.selectbox("Select District", DISTRICTS, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Date Range")
    start_year = st.sidebar.slider("Start Year", 1985, 2020, 2010)
    end_year   = st.sidebar.slider("End Year",   start_year + 1, 2023, 2023)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Drought Threshold")
    threshold = st.sidebar.slider(
        "SPI Drought Threshold", -2.5, -0.5, -1.0, step=0.1,
        help="SPI value below which conditions are classified as drought."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Sources**")
    st.sidebar.markdown(
        "- [CHIRPS](https://data.chc.ucsb.edu/products/CHIRPS-2.0/)  \n"
        "- [ERA5-Land](https://cds.climate.copernicus.eu/)  \n"
        "- [SWALIM](https://www.faoswalim.org/)  \n"
        "- [World Bank](https://data.worldbank.org/)  \n"
        "- [NOAA ONI](https://www.cpc.ncep.noaa.gov/)"
    )

    return district, start_year, end_year, threshold


# ── Main content ──────────────────────────────────────────────
def render_header():
    st.title("🌍 Somaliland Drought Early Warning System")
    st.markdown(
        "**3-month ahead district-level drought prediction** using CHIRPS, ERA5-Land, "
        "SWALIM, FAO ASIS, World Bank, and ENSO data."
    )
    st.markdown("---")


def render_kpi_cards(district: str, df: pd.DataFrame | None, reg_pipe, cls_pipe, feature_cols):
    """Render top KPI metric cards."""
    col1, col2, col3, col4 = st.columns(4)

    if df is not None and reg_pipe is not None and feature_cols is not None:
        d = df[df["district"] == district].sort_values("date").tail(1)
        if not d.empty:
            try:
                X = d[[c for c in feature_cols if c in d.columns]]
                # Pad missing columns with 0
                for c in feature_cols:
                    if c not in X.columns:
                        X[c] = 0.0
                X = X[feature_cols]

                pred_spi  = reg_pipe.predict(X)[0]
                pred_prob = cls_pipe.predict_proba(X)[0, 1] if cls_pipe else 0.5
                current_spi = d["spi_3"].values[0] if "spi_3" in d.columns else np.nan
                current_rain = d["rainfall_mm"].values[0] if "rainfall_mm" in d.columns else np.nan

                spi_label, spi_color = spi_to_category(pred_spi)
                risk_label, risk_color = drought_probability_label(pred_prob)

                with col1:
                    st.metric("📍 District", district)
                with col2:
                    st.metric("🌧️ Current SPI-3", f"{current_spi:.2f}" if not np.isnan(current_spi) else "N/A")
                with col3:
                    st.metric(
                        "🔭 Forecast SPI-3 (+3mo)",
                        f"{pred_spi:.2f}",
                        delta=f"{pred_spi - current_spi:+.2f}" if not np.isnan(current_spi) else None,
                    )
                with col4:
                    st.metric("⚠️ Drought Probability", f"{pred_prob*100:.0f}%")
                return
            except Exception:
                pass

    with col1: st.metric("📍 District", district)
    with col2: st.metric("🌧️ Current SPI-3", "N/A")
    with col3: st.metric("🔭 Forecast SPI-3 (+3mo)", "N/A")
    with col4: st.metric("⚠️ Drought Probability", "N/A")


def render_spi_timeseries(district: str, df: pd.DataFrame, start_year: int, end_year: int, threshold: float):
    """Interactive SPI-3 time series chart."""
    d = df[
        (df["district"] == district) &
        (df["date"].dt.year >= start_year) &
        (df["date"].dt.year <= end_year)
    ].sort_values("date")

    if d.empty:
        st.warning(f"No data available for {district} in selected range.")
        return

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("SPI-3 — Observed", "Monthly Rainfall (mm)"),
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
    )

    # SPI-3 line
    if "spi_3" in d.columns:
        spi = d["spi_3"]
        fig.add_trace(go.Scatter(
            x=d["date"], y=spi, name="SPI-3",
            line=dict(color="#2196F3", width=2),
        ), row=1, col=1)

        # Shade drought
        fig.add_shape(
            type="rect", x0=d["date"].min(), x1=d["date"].max(),
            y0=-4, y1=threshold,
            fillcolor="rgba(220,20,60,0.1)", line_width=0,
            row=1, col=1,
        )
        # Threshold line
        fig.add_hline(y=threshold, line_dash="dot", line_color="red",
                      annotation_text=f"Drought threshold ({threshold})", row=1, col=1)
        fig.add_hline(y=0, line_color="gray", line_width=0.5, row=1, col=1)

    # Rainfall bars
    if "rainfall_mm" in d.columns:
        fig.add_trace(go.Bar(
            x=d["date"], y=d["rainfall_mm"], name="Rainfall (mm)",
            marker_color="#2980B9", opacity=0.7,
        ), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"Climate Overview — {district}", font=dict(size=16)),
        height=520, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        showlegend=True,
    )
    fig.update_yaxes(title_text="SPI-3", row=1, col=1)
    fig.update_yaxes(title_text="Rainfall (mm)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def render_district_map(df: pd.DataFrame, threshold: float):
    """Interactive Plotly map of drought frequency by district."""
    if "spi_3" not in df.columns:
        return

    freq = (
        df.groupby("district")
        .apply(lambda g: (g["spi_3"] < threshold).mean() * 100)
        .reset_index(name="drought_pct")
    )
    freq["lon"] = freq["district"].map(lambda d: DISTRICT_COORDS.get(d, (45, 9.5))[0])
    freq["lat"] = freq["district"].map(lambda d: DISTRICT_COORDS.get(d, (45, 9.5))[1])

    fig = go.Figure(go.Scattermapbox(
        lat=freq["lat"], lon=freq["lon"],
        mode="markers+text",
        marker=dict(
            size=freq["drought_pct"] / 2 + 10,
            color=freq["drought_pct"],
            colorscale="RdYlGn_r",
            colorbar=dict(title="Drought %"),
            opacity=0.8,
        ),
        text=freq["district"],
        textposition="top center",
        customdata=freq["drought_pct"].round(1),
        hovertemplate="<b>%{text}</b><br>Drought frequency: %{customdata}%<extra></extra>",
    ))

    fig.update_layout(
        mapbox=dict(style="open-street-map", zoom=5.5, center=dict(lat=9.8, lon=45.5)),
        margin=dict(l=0, r=0, t=30, b=0),
        height=420,
        title="Drought Frequency by District (% months with SPI < threshold)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_download_section(district: str, df: pd.DataFrame, reg_pipe, cls_pipe, feature_cols):
    """Render forecast download section."""
    st.markdown("### 📥 Download Forecast")

    if df is None or reg_pipe is None or feature_cols is None:
        st.info("Load trained models and processed data to enable forecasts.")
        return

    d = df[df["district"] == district].sort_values("date").tail(24)
    if d.empty:
        return

    try:
        X = d.copy()
        for c in feature_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feature_cols]

        pred_spi  = reg_pipe.predict(X)
        pred_prob = cls_pipe.predict_proba(X)[:, 1] if cls_pipe else np.full(len(X), 0.5)
        pred_cls  = (pred_prob >= 0.5).astype(int)

        out = d[["date", "district"]].copy()
        out["predicted_spi3"]         = pred_spi.round(3)
        out["drought_probability_pct"] = (pred_prob * 100).round(1)
        out["drought_predicted"]       = pred_cls
        out["forecast_lead_months"]    = 3

        csv = out.to_csv(index=False)
        st.download_button(
            label="📊 Download Forecast CSV",
            data=csv,
            file_name=f"drought_forecast_{district}_{pd.Timestamp.now().date()}.csv",
            mime="text/csv",
        )
        st.dataframe(out.tail(12), use_container_width=True)

    except Exception as e:
        st.error(f"Forecast generation failed: {e}")


# ── Main app ──────────────────────────────────────────────────
def main():
    district, start_year, end_year, threshold = render_sidebar()
    render_header()

    # Load data and models
    df           = load_processed_data()
    reg_pipe, cls_pipe, feature_cols = load_models()

    if df is None:
        st.warning(
            "⚠️ No processed data found. Please run the notebook first to generate "
            "`data/processed/features_full.csv`."
        )
        st.info(
            "The app will display with placeholder content. "
            "Run `notebooks/drought_prediction.ipynb` to generate real data."
        )

    if reg_pipe is None:
        st.warning(
            "⚠️ No trained models found in `models/`. "
            "Run the notebook to train and save models."
        )

    # KPI cards
    render_kpi_cards(district, df, reg_pipe, cls_pipe, feature_cols)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Time Series", "🗺️ District Map", "📥 Forecast", "ℹ️ About"
    ])

    with tab1:
        if df is not None:
            render_spi_timeseries(district, df, start_year, end_year, threshold)
        else:
            st.info("Run the notebook to generate data for visualization.")

    with tab2:
        if df is not None:
            render_district_map(df, threshold)
        else:
            st.info("Run the notebook to generate data for the map.")

    with tab3:
        render_download_section(district, df, reg_pipe, cls_pipe, feature_cols)

    with tab4:
        st.markdown("""
        ## About This Dashboard

        **Somaliland Drought Early Warning System** provides 3-month ahead
        drought forecasts at district level using machine learning.

        ### Model
        - **Regression:** Predicts SPI-3 value 3 months ahead
        - **Classification:** Predicts probability of drought (SPI-3 < −1)
        - **Algorithm:** LightGBM / XGBoost / Random Forest (ensemble pipeline)

        ### SPI Classification
        | SPI Value | Category |
        |-----------|----------|
        | ≥ +2.0 | Extremely Wet |
        | +1.0 to +1.99 | Moderately Wet |
        | −0.99 to +0.99 | Near Normal |
        | −1.0 to −1.49 | Moderate Drought |
        | −1.5 to −1.99 | Severe Drought |
        | ≤ −2.0 | Extreme Drought |

        ### Data Sources
        - CHIRPS v2 (UCSB) — Rainfall
        - ERA5-Land (ECMWF) — Temperature, Soil Moisture, PET
        - SWALIM (FAO) — Station Data, ASIS
        - World Bank Open Data — Socioeconomic Indicators
        - NOAA CPC — ENSO/ONI Index

        ### Somaliland Seasons
        - **Gu (Apr–Jun):** Long rains — primary agricultural season
        - **Deyr (Oct–Nov):** Short rains
        - **Hagaa (Jul–Sep):** Dry season
        - **Jilaal (Dec–Mar):** Cold dry season
        """)


if __name__ == "__main__":
    main()
