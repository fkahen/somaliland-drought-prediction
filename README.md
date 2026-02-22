# Somaliland District-Level Drought Prediction

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **production-ready machine learning pipeline** for predicting drought 3 months ahead at district level across Somaliland, using multi-source climate and socioeconomic data.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Data Sources](#-data-sources)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Data Download Guide](#-data-download-guide)
- [How to Run](#-how-to-run)
- [Model Description](#-model-description)
- [Evaluation Results](#-evaluation-results)
- [Streamlit App](#-streamlit-app)
- [Docker Deployment](#-docker-deployment)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## Project Overview

Somaliland faces recurrent droughts that devastate pastoral and agricultural livelihoods. This project builds a **district-level early warning system** that predicts drought conditions **3 months ahead**, giving communities, NGOs, and government agencies time to mobilise resources.

### Targets

| Target | Type | Description |
|--------|------|-------------|
| `target_spi3` | Regression | SPI-3 value at *t* + 3 months |
| `target_drought` | Classification | 1 if SPI-3 < −1 at *t* + 3 months |

### Somaliland Climate Context

| Season | Months | Role |
|--------|--------|------|
| **Gu** | April – June | Long rains — primary agricultural season |
| **Deyr** | October – November | Short rains — secondary season |
| **Hagaa** | July – September | Dry season |
| **Jilaal** | December – March | Cold dry season |

Drought in this region is driven by ENSO variability, Indian Ocean Dipole anomalies, long-term warming trends, and land degradation.

---

## Data Sources

### 1. CHIRPS v2 — Rainfall
- **Provider:** Climate Hazards Group, UC Santa Barbara  
- **URL:** https://data.chc.ucsb.edu/products/CHIRPS-2.0/  
- **Resolution:** 0.05° / monthly  
- **Period:** 1981–present  
- **Variables used:** Monthly accumulated precipitation (mm)

### 2. ERA5-Land — Climate Reanalysis
- **Provider:** ECMWF / Copernicus Climate Change Service  
- **URL:** https://cds.climate.copernicus.eu/  
- **Resolution:** 0.1° / monthly  
- **Period:** 1950–present  
- **Variables used:** 2m temperature, volumetric soil water layer 1, potential evaporation

### 3. SWALIM — Ground Truth Station Data
- **Provider:** FAO Somalia Water and Land Information Management  
- **URL:** https://www.faoswalim.org/  
- **Variables used:** Rainfall station observations, drought reports  
- **Access:** Requires account registration at the SWALIM data portal

### 4. FAO ASIS — Agricultural Stress Index
- **Provider:** FAO GIEWS Earth Observation  
- **URL:** https://www.fao.org/giews/earthobservation/  
- **Variables used:** Agricultural Stress Index (0–100) per district

### 5. World Bank Open Data — Socioeconomic Indicators
- **Provider:** World Bank  
- **URL:** https://data.worldbank.org/  
- **API:** via `wbgapi` Python package  
- **Country:** Somalia (SO) — Somaliland not separately recognised  
- **Variables used:** Agriculture value added (% GDP), rural population %, GDP per capita, arable land %

### 6. NOAA CPC — ENSO/ONI Index
- **Provider:** NOAA Climate Prediction Center  
- **URL:** https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt  
- **Variables used:** Oceanic Niño Index (3-month running mean SST anomaly, Niño 3.4 region)

---

## Repository Structure

```
somaliland-drought-prediction/
│
├── data/
│   ├── raw/                    # Original downloaded data (gitignored)
│   └── processed/              # Engineered features, train/test splits
│
├── notebooks/
│   └── drought_prediction.ipynb  # Main end-to-end analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion from all sources
│   ├── feature_engineering.py  # SPI, anomalies, lags, rolling features
│   ├── modeling.py             # Model training, pipelines, serialization
│   ├── evaluation.py           # Metrics, visualizations, SHAP
│   └── utils.py                # Logging, seed, path helpers
│
├── models/                     # Saved .pkl pipelines (gitignored)
│
├── reports/
│   └── figures/                # Generated plots and charts
│
├── streamlit_app/
│   └── app.py                  # Interactive drought dashboard
│
├── Dockerfile                  # Container for reproducible environment
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

---

## Installation

### Option 1 — Local Python environment

```bash
# 1. Clone the repository
git clone https://github.com/your-org/somaliland-drought-prediction.git
cd somaliland-drought-prediction

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install project package (editable)
pip install -e .
```

### Option 2 — Docker (recommended for reproducibility)

```bash
docker build -t somaliland-drought .
docker run -p 8888:8888 -p 8501:8501 somaliland-drought
```

---

## Data Download Guide

### CHIRPS Rainfall

```bash
# Download a single monthly file (example: January 1981)
wget https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/chirps-v2.0.1981.01.tif.gz \
     -P data/raw/

# Or use the bulk download script (generates commands for 1981–2023)
python -c "
years = range(1981, 2024)
months = range(1, 13)
for y in years:
    for m in months:
        fn = f'chirps-v2.0.{y}.{m:02d}.tif.gz'
        print(f'wget https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/{fn} -P data/raw/')
" > download_chirps.sh
bash download_chirps.sh
```

### ERA5-Land via CDS API

```bash
# 1. Register at https://cds.climate.copernicus.eu/
# 2. Install cdsapi: pip install cdsapi
# 3. Create ~/.cdsapirc:
cat > ~/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
EOF

# 4. Download (handled automatically by src/data_loader.py when credentials are present)
python -c "from src.data_loader import download_era5; from pathlib import Path; download_era5(Path('data/raw/era5_somaliland.nc'))"
```

### SWALIM Data

1. Register at https://www.faoswalim.org/user/register
2. Navigate to **Data & Products → Rainfall Data**
3. Download monthly district-level data
4. Save as `data/raw/swalim_asis.csv` with columns: `date, district, asis_index`

### World Bank Data (automatic)

World Bank data is fetched automatically via the `wbgapi` Python package. No manual download required.

### ENSO / ONI Index (automatic)

ONI data is fetched automatically from NOAA CPC. No manual download required.

---

## How to Run

### Run the full notebook

```bash
cd notebooks
jupyter notebook drought_prediction.ipynb
```

### Run via Python scripts

```python
from src.data_loader import load_all_data
from src.feature_engineering import build_feature_matrix, get_feature_columns
from src.modeling import temporal_split, train_regression_models, train_classification_models, save_all_models
from src.evaluation import run_regression_evaluation, run_classification_evaluation
from src.utils import setup_logging, set_seed

# Setup
logger = setup_logging()
set_seed(42)

# Load data
data = load_all_data()

# Merge (see notebook for full merge logic)
# ... master_df = merge_all(...)

# Feature engineering
feature_df = build_feature_matrix(master_df)
feature_cols = get_feature_columns(feature_df)

# Split
split = temporal_split(feature_df, feature_cols)

# Train
reg_pipes = train_regression_models(split["X_train"], split["y_train_reg"])
cls_pipes = train_classification_models(split["X_train"], split["y_train_cls"])

# Evaluate
reg_results, reg_preds = run_regression_evaluation(reg_pipes, split["X_test"], split["y_test_reg"])
cls_results, cls_preds, cls_probs = run_classification_evaluation(cls_pipes, split["X_test"], split["y_test_cls"])

# Save
save_all_models(reg_pipes, cls_pipes, feature_cols, save_dir=Path("models/"))
```

### Launch Streamlit dashboard

```bash
streamlit run streamlit_app/app.py
```

---

## Model Description

### Architecture

All models are wrapped in a **scikit-learn Pipeline**:

```
Input features → SimpleImputer (median) → StandardScaler → Estimator → Output
```

### Feature Engineering

| Feature Group | Variables | Count |
|--------------|-----------|-------|
| SPI | SPI-1, SPI-3, SPI-6 | 3 |
| Climate anomalies | Rainfall, temperature, soil moisture, PET | 4 |
| Lag features | 1, 3, 6-month lags of key variables | ~60 |
| Rolling means | 3, 6, 12-month rolling averages | ~30 |
| Derived | Aridity index, water deficit, drought stress | 5 |
| ENSO interactions | ONI × season | 2 |
| Calendar | Month sin/cos, season dummies | 4 |
| Spatial | Longitude, latitude, district code | 3 |
| Socioeconomic | GDP, rural pop, agriculture | 4 |

### Models Compared

| Model | Hyperparameters |
|-------|----------------|
| Random Forest | n_estimators=200, max_depth=12, min_samples_leaf=5 |
| XGBoost | n_estimators=300, max_depth=6, lr=0.05, subsample=0.8 |
| LightGBM | n_estimators=300, max_depth=8, lr=0.05, subsample=0.8 |

### Train / Test Split

- **Train:** January 1985 – December 2015  
- **Test:** January 2016 – December 2023  
- **Note:** Random splits are explicitly avoided — time-series data requires strict chronological splitting to prevent future leakage.

---

## Evaluation Results

*Results below are from synthetic data (real CHIRPS/ERA5 will differ).*

### Regression — SPI-3 Prediction

| Model | RMSE ↓ | R² ↑ | MAE ↓ |
|-------|--------|-------|-------|
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| LightGBM | — | — | — |

### Classification — Drought Binary

| Model | AUC ↑ | F1 ↑ | Precision | Recall |
|-------|--------|-------|-----------|--------|
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |
| LightGBM | — | — | — | — |

*Run the notebook with real data to populate these tables.*

---

## Streamlit App

The interactive dashboard allows users to:
- Select district and date range
- View 3-month drought probability forecast
- Explore historical SPI time series
- Download predictions as CSV

```bash
streamlit run streamlit_app/app.py
# Open: http://localhost:8501
```

---

## Docker Deployment

```bash
# Build image
docker build -t somaliland-drought:latest .

# Run Jupyter notebook server
docker run -p 8888:8888 -v $(pwd)/data:/app/data somaliland-drought

# Run Streamlit app
docker run -p 8501:8501 somaliland-drought streamlit run streamlit_app/app.py
```

---

## Future Improvements

- [ ] **Real data integration** — connect CHIRPS raster download pipeline with `rasterstats` for true district-level zonal statistics
- [ ] **Hyperparameter optimisation** — implement Optuna Bayesian search with time-series cross-validation
- [ ] **Blocked time-series CV** — use `sklearn.model_selection.TimeSeriesSplit` for robust validation
- [ ] **Deep learning** — LSTM or Temporal Fusion Transformer for sequence modeling
- [ ] **NDVI features** — integrate MODIS/Sentinel-2 vegetation indices for rangeland condition
- [ ] **Indian Ocean Dipole** — add IOD index as additional ENSO-like predictor
- [ ] **Ensemble** — build a stacking ensemble of RF, XGBoost, and LightGBM
- [ ] **Automated retraining** — monthly cron job + MLflow experiment tracking
- [ ] **Alert system** — email/SMS notifications when drought probability exceeds threshold
- [ ] **Uncertainty quantification** — prediction intervals via quantile regression or conformal prediction
- [ ] **Spatial downscaling** — sub-district predictions using kriging or spatial interpolation

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m 'Add my feature'`)
4. Push to branch (`git push origin feature/my-feature`)
5. Open a Pull Request

Please ensure code passes `black`, `isort`, and `flake8` checks before submitting.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **CHIRPS:** Funk, C., Peterson, P., Landsfeld, M. et al. (2015). The climate hazards infrared precipitation with stations — a new environmental record for monitoring extremes. *Scientific Data*, 2, 150066.
- **ERA5-Land:** Muñoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13, 4349–4383.
- **SWALIM:** FAO Somalia Water and Land Information Management — http://www.faoswalim.org/
- **World Bank:** World Bank Open Data — https://data.worldbank.org/
- **SPI methodology:** McKee, T.B., Doesken, N.J., Kleist, J. (1993). The relationship of drought frequency and duration to time scale. *8th Conference on Applied Climatology*.
