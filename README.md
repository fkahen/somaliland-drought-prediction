# Somaliland Drought Prediction

A machine learning pipeline for district-level drought forecasting three months ahead across Somaliland, designed to support humanitarian early warning systems.

---

## Background

The Horn of Africa experiences recurrent drought cycles that drive food insecurity, displacement, and economic loss. In Somaliland, where agriculture and pastoralism underpin rural livelihoods, seasonal drought can escalate rapidly into humanitarian crises. Effective early warning—weeks or months before conditions deteriorate—gives governments, NGOs, and aid organizations the lead time needed for pre-positioned response.

This project builds a reproducible forecasting system that estimates drought risk at the district level with a three-month horizon. It integrates satellite-derived rainfall estimates, climate reanalysis fields, global teleconnection indices, and socioeconomic covariates into an ensemble modeling framework.

---

## Problem Statement

Given observational and reanalysis data available at time *t*, predict whether a district will experience drought conditions at time *t + 3 months*, operationalized as:

- **Regression target**: SPI-3 value three months ahead (`target_spi3`)
- **Classification target**: binary drought indicator, positive when SPI-3 < −1 (`target_drought`)

The three-month lead time was chosen to align with operational planning cycles used by humanitarian logistics and food security assessments.

---

## Datasets

| Source | Variable | Coverage |
|---|---|---|
| CHIRPS v2.0 | Monthly precipitation | 1981–present, 0.05° |
| ERA5-Land (ECMWF) | Temperature, soil moisture, evapotranspiration | 1950–present, 0.1° |
| SWALIM | Ground station rainfall and hydrology | Somaliland network |
| NOAA | ENSO index (Niño 3.4 SST anomaly) | 1950–present |
| World Bank | GDP, rural population share | Country/district level |

CHIRPS and ERA5 are the primary drivers. SWALIM station data provides in-situ validation. ENSO and socioeconomic variables are fetched automatically by the pipeline at runtime. Download instructions for each source are provided below.

---

## Methodology

### Feature Engineering

Raw inputs are transformed into a feature matrix spanning several categories:

- **SPI indices**: standardized precipitation indices at 1, 3, and 6-month accumulation windows
- **Climate anomalies**: rainfall, temperature, and soil moisture departures from climatological means
- **Lag features**: 1, 3, and 6-month lags of key variables
- **Rolling statistics**: 3, 6, and 12-month moving averages
- **Derived indicators**: aridity index, drought stress proxy
- **Climate interaction terms**: ENSO × seasonal encoding
- **Calendar encoding**: month expressed as sine/cosine to capture seasonality
- **Spatial covariates**: district centroid coordinates
- **Socioeconomic context**: GDP per capita, rural population fraction

### Models

Three gradient-based and tree ensemble models are trained in parallel:

- **Random Forest** — bagged decision trees, robust to noisy features
- **XGBoost** — gradient boosting with regularization, strong baseline
- **LightGBM** — histogram-based boosting, efficient on larger feature sets

Each model is wrapped in a scikit-learn pipeline: `features → imputation → scaling → estimator → prediction`. This ensures consistent preprocessing and simplifies deployment.

### Train/Test Split

The pipeline uses strict chronological splitting to prevent data leakage:

| Split | Period |
|---|---|
| Training | 1985–2015 |
| Test | 2016–2023 |

Random cross-validation is not used because climate observations are serially correlated.

---

## Pipeline

The full workflow—data loading, feature construction, model training, evaluation, and artifact saving—runs with a single command:

```bash
python -m src.run_pipeline
```

This executes the following stages in sequence:

1. Load raw datasets (CHIRPS, ERA5, SWALIM, ENSO, World Bank)
2. Merge, align, and clean inputs at district-month resolution
3. Construct the feature matrix
4. Train regression and classification models
5. Evaluate on the held-out test period
6. Write trained model artifacts to `models/`

The pipeline is designed to be reproducible. Given the same input data, it will produce the same outputs.

---

## Installation

```bash
git clone https://github.com/fkahen/somaliland-drought-prediction.git
cd somaliland-drought-prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Data Download

**CHIRPS rainfall**

Single month example:
```bash
wget https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/chirps-v2.0.1981.01.tif.gz -P data/raw/
```

For bulk download:
```bash
bash download_chirps.sh
```

**ERA5-Land**

Register at the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/), install the API client, and configure `~/.cdsapirc` with your key:

```bash
pip install cdsapi
python -c "from src.data_loader import download_era5; from pathlib import Path; download_era5(Path('data/raw/era5_somaliland.nc'))"
```

**SWALIM**

Download manually from the [SWALIM portal](https://www.faoswalim.org/) and save as `data/raw/swalim_asis.csv`.

World Bank and ENSO index data are fetched automatically at runtime.

---

## Running the Code

**Full pipeline:**
```bash
python -m src.run_pipeline
```

**Interactive exploration:**
```bash
jupyter notebook notebooks/drought_prediction.ipynb
```

**Streamlit dashboard** (district-level forecast visualization):
```bash
streamlit run streamlit_app/app.py
```

The dashboard runs at `http://localhost:8501` and supports district and date range selection, three-month drought probability forecasts, historical SPI visualization, and prediction export.

---

## Project Structure

```
somaliland-drought-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
├── notebooks/
│   └── drought_prediction.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── utils.py
├── streamlit_app/
│   └── app.py
├── run_pipeline.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Future Directions

- **Vegetation signals**: incorporate NDVI from MODIS or Sentinel-2 as an additional drought proxy
- **Deep learning**: evaluate LSTM and Temporal Fusion Transformer architectures for sequence modeling
- **Hyperparameter search**: integrate Optuna for systematic tuning
- **Spatial resolution**: downscale forecasts to sub-district level using spatial disaggregation
- **Operational deployment**: automated monthly retraining and alert delivery via email/SMS

---

## Contributing

Contributions from climate scientists, ML practitioners, and humanitarian data specialists are welcome. To contribute:

1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Open a pull request

Please ensure code passes `black`, `isort`, and `flake8` before submitting.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

This project relies on open datasets from the [Climate Hazards Center](https://www.chc.ucsb.edu/), [ECMWF](https://www.ecmwf.int/), [FAO/SWALIM](https://www.faoswalim.org/), [World Bank](https://data.worldbank.org/), and [NOAA](https://www.noaa.gov/). Their commitment to open data access makes operational climate monitoring research possible.
