"""
src/data_loader.py
==================
Data ingestion module for the Somaliland Drought Prediction project.

Sources:
    - CHIRPS v2 monthly rainfall (UCSB)
    - ERA5-Land reanalysis (CDS API)
    - FAO SWALIM station data + ASIS index
    - World Bank Open Data (wbgapi)
    - NOAA CPC ONI/ENSO index
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED = 42

DISTRICTS = [
    "Hargeisa", "Berbera", "Borama", "Burao", "Erigavo",
    "Las_Anod", "Gabiley", "Sheikh", "Odweyne", "Zeila",
]

DISTRICT_COORDS = {
    "Hargeisa": (44.065, 9.560),  "Berbera": (45.014, 10.439),
    "Borama":   (43.183, 9.935),  "Burao":   (45.533,  9.517),
    "Erigavo":  (47.367, 10.617), "Las_Anod":(47.367,  8.483),
    "Gabiley":  (43.467, 9.983),  "Sheikh":  (45.200,  9.933),
    "Odweyne":  (45.067, 9.417),  "Zeila":   (43.483, 11.350),
}

SOMALILAND_BBOX = dict(min_lon=42.5, max_lon=49.5, min_lat=8.0, max_lat=11.5)

WB_INDICATORS = {
    "AG.AGR.TRVL.ZS": "agr_value_added_pct",
    "SP.RUR.TOTL.ZS":  "rural_pop_pct",
    "NY.GDP.PCAP.CD":  "gdp_per_capita",
    "AG.LND.ARBL.ZS":  "arable_land_pct",
}

ONI_SEASON_MAP = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}

# ── Utilities ────────────────────────────────────────────────────────────────

def safe_get(url: str, timeout: int = 20, **kwargs) -> requests.Response | None:
    """HTTP GET with graceful error handling."""
    try:
        r = requests.get(url, timeout=timeout, **kwargs)
        r.raise_for_status()
        return r
    except Exception as e:
        logger.warning(f"HTTP GET failed [{url}]: {e}")
        return None


def safe_download(url: str, dest: Path, timeout: int = 30) -> bool:
    """Stream-download a file to disk. Returns True on success."""
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Downloaded: {dest.name}")
        return True
    except Exception as e:
        logger.warning(f"Download failed [{url}]: {e}")
        return False


# ── CHIRPS ────────────────────────────────────────────────────────────────────

def download_chirps_monthly(year: int, month: int, dest_dir: Path) -> Path | None:
    """
    Download a single CHIRPS v2 monthly GeoTIFF from UCSB.

    URL pattern:
        https://data.chc.ucsb.edu/products/CHIRPS-2.0/
        global_monthly/tifs/chirps-v2.0.{YEAR}.{MM}.tif.gz

    After download, use extract_chirps_district_means() to derive
    per-district mean rainfall via rasterstats.
    """
    fn = f"chirps-v2.0.{year}.{month:02d}.tif.gz"
    url = (
        "https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
        f"global_monthly/tifs/{fn}"
    )
    dest = dest_dir / fn
    if dest.exists():
        return dest
    return dest if safe_download(url, dest) else None


def extract_chirps_district_means(
    tif_gz_path: Path,
    district_gdf,
) -> dict:
    """
    Compute mean CHIRPS rainfall for each district polygon.

    Requires:
        pip install rasterstats

    Parameters
    ----------
    tif_gz_path : Path  — path to .tif.gz file
    district_gdf : GeoDataFrame — with 'district' column

    Returns
    -------
    dict : {district_name: mean_rainfall_mm}
    """
    try:
        from rasterstats import zonal_stats
        import gzip, shutil

        tif_path = tif_gz_path.with_suffix("")
        if not tif_path.exists():
            with gzip.open(tif_gz_path, "rb") as f_in, open(tif_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        results = zonal_stats(district_gdf, str(tif_path), stats=["mean"])
        return {
            row.district: stat["mean"]
            for row, stat in zip(district_gdf.itertuples(), results)
        }
    except Exception as e:
        logger.warning(f"Zonal stats failed: {e}")
        return {}


def generate_chirps_synthetic(
    districts: list,
    date_range: pd.DatetimeIndex,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Synthetic CHIRPS-like monthly rainfall with correct bimodal
    Somaliland climatology:
        Gu   (Apr-Jun): Long rains
        Deyr (Oct-Nov): Short rains
        Dry  (Jul-Sep, Dec-Mar): Near-zero
    """
    rng = np.random.default_rng(seed)
    # Monthly climatology (mm) — Hargeisa baseline
    #           J   F   M   A   M   J   J   A   S   O   N   D
    clim = np.array([5, 8, 18, 55, 90, 60, 12, 8, 5, 45, 60, 10], dtype=float)
    district_mult = dict(
        Hargeisa=1.0, Berbera=0.4, Borama=1.1, Burao=1.2,
        Erigavo=1.5, Las_Anod=1.3, Gabiley=1.1,
        Sheikh=1.2, Odweyne=1.1, Zeila=0.6,
    )
    rows = []
    for d in districts:
        m = district_mult.get(d, 1.0)
        for dt in date_range:
            mean_r = clim[dt.month - 1] * m
            rain = max(0, rng.gamma(1.5, mean_r / 1.5) - 0.04 * (dt.year - 1981))
            rows.append({"date": dt, "district": d, "rainfall_mm": round(rain, 2)})
    logger.info(f"Synthetic CHIRPS: {len(rows)} records")
    return pd.DataFrame(rows)


def load_chirps(
    districts: list,
    date_range: pd.DatetimeIndex,
    raw_dir: Path = None,
    district_gdf=None,
) -> pd.DataFrame:
    """
    Load CHIRPS rainfall. Attempts raster download; falls back to synthetic.

    Production usage:
        Provide a `district_gdf` (GeoDataFrame with district polygons) and
        `raw_dir` for GeoTIFF caching. The function will download monthly
        rasters and extract district-level means via zonal statistics.
    """
    if district_gdf is not None and raw_dir is not None:
        records = []
        for dt in date_range:
            tif_path = download_chirps_monthly(dt.year, dt.month, raw_dir)
            if tif_path:
                means = extract_chirps_district_means(tif_path, district_gdf)
                for d, val in means.items():
                    records.append({"date": dt, "district": d, "rainfall_mm": val})
        if records:
            logger.info("CHIRPS loaded from rasters.")
            return pd.DataFrame(records)

    logger.info("Using synthetic CHIRPS data.")
    return generate_chirps_synthetic(districts, date_range)


# ── ERA5-Land ─────────────────────────────────────────────────────────────────

ERA5_CDS_REQUEST = {
    "product_type": "monthly_averaged_reanalysis",
    "variable": [
        "2m_temperature",
        "volumetric_soil_water_layer_1",
        "potential_evaporation",
    ],
    "year": [str(y) for y in range(1981, 2024)],
    "month": [f"{m:02d}" for m in range(1, 13)],
    "time": "00:00",
    "area": [
        SOMALILAND_BBOX["max_lat"],
        SOMALILAND_BBOX["min_lon"],
        SOMALILAND_BBOX["min_lat"],
        SOMALILAND_BBOX["max_lon"],
    ],
    "format": "netcdf",
}


def download_era5(dest_path: Path) -> bool:
    """
    Download ERA5-Land monthly data via CDS API.

    Prerequisites:
        1. pip install cdsapi
        2. Create ~/.cdsapirc:
               url: https://cds.climate.copernicus.eu/api/v2
               key: <UID>:<API_KEY>
    """
    try:
        import cdsapi
        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-land-monthly-means",
            ERA5_CDS_REQUEST,
            str(dest_path),
        )
        logger.info(f"ERA5 downloaded to {dest_path}")
        return True
    except Exception as e:
        logger.warning(f"CDS API failed: {e}")
        return False


def extract_era5_districts(nc_path: Path, districts: list, coords: dict) -> pd.DataFrame:
    """
    Extract district-level monthly ERA5 values from NetCDF.
    Uses nearest-neighbour lookup to district centroids.
    """
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path)
        records = []
        for d, (lon, lat) in coords.items():
            if d not in districts:
                continue
            t_series  = ds["t2m"].sel(longitude=lon, latitude=lat, method="nearest") - 273.15
            sm_series = ds["swvl1"].sel(longitude=lon, latitude=lat, method="nearest")
            pe_series = ds["pev"].sel(longitude=lon, latitude=lat, method="nearest") * -1000  # m→mm
            for i, dt in enumerate(pd.to_datetime(ds.time.values)):
                records.append({
                    "date": dt,
                    "district": d,
                    "temp_2m_c":    float(t_series.values[i]),
                    "soil_moisture": float(sm_series.values[i]),
                    "pet_mm":        float(pe_series.values[i]),
                })
        ds.close()
        return pd.DataFrame(records)
    except Exception as e:
        logger.warning(f"ERA5 extraction failed: {e}")
        return pd.DataFrame()


def generate_era5_synthetic(
    districts: list,
    date_range: pd.DatetimeIndex,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Synthetic ERA5-like monthly climate data for Somaliland districts."""
    rng = np.random.default_rng(seed + 1)
    #           J    F    M    A    M    J    J    A    S    O    N    D
    t_clim  = np.array([23, 24, 26, 27, 28, 28, 25, 25, 26, 26, 25, 23], dtype=float)
    sm_clim = np.array([.12,.11,.12,.16,.22,.20,.16,.14,.13,.17,.20,.13], dtype=float)
    pt_clim = np.array([110,115,130,135,140,135,120,120,125,120,110,105], dtype=float)
    t_off   = dict(Hargeisa=0,Berbera=6,Borama=-1,Burao=1,Erigavo=-2,
                   Las_Anod=3,Gabiley=-1,Sheikh=1,Odweyne=1,Zeila=7)
    rows = []
    for d in districts:
        for dt in date_range:
            m, yr = dt.month - 1, dt.year - 1981
            rows.append({
                "date": dt, "district": d,
                "temp_2m_c":    round(t_clim[m] + t_off.get(d, 0) + 0.02*yr + rng.normal(0, .8), 2),
                "soil_moisture": round(max(0.05, sm_clim[m] + rng.normal(0, .02)), 4),
                "pet_mm":        round(max(0, pt_clim[m] + 0.3*yr + rng.normal(0, 8)), 2),
            })
    logger.info(f"Synthetic ERA5: {len(rows)} records")
    return pd.DataFrame(rows)


def load_era5(
    districts: list,
    date_range: pd.DatetimeIndex,
    raw_dir: Path = None,
    coords: dict = None,
) -> pd.DataFrame:
    """Load ERA5-Land. Attempts CDS download; falls back to synthetic."""
    if raw_dir is not None:
        nc_path = raw_dir / "era5_somaliland.nc"
        if not nc_path.exists():
            download_era5(nc_path)
        if nc_path.exists():
            df = extract_era5_districts(nc_path, districts, coords or DISTRICT_COORDS)
            if not df.empty:
                return df
    return generate_era5_synthetic(districts, date_range)


# ── SWALIM / FAO ASIS ─────────────────────────────────────────────────────────

def generate_swalim_synthetic(
    districts: list,
    date_range: pd.DatetimeIndex,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Synthetic ASIS (Agricultural Stress Index) data.
    ASIS range: 0–100 (higher = more crop stress).
    """
    rng = np.random.default_rng(seed + 3)
    #              J   F   M   A   M   J   J   A   S   O   N   D
    asis_clim = np.array([70, 65, 55, 30, 20, 25, 50, 55, 60, 30, 25, 65], dtype=float)
    rows = []
    for d in districts:
        for dt in date_range:
            rows.append({
                "date": dt, "district": d,
                "asis_index": round(np.clip(asis_clim[dt.month-1] + rng.normal(0, 12), 0, 100), 1),
            })
    return pd.DataFrame(rows)


def load_swalim(
    districts: list,
    date_range: pd.DatetimeIndex,
    raw_dir: Path = None,
) -> pd.DataFrame:
    """
    Load SWALIM station data + FAO ASIS index.

    Production:
        1. Download from https://www.faoswalim.org/sections/water/data (account required)
        2. Download ASIS from https://www.fao.org/giews/earthobservation/
        3. Save as data/raw/swalim_asis.csv with columns:
               date, district, asis_index [, swalim_rainfall_mm]
    """
    if raw_dir is not None:
        csv_path = raw_dir / "swalim_asis.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["date"])
            logger.info(f"SWALIM loaded from {csv_path}")
            return df
    logger.warning("SWALIM CSV not found — using synthetic ASIS data.")
    return generate_swalim_synthetic(districts, date_range)


# ── World Bank ────────────────────────────────────────────────────────────────

def generate_wb_synthetic(
    date_range: pd.DatetimeIndex,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Synthetic World Bank annual indicators for Somalia/Somaliland."""
    rng = np.random.default_rng(seed + 2)
    years = sorted(set(date_range.year))
    rows = []
    for i, y in enumerate(years):
        rows.append({
            "year": y,
            "agr_value_added_pct": 60 + rng.normal(0, 3) - 0.1 * i,
            "rural_pop_pct":       72 + rng.normal(0, 1) - 0.15 * i,
            "gdp_per_capita":     120 + 3 * i + rng.normal(0, 10),
            "arable_land_pct":    1.8 + rng.normal(0, 0.1),
        })
    return pd.DataFrame(rows).set_index("year")


def load_world_bank(
    date_range: pd.DatetimeIndex,
    country: str = "SO",
) -> pd.DataFrame:
    """Fetch World Bank indicators via wbgapi. Falls back to synthetic."""
    try:
        import wbgapi as wb
        frames = []
        for code, name in WB_INDICATORS.items():
            df = wb.data.DataFrame(code, country, time=range(1981, 2024)).T
            df.index = pd.to_datetime(df.index.str.replace("YR", ""), format="%Y")
            df.columns = [name]
            frames.append(df)
        result = pd.concat(frames, axis=1)
        result.index.name = "year"
        logger.info(f"World Bank data fetched: {result.shape}")
        return result
    except Exception as e:
        logger.warning(f"World Bank API failed: {e} — using synthetic.")
        return generate_wb_synthetic(date_range)


# ── ENSO / ONI ────────────────────────────────────────────────────────────────

def generate_oni_synthetic(
    date_range: pd.DatetimeIndex,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Synthetic ENSO index with realistic ~4-year and ~7-year cycles."""
    rng = np.random.default_rng(seed + 4)
    t = np.arange(len(date_range))
    oni = (
        0.8 * np.sin(2 * np.pi * t / 48)
        + 0.4 * np.sin(2 * np.pi * t / 84)
        + rng.normal(0, 0.3, len(t))
    )
    return pd.DataFrame({"date": date_range, "oni": np.round(oni, 3)})


def load_oni(date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Download NOAA ONI (Oceanic Niño Index) from CPC.
    Falls back to synthetic ENSO.

    Climate context:
        ONI > +0.5 → El Niño: Deyr rains may increase
        ONI < -0.5 → La Niña: drought risk increases in Somaliland
    """
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    r = safe_get(url)
    if r is not None:
        try:
            rows = []
            for line in r.text.strip().split("\n")[1:]:
                p = line.split()
                if len(p) >= 3 and p[0] in ONI_SEASON_MAP:
                    rows.append({
                        "date": pd.Timestamp(year=int(p[1]), month=ONI_SEASON_MAP[p[0]], day=1),
                        "oni": float(p[2]),
                    })
            df = (
                pd.DataFrame(rows)
                .drop_duplicates("date")
                .set_index("date")
                .reindex(date_range)
                .interpolate()
                .reset_index()
            )
            df.columns = ["date", "oni"]
            logger.info("ONI fetched from NOAA CPC.")
            return df
        except Exception as e:
            logger.warning(f"ONI parse failed: {e}")
    return generate_oni_synthetic(date_range)


# ── Master loader ─────────────────────────────────────────────────────────────

def load_all_data(
    districts: list = None,
    date_range: pd.DatetimeIndex = None,
    raw_dir: Path = None,
    district_gdf=None,
) -> dict:
    """
    Load all data sources and return as a dictionary of DataFrames.

    Returns
    -------
    dict with keys: 'chirps', 'era5', 'swalim', 'oni', 'worldbank'
    """
    districts  = districts or DISTRICTS
    date_range = date_range or pd.date_range("1981-01", "2023-12", freq="MS")

    return {
        "chirps":    load_chirps(districts, date_range, raw_dir, district_gdf),
        "era5":      load_era5(districts, date_range, raw_dir),
        "swalim":    load_swalim(districts, date_range, raw_dir),
        "oni":       load_oni(date_range),
        "worldbank": load_world_bank(date_range),
    }
