# RCM–Sentinel-1 SAR Dataset Construction Pipeline

## Overview

This repository provides a complete pipeline for constructing a multi-sensor SAR dataset by integrating **RADARSAT Constellation Mission (RCM)** data with **Sentinel-1 (S1)** scenes from the AI4Arctic dataset.

The pipeline includes:

* Querying and downloading RCM scenes from EODMS
* Matching RCM scenes with Sentinel-1 acquisitions
* Extracting and processing geolocation metadata
* Radiometric calibration (DN → σ⁰)
* Co-registration of RCM data to Sentinel-1 grid
* Integration into AI4Arctic NetCDF format

This dataset enables **multi-sensor fusion for sea ice mapping**, supporting tasks such as:

* Sea Ice Concentration (SIC) regression
* Stage of Development (SOD) classification
* Floe Size (FLOE) classification

---

## Repository Structure

```
rcm_sar_dataset/
│
├── rcm_shapefile_creator.py      # Generates shapefiles for RCM scenes
├── sentinel1_shapefile_creator.py# Generates shapefiles for S1 scenes
├── RCM_SAR_download.ipynb        # Interactive notebook for downloading RCM data
├── rcm_search_download.py        # Automated RCM querying via EODMS API
├── best_match_overlap.py         # Finds best RCM scene for each S1 scene
├── calibration_sigma.py          # Converts DN to sigma0 (dB)
├── coregisteration_RCM_S1.py     # Co-registers RCM to S1 grid
├── updating_AI4Arctic_RCM.py     # Adds RCM channels to AI4Arctic NetCDF
│
├── rcm_histograms.py             # Data distribution visualization
├── min_max_mean_std.py           # Dataset statistics
├── check_new_dataset.py          # Validation checks
├── globplot.py                   # Geographic visualization
│
├── match_rcm_scenes_summary.xlsx # Matching results summary
└── LICENSE
```

---

## Pipeline Workflow

### 0. Shapefile Generation

Create footprint polygons for spatial operations

---

### 1. RCM Scene Search & Download

* Uses `rcm_search_download.py`
* Queries EODMS API using:

  * Time window (±5 hours around S1 acquisition)
  * Spatial overlap (AOI shapefiles)
  * Filters: dual-pol (HH/HV), GRD products

---

### 2. Scene Matching (RCM ↔ S1)

* Computes and plot spatial overlap between RCM and Sentinel-1 scenes

Output:

* Excel summary of matched scenes
* Plot RCM and S1 polygons

---

### 3. Radiometric Calibration

Convert raw DN values to sigma naught (σ⁰):

* Uses LUT files (`lutSigma_HH.xml`, `lutSigma_HV.xml`)
* Applies interpolation across range dimension
* Converts to dB scale

---

### 4. Co-registration (RCM → S1 Grid)

Align RCM data to Sentinel-1 geometry:

* Interpolates S1 geolocation grid (lat/lon)
* Maps coordinates into RCM space
* Applies bilinear interpolation

Output:

* Co-registered RCM HH and HV aligned with S1 pixels

---

### 5. Dataset Integration (AI4Arctic Extension)

Add processed RCM channels into NetCDF files:

New variables added:

* `sar_RCM_HH_cor_cal`
* `sar_RCM_HV_cor_cal`
* `orig_sar_RCM_HH`
* `orig_sar_RCM_HV`
* `sar_grid_line_rcm`
* `sar_grid_sample_rcm`
* `sar_grid_latitude_rcm`
* `sar_grid_longitude_rcm`
* `mask_sentinel`

---

### 6. Validation & Analysis

Check dataset integrity:

```
python check_new_dataset.py
```

Compute statistics:

```
python min_max_mean_std.py
```

Visualize distributions:

```
python rcm_histograms.py
```

---

## Requirements

### Python Environment

Recommended: Python 3.10+

Install dependencies:

```bash
pip install numpy pandas xarray h5netcdf rasterio geopandas shapely matplotlib getpass xml.etree.ElementTree scipy pathlib re
```

for API access:

```bash
pip install eodms-api-client
```

---

## Data Sources

* Sentinel-1: AI4Arctic Dataset
* RCM: EODMS (Earth Observation Data Management System)

---

## Key Features

* Multi-sensor SAR dataset construction
* Automated RCM retrieval and filtering
* Accurate geolocation-based co-registration
* Radiometric calibration using official LUTs
* Seamless integration into AI4Arctic format

---

## Use Cases

* Multi-sensor fusion research
* Sea ice mapping (SIC, SOD, FLOE)
* Domain adaptation between SAR sensors
* Robustness studies (sensor dropout scenarios)

---

## Notes

* Ensure correct file paths for:

  * AI4Arctic dataset
  * RCM product directories
  * LUT calibration files

* Co-registration assumes:

  * Valid tie-point metadata in both S1 and RCM

---

## Author

Niloofar Azad
MASc Systems Design Engineering
University of Waterloo

---

## License

See `LICENSE` file for details.
