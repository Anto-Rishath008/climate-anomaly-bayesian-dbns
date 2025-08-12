# Climate Anomaly Forecasting using Bayesian Deep Belief Networks

## Overview
This project forecasts climate anomalies using **ERA5 Reanalysis Climate Data** and **Bayesian Deep Belief Networks (DBNs)**.  
Our goal is to capture uncertainties in climate predictions and provide a robust framework for anomaly detection and forecasting.

---

## Features
- **Data Source:** ERA5 Reanalysis (2m Air Temperature and related variables)
- **Preprocessing:** Xarray & cfgrib for GRIB/NetCDF handling
- **Model:** Bayesian Deep Belief Network implemented in PyTorch
- **Uncertainty Quantification:** Predictive distributions rather than point estimates
- **Visualization:** Interactive maps & temporal anomaly plots
- **Web App:** Flask/Streamlit app to allow user-friendly access to forecasts

---

## Repository Structure
```
.
├── explore_era5.py           # Explore and visualize ERA5 dataset
├── test_download_era5.py     # Script to download ERA5 sample data from CDS API
├── era5_test_day.nc          # Sample ERA5 dataset file
├── requirements.txt          # Dependencies
├── .gitignore                # Ignored files and folders
├── README.md                 # This file
└── B19_PR_Report.pdf         # Project proposal report
```

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Anto-Rishath008/climate-anomaly-bayesian-dbns.git
cd climate-anomaly-bayesian-dbns
```

2. **Create a Python environment**
```bash
python -m venv climate_env
source climate_env/bin/activate    # Linux/Mac
climate_env\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up Copernicus API credentials**
Create a file named `.cdsapirc` in your home directory:
```
url: https://cds.climate.copernicus.eu/api
key: <your_uid>:<your_api_key>
```

---

## Usage

### Download Sample Data
```bash
python test_download_era5.py
```

### Explore Data
```bash
python explore_era5.py
```

---

## Project Workflow

**Week 1:** ERA5 data download & exploration  
**Week 2:** Data preprocessing & anomaly labeling  
**Week 3:** Bayesian DBN model building & training  
**Week 4:** Model evaluation, uncertainty analysis, and web app deployment  

---

## References
- [ERA5 Reanalysis Data](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels)
- [Bayesian Deep Belief Networks - Original Paper](https://www.cs.toronto.edu/~hinton/absps/dbn.pdf)
- [Copernicus Climate Data Store API](https://cds.climate.copernicus.eu/api-how-to)
