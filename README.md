# Climate Anomaly Forecasting using Bayesian Deep Belief Networks

This project focuses on forecasting climate anomalies using Bayesian Deep Belief Networks (DBNs) and ERA5 Reanalysis data. It incorporates uncertainty quantification and interpretable probabilistic reasoning.

## 🔍 Project Description

We use the CDS API to download ERA5 2m air temperature data, visualize spatial patterns, and later model these anomalies using Bayesian DBNs. This project is structured weekly to reflect progress.

## 📁 Repository Structure

- `test_download_era5.py`: Script to download ERA5 temperature data using CDS API.
- `explore_era5.py`: Script to visualize 2m temperature data.
- `era5_test_day.nc`: Sample ERA5 NetCDF data.
- `README.md`: Project overview and progress documentation.
- `requirements.txt`: Python environment requirements.

## 🔄 Weekly Breakdown

- **Week 1**: Data access via CDS API, download and visualization ✅
- **Week 2**: Preprocessing and anomaly detection
- **Week 3**: Bayesian DBN model design & training
- **Week 4**: Evaluation and Uncertainty Quantification

## 🧠 Algorithms

- Bayesian Deep Belief Networks (using probabilistic inference)
- ERA5 anomaly detection (mean deviation, z-score, etc.)

## 📚 References

- ERA5 Dataset: https://cds.climate.copernicus.eu
- Bayesian Deep Learning: Yarin Gal (2016), etc.

---
Developed by Anto Rishath A | B.Tech CSE - AI | Amrita Vishwa Vidyapeetham
