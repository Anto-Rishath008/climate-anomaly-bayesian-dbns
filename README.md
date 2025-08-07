# Climate Anomaly Forecasting using Bayesian Deep Belief Networks

This project focuses on forecasting climate anomalies using **Bayesian Deep Belief Networks (DBNs)** based on **ERA5 reanalysis climate data**.

---

## 📌 Project Structure

- `test_download_era5.py` – Script to download ERA5 2m air temperature data using the CDS API
- `explore_era5.py` – Script to explore, visualize, and preprocess the ERA5 dataset
- `era5_test_day.nc` – Sample downloaded climate dataset (NetCDF format)
- `train_data.npy`, `val_data.npy`, `test_data.npy` – Preprocessed anomaly datasets
- `README.md` – Project overview and instructions
- `requirements.txt` – Python dependencies

---

## 🛰️ Dataset Used

- **Source**: [ERA5 Reanalysis Data - Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
- **Variable**: 2m air temperature (`t2m`)
- **Spatial Resolution**: 0.25° x 0.25°
- **Temporal Frequency**: Hourly

---

## 🧪 Preprocessing Steps

1. Downloaded ERA5 `.nc` file using CDS API
2. Computed climatology across time steps
3. Derived anomalies: `anomaly = t2m - climatology`
4. Flattened the data to 2D structure
5. Normalized with `StandardScaler`
6. Split into train (70%), val (15%), test (15%) sets

---

## 🔮 Next Step (Week 3)

Implementation of **Bayesian Deep Belief Networks (DBNs)** for temporal forecasting and uncertainty quantification.

---

## ⚙️ Setup Instructions

```bash
git clone https://github.com/Anto-Rishath008/climate-anomaly-bayesian-dbns.git
cd climate-anomaly-bayesian-dbns
pip install -r requirements.txt
```

---

## 👨‍💻 Authors

- **Anto Rishath A** – 3rd Year B.Tech CSE (AI)  
  Amrita Vishwa Vidyapeetham

---