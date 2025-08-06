import xarray as xr
import matplotlib.pyplot as plt

# Load the ERA5 dataset
ds = xr.open_dataset("era5_test_day.nc")  # or your actual filename

# Print dataset structure
print(ds)

# ✅ Corrected: Use 'valid_time' instead of 'time'
temp = ds['t2m'].isel(valid_time=0)

# Convert Kelvin to Celsius
temp_celsius = temp - 273.15

# Plot temperature
plt.figure(figsize=(10, 6))
temp_celsius.plot(cmap='coolwarm')
plt.title("2m Air Temperature (°C) - ERA5 Sample")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
