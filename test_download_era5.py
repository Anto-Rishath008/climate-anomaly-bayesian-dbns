import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': ['2m_temperature'],
        'year': '2022',
        'month': '07',
        'day': '15',
        'time': [
            '00:00', '06:00', '12:00', '18:00'
        ],
        'area': [30, 70, 0, 100],  # [North, West, South, East] - India bounding box
    },
    'era5_test_day.nc'
)
