import xarray as xr
import matplotlib.pyplot as plt

url = 'https://dapds00.nci.org.au/thredds/dodsC/ub8/au/OzWALD/8day/Ssoil/OzWALD.Ssoil.2020.nc'
ds = xr.open_dataset(url)

north, south, west, east = -34.350050, -34.479314, 148.427637, 148.543866
subset = ds.sel(longitude=slice(west, east), latitude=slice(north, south))

soil_moisture = subset['Ssoil']
time = subset['time']
lat = subset['latitude']
lon = subset['longitude']

time_index = 0  
soil_moisture_at_time = soil_moisture.isel(time=time_index)

# Plot the 2D map of soil moisture at the selected timepoint
plt.figure(figsize=(10, 8))
plt.pcolormesh(lat, lon, soil_moisture_at_time, shading='nearest')
plt.colorbar(label='Soil Moisture')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Soil Moisture at Time Index {time_index}')
plt.show()

# Select a specific cell (e.g., at index [10, 10])
lat_index = 10
lon_index = 10
soil_moisture_at_cell = soil_moisture[:, lat_index, lon_index]

# Plot the time series of soil moisture at the selected cell
plt.figure(figsize=(10, 6))
plt.plot(time, soil_moisture_at_cell)
plt.xlabel('Time')
plt.ylabel('Soil Moisture')
plt.title(f'Soil Moisture Time Series at Cell ({lat_index}, {lon_index})')
plt.show()