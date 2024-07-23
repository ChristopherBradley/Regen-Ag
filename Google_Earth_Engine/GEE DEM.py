import ee
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from IPython.display import Image
from rasterio.transform import from_origin
import requests
import zipfile
import os

ee.Authenticate()
ee.Initialize()

# Can conveniently generately these coordinates using geojson.io
roi = ee.Geometry.Polygon([
          [
            [
              149.08449525031176,
              -35.2570704477603
            ],
            [
              149.08449525031176,
              -35.28584003593624
            ],
            [
              149.1124336866019,
              -35.28584003593624
            ],
            [
              149.1124336866019,
              -35.2570704477603
            ],
            [
              149.08449525031176,
              -35.2570704477603
            ]
          ]
])
# From the DEM-H Data page, converted from javascript to python
dataset = ee.Image('AU/GA/DEM_1SEC/v10/DEM-H');
elevation = dataset.select('elevation');
Image(url=elevation.getThumbURL({
    'min': -10.0, 'max': 1300, 'dimensions': 512, 'region': roi,
    'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}))

asset_url = elevation.getDownloadURL({
    'scale': 30,
    'crs': 'EPSG:4326',
    'fileFormat': 'GeoTIFF',
    'region': roi})
print(link)

# +
# Download the zip file
response = requests.get(asset_url, stream=True)
zip_path = 'asset.zip'
with open(zip_path, 'wb') as file:
    for chunk in response.iter_content(chunk_size=128):
        file.write(chunk)

# Unzip the file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('asset')

# Read the TIFF file using rasterio
tif_files = [f for f in os.listdir('dem_asset') if f.endswith('.tif')]
tif_path = os.path.join('asset', tif_files[0])
with rasterio.open(tif_path) as src:
    dem_data = src.read(1)  # Read the first band into a NumPy array

# Display the NumPy array
plt.imshow(dem_data, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.title('DEM')
plt.show()

# Clean up
os.remove(zip_path)
for f in os.listdir('dem_asset'):
    os.remove(os.path.join('dem_asset', f))
os.rmdir('dem_asset')
# -


