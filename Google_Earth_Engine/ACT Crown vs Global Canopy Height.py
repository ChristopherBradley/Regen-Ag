import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from scipy.stats import gaussian_kde
from datetime import datetime

start = datetime.now()

# Read the GeoPackage. John has stored this on ANU NCI at: /g/data/xe2/John/Data/ACTtrees
gdf = gpd.read_file("ACTGOV_Mature_Trees_2020_sppID_singles.gpkg")

# Open the Global Canopy Height TIFF file. I downloaded this with: aws s3 cp --no-sign-request s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm/311230213.tif .
with rasterio.open("Canberra tiles/311230302.tif") as src:
    bbox = box(*src.bounds)
    canopy_height_data = src.read(1)  # Read the first band
    canopy_height_meta = src.meta
canopy_height_meta

# Clip the ACT Tree Crowns to the area within this Global Canopy Height bbox
raster_crs = src.meta['crs']
gdf = gdf.to_crs(raster_crs)
gdf_clipped = gdf[gdf.intersects(bbox)]
print(f"Number of tree crowns within raster bounds: {len(gdf_clipped)}")

# Using a subset of 20k trees to speed things up
gdf_clipped = gdf_clipped[:20000]

def get_max_height(polygon, src):
    """Find the maximum height within the polygon"""
    out_image, _ = mask(src, [polygon], crop=True)
    data = out_image[0]
    max_height = np.nanmax(data[data != src.nodata])
    return max_height

# Find the maximum global canopy height within each ACT Tree Crown
results = []
with rasterio.open("Canberra tiles/311230302.tif") as src:
    for index, row in gdf_clipped.iterrows():
        polygon = row['geometry']
        max_canopy_height = get_max_height(polygon, src)
        results.append({
            'id': row['crownID'],
            'max_height_geopackage': row['max_height'],
            'max_height_tiff': max_canopy_height
        })
results_df = pd.DataFrame(results)

# Density plot to compare the two datasets
nbins = 300
x = results_df['max_height_geopackage']
y = results_df['max_height_tiff']

k = gaussian_kde([x,y])
xi, yi = np.mgrid[
   x.min():x.max():nbins*1j,
   y.min():y.max():nbins*1j
]
zi = k(np.vstack([
   xi.flatten(),
   yi.flatten()
])).reshape(xi.shape)

fig, ax = plt.subplots(figsize=(8,8))
ax.pcolormesh(xi, yi, zi)
ax.set_xlabel('ACT Tree Crowns', fontsize=12)
ax.set_ylabel('Global Canopy Height', fontsize=12)
plt.show()

end = datetime.now()
print("Time taken: ", end-start)
