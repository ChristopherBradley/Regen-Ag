# +
import numpy as np
import pandas as pd
import seaborn as sns
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt

from rasterio.mask import mask
from shapely.geometry import box
from scipy.stats import gaussian_kde
# -


# Read the GeoPackage
gdf = gpd.read_file("ACTGOV_Mature_Trees_2020_sppID_singles.gpkg")
gdf_crs = gdf.crs

gdf.crs

# Open the TIFF file and get its bounding box
with rasterio.open("Canberra tiles/311230302.tif") as src:
    bbox = box(*src.bounds)
    canopy_height_data = src.read(1)  # Read the first band
    canopy_height_meta = src.meta
canopy_height_meta

raster_crs = src.meta['crs']
gdf_crs = gdf.crs

print(raster_crs)
print(gdf_crs)

print(type(raster_crs))
print(type(gdf_crs))

gdf = gdf.to_crs(raster_crs)
print("GeoPackage reprojected to match raster CRS.")

gdf_clipped = gdf[gdf.intersects(bbox)]
print(f"Number of tree crowns within raster bounds: {len(gdf_clipped)}")


# Function to get the maximum height within a polygon
def get_max_height(polygon, src):
    # Mask the raster with the polygon
    out_image, out_transform = mask(src, [polygon], crop=True)
    
    # Get the data array within the mask
    data = out_image[0]
    
    # Calculate the maximum value, ignoring nodata values
    max_height = np.nanmax(data[data != src.nodata])
    
    return max_height


len(gdf_clipped)

# Store results
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

nbins = 300

x = results_df['max_height_geopackage'] # change 'x' with your column name
y = results_df['max_height_tiff'] # change 'y' with your column name

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
plt.show()




# Extract the data
# x = results_df['max_height_geopackage']
# y = results_df['max_height_tiff']

# Calculate the point density
# xy = np.vstack([x, y])
# z = gaussian_kde(xy)(xy)

# # Sort the points by density, so that the densest points are plotted last
# idx = z.argsort()
# x, y, z = x[idx], y[idx], z[idx]

# # Create the scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, c=z, s=50, edgecolor='black', cmap='viridis')
# plt.colorbar(label='Density')
# plt.xlabel('Max Height GeoPackage')
# plt.ylabel('Max Height TIFF')
# plt.title('Scatter Plot of Max Heights Colored by Density')
# plt.grid(True)
# plt.show()

# sns.pairplot(results_df, vars=['max_height_geopackage', 'max_height_tiff'], plot_kws={'alpha': 0.3})
# plt.show()


# +
# # Define the height bins and labels
# bins = [0, 3, 5, 10, 15, 20, float('inf')]
# labels = ['0-3', '3-5', '5-10', '10-15', '15-20', '>20']

# # Create a new column for the height categories
# results_df['height_bin'] = pd.cut(results_df['max_height_geopackage'], bins=bins, labels=labels, right=False)

# # -

# plt.figure(figsize=(12, 8))
# sns.boxplot(x='height_bin', y='max_height_tiff', data=results_df)
# plt.xlabel('Max Height GeoPackage (bins)')
# plt.ylabel('Max Height TIFF')
# plt.title('Boxplot of Max Height TIFF for Different Height Ranges of GeoPackage')
# plt.grid(True)
# plt.show()

print()
