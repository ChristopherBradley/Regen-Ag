from pysheds.grid import Grid
import matplotlib.pyplot as plt
from affine import Affine
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from datetime import datetime

# Load both the dem (basically a numpy array), and the grid (all the metadata like the extent)
grid = Grid.from_raster('test.tif')
dem = grid.read_raster('test.tif')

# Hydrologically enforce the DEM so water can flow downhill to the edge and not get stuck
pit_filled_dem = grid.fill_pits(dem)
flooded_dem = grid.fill_depressions(pit_filled_dem)
inflated_dem = grid.resolve_flats(flooded_dem)

# Calculate the direction and accumulation of water
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
acc = grid.accumulation(fdir, dirmap=dirmap)

# Progressively delineate each catchment by the pixel with the highest accumulation
catchment_id = 1
all_catchments = np.zeros(acc.shape, dtype=int)
acc_updated = acc.copy()
while np.any(all_catchments == 0) and catchment_id <= 10:
    # Find the coordinate with maximum accumulation
    max_index = np.argmax(acc_updated)
    max_coords = np.unravel_index(max_index, acc_updated.shape)
    x, y = grid.affine * (max_coords[1], max_coords[0])

    # # Delineate the largest catchment
    catch = grid.catchment(x=x, y=y, fdir=fdir, dirmap=dirmap, 
                        xytype='coordinate')
    all_catchments[catch] = catchment_id
    acc_updated[catch] = 0
    catchment_id += 1
plt.imshow(all_catchments)
plt.show()

from scipy import ndimage
sobel_x = ndimage.sobel(all_catchments, axis=0)  # horizontal gradient
sobel_y = ndimage.sobel(all_catchments, axis=1)  # vertical gradient
edges = np.hypot(sobel_x, sobel_y)    # magnitude of the gradient


edges_bool = edges > 0

plt.imshow(edges_bool)

# Visualise the accumulation (so cool!)
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=0)
im = ax.imshow(edges, extent=grid.extent, zorder=2,
               cmap='cubehelix',
               norm=colors.LogNorm(1, acc.max()),
               interpolation='bilinear')
plt.colorbar(im, ax=ax, label='Upstream Cells')
plt.title('Flow Accumulation', size=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()




