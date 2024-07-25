import pysheds
from pysheds.grid import Grid
import matplotlib.pyplot as plt
from affine import Affine
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

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

# Visualise the accumulation (so cool!)
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=0)
im = ax.imshow(acc, extent=grid.extent, zorder=2,
               cmap='cubehelix',
               norm=colors.LogNorm(1, acc.max()),
               interpolation='bilinear')
plt.colorbar(im, ax=ax, label='Upstream Cells')
plt.title('Flow Accumulation', size=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()

# Find the coordinate with maximum accumulation
max_index = np.argmax(acc)
max_coords = np.unravel_index(max_index, acc.shape)
x, y = grid.affine * (max_coords[1], max_coords[0])

# # Delineate the largest catchment
x_snap, y_snap = grid.snap_to_mask(acc > 1000, (x, y))
catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, 
                       xytype='coordinate')

# Find the second largest catchment
acc2 = acc.copy()
acc2[catch] = 0
max_index = np.argmax(acc2)
max_coords = np.unravel_index(max_index, acc.shape)
x, y = grid.affine * (max_coords[1], max_coords[0])

catch2 = grid.catchment(x=x, y=y, fdir=fdir, dirmap=dirmap, 
                       xytype='coordinate')

catch2

plt.imshow(catch)

plt.imshow(catch2)

# +
result = np.zeros(catch.shape, dtype=int)

# Assign values based on conditions
result[catch & ~catch2] = 1  # True in catch only
result[~catch & catch2] = 2  # True in catch2 only
result[catch & catch2] = 3   # True in both catch and catch2
# -

plt.imshow(result)

# Extract river network
# ---------------------
branches = grid.extract_river_network(fdir, acc > 1000, dirmap=dirmap)

# +
# Visualise the extracted network
sns.set_palette('husl')
fig, ax = plt.subplots(figsize=(8.5,6.5))

plt.xlim(grid.bbox[0], grid.bbox[2])
plt.ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect('equal')

for branch in branches['features']:
    line = np.asarray(branch['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])
    
_ = plt.title('D8 channels', size=14)
# +
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Example array
arr = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

# Compute the gradient of the array using the Sobel operator
sobel_x = ndimage.sobel(arr, axis=0)  # horizontal gradient
sobel_y = ndimage.sobel(arr, axis=1)  # vertical gradient
edges = np.hypot(sobel_x, sobel_y)    # magnitude of the gradient

# Binarize the edges
edges = edges > 0

# Plot the original array and the edges
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Array")
plt.imshow(arr, cmap='gray', interpolation='nearest')

plt.subplot(1, 2, 2)
plt.title("Edges")
plt.imshow(edges, cmap='gray', interpolation='nearest')

plt.show()

# -



