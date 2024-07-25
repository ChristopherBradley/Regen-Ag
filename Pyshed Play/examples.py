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

max_acc = np.max(acc)

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
# Delineate a catchment
# ---------------------

# Snap pour point to high accumulation cell
# x_snap, y_snap = grid.snap_to_mask(acc > 1000, (x, y))

# Delineate the catchment
# catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, 
#                        xytype='coordinate')

# Crop and plot the catchment
# ---------------------------
# Clip the bounding box to the catchment
# grid.clip_to(catch)
clipped_catch = grid.view(catch)
# -

grid_original = grid

# +
# Plot the catchment
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.grid('on', zorder=0)
im = ax.imshow(np.where(clipped_catch, clipped_catch, np.nan), extent=grid.extent,
               zorder=1, cmap='Greys_r')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Delineated Catchment', size=14)
# -

start = datetime.now()
branches = grid.extract_river_network(fdir, acc > max_acc/100, dirmap=dirmap)
end = datetime.now()
print(end - start)

# +
sns.set_palette('husl')
fig, ax = plt.subplots(figsize=(8.5,6.5))

plt.xlim(grid.bbox[0], grid.bbox[2])
plt.ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect('equal')

for branch in branches['features']:
    line = np.asarray(branch['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])

plt.scatter(16597867.038212, -4201352.527325, color='green', zorder=5)
plt.scatter(16599243.382298, -4201304.7376, color='red', zorder=5)


_ = plt.title('D8 channels', size=14)
# -

network1 = -1
max_len = -1
max_len_index = -1
for i, feature in enumerate(branches["features"]):
    line_coords = feature['geometry']['coordinates']
    if len(line_coords) > max_len:
        max_len = len(line_coords)
        max_len_index = i

max_len_index

max_len

branches["features"][105]["geometry"]["coordinates"][0]

branches["features"][105]["geometry"]["coordinates"][-1]

feature = branches[0]
line_coords = feature['geometry']['coordinates']

[16596256.524473, -4198819.671889] in line_coords

[x, y] in line_coords


