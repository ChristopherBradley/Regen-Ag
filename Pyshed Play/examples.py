import pysheds
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

max_coords

grid.affine

~grid.affine

# +
# Convert geographic coordinates back to array indices
col, row = ~grid.affine * (x, y)

# Convert to integer indices
row, col = int(round(row)), int(round(col))
# -

row, col

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

network1 = -1
max_len = -1
max_len_index = -1
branches_np = []
for i, feature in enumerate(branches["features"]):
    line_coords = feature['geometry']['coordinates']
    branch_np = []
    for coord in line_coords:
        col, row = ~grid.affine * (coord[0], coord[1])
        row, col = int(round(row)), int(round(col))
        branch_np.append([row,col])
    branches_np.append(branch_np)
len(branches_np)

coord1 = branches_np[0][0]
coord2 = branches_np[0][-1]
x1, y1 = grid.affine * (coord1[1], coord1[0])
x2, y2 = grid.affine * (coord2[1], coord2[0])

# +
sns.set_palette('husl')
fig, ax = plt.subplots(figsize=(8.5,6.5))

plt.xlim(grid.bbox[0], grid.bbox[2])
plt.ylim(grid.bbox[1], grid.bbox[3])
ax.set_aspect('equal')

for branch in branches['features']:
    line = np.asarray(branch['geometry']['coordinates'])
    plt.plot(line[:, 0], line[:, 1])

plt.scatter(x1, y1, color='green', zorder=5)
plt.scatter(x2, y2, color='red', zorder=5)


_ = plt.title('D8 channels', size=14)
# -

print(branches["features"][i]["geometry"]["coordinates"][0])
print(branches["features"][i]["geometry"]["coordinates"][-1])

# +
sns.set_palette('husl')
fig, ax = plt.subplots(figsize=(8.5,6.5))

for branch in branches_np:
    line = np.asarray(branch)
    plt.plot(line[:, 0], line[:, 1])

plt.scatter(coord1[0], coord1[1], color='green', zorder=5)
plt.scatter(coord2[0], coord2[1], color='red', zorder=5)


_ = plt.title('D8 channels', size=14)


# -

def find_segment_above(coord, branches_np):
    """Look for a segment upstream with the highest accumulation"""
    segment_above = None
    acc_above = -1
    for i, branch in enumerate(branches_np):
        if branch[-1] == coord:
            branch_acc = acc[branch[-2][0], branch[-2][1]] 
            if branch_acc > acc_above:
                segment_above = i
                acc_above = branch_acc
    return segment_above


# Finding the second highest accumulation of each branch. The highest often matches two branches that intersect.
branch_accs = []
for branch in branches_np:
    bottom_coord = branch[-2]
    branch_acc = acc[bottom_coord[0], bottom_coord[1]]
    branch_accs.append(branch_acc)
sorted_branch_ids = np.argsort(branch_accs)[::-1]

# Follow the stream all the way up the branches
largest_branch = sorted_branch_ids[0]
branch_segment_ids = []
while largest_branch != None:
    upper_coord = branches_np[largest_branch][0]
    branch_segment_ids.append(largest_branch)
    largest_branch = find_segment_above(upper_coord, branches_np)
branch_segment_ids

branch_segments = [branches_np[i] for i in sorted(branch_segment_ids)]
branch_combined = [item for sublist in branch_segments for item in sublist]

# Sort indices in descending order to avoid index shifting issues during deletion
branch_segments_sorted = sorted(branch_segment_ids, reverse=True)
for i in branch_segments_sorted:
    del branches_np[i]

len(branches_np)

branches_np = branches_np[:-1]

branches_np.append(branch_combined)

# +
sns.set_palette('husl')
fig, ax = plt.subplots(figsize=(8.5,6.5))

for branch in branches_np:
    line = np.asarray(branch)
    plt.plot(line[:, 0], line[:, 1])

plt.scatter(coord1[0], coord1[1], color='green', zorder=5)
plt.scatter(coord2[0], coord2[1], color='red', zorder=5)


_ = plt.title('D8 channels', size=14)
# -


