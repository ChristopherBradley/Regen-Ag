from pysheds.grid import Grid
import matplotlib.pyplot as plt
from affine import Affine
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from datetime import datetime
from scipy import ndimage

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

# Find the edges
sobel_x = ndimage.sobel(all_catchments, axis=0)
sobel_y = ndimage.sobel(all_catchments, axis=1)  
edges = np.hypot(sobel_x, sobel_y) 


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

branches = grid.extract_river_network(fdir, acc > np.max(acc)/100, dirmap=dirmap)

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
# -

# Convert to numpy coordinates 
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


# +
full_branches = []

for i in range(10):
    
    # Find the branch with the highest accumulation. Using the second last pixel before it's merged with another branch.
    branch_accs = [acc[branch[-2][0], branch[-2][1]] for branch in branches_np]
    largest_branch = np.argmax(branch_accs)

    # Follow the stream all the way up this branch
    branch_segment_ids = []
    while largest_branch != None:
        upper_coord = branches_np[largest_branch][0]
        branch_segment_ids.append(largest_branch)
        largest_branch = find_segment_above(upper_coord, branches_np)

    # Combine the segments in this branch
    branch_segments = [branches_np[i] for i in sorted(branch_segment_ids)]
    branch_combined = [item for sublist in branch_segments for item in sublist]
    full_branches.append(branch_combined)

    # Remove all the segments from that branch and start again
    branch_segments_sorted = sorted(branch_segment_ids, reverse=True)
    for i in branch_segments_sorted:
        del branches_np[i]
# -

len(full_branches)

# +
sns.set_palette('husl')
fig, ax = plt.subplots(figsize=(8.5,6.5))

for branch in full_branches:
    line = np.asarray(branch)
    plt.plot(line[:, 0], line[:, 1])

_ = plt.title('D8 channels', size=14)
# -


