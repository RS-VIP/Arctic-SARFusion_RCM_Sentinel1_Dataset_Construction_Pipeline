# plotting the RCM scenes on the glob map based on RCM shapefiles


import os
import glob
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


zip_folder = "./RCM_shapefiles"
output_path = "rcm_scene_distribution_square.png"
central_lon = -35

# load RCM shapefiles
zip_files = sorted(glob.glob(os.path.join(zip_folder, "*.zip")))
print(f"Found {len(zip_files)} zip files")

gdf_list = []

for zf in zip_files:
    try:
        gdf = gpd.read_file(f"zip://{os.path.abspath(zf)}")
        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf = gdf.to_crs("EPSG:4326")
        gdf_list.append(gdf)
        print(f"Loaded: {os.path.basename(zf)}")
    except Exception as e:
        print(f"Error reading {os.path.basename(zf)}: {e}")


all_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs="EPSG:4326")
print(f"Total polygons: {len(all_gdf)}")

# map
fig = plt.figure(figsize=(8, 8))
proj = ccrs.PlateCarree()   # flat map (lon/lat)
ax = fig.add_axes([0.10, 0.10, 0.80, 0.80], projection=proj)
ax.set_extent([-140, 20, 55, 88], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.OCEAN, facecolor="#dfe8ec", zorder=0)
ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black", linewidth=0.5, zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=2)

gl = ax.gridlines( crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", alpha=0.35, linestyle="-")
gl.top_labels = True
gl.bottom_labels = True
gl.left_labels = True
gl.right_labels = False
gl.xlabel_style = {"size": 9}
gl.ylabel_style = {"size": 9}
all_gdf.plot( ax=ax, transform=ccrs.PlateCarree(), facecolor="#08306b", edgecolor="none", alpha=0.18, zorder=3)
plt.savefig(output_path, dpi=300, facecolor="white")
plt.close()
print(f"Saved to {output_path}")