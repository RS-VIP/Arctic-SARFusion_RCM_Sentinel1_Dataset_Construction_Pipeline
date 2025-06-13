import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import geopandas as gpd
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon
import xarray as xr
import matplotlib.pyplot as plt
import os
import re
import datetime
import zipfile

# create Sentinel-1 shapefiles 
folder_path = "./AI4Arctic"
output_folder = "AI4Arctic_shapefiles"
os.makedirs(output_folder, exist_ok=True)
nc_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".nc")])

for i, filename in enumerate(nc_files, start=1):
    file_path = os.path.join(folder_path, filename)
    print(f"\nProcessing file {i}/{len(nc_files)}: {filename}")
    try:
        # extract acquisition start/stop time from filename
        match = re.search(r'(\d{8}T\d{6})_(\d{8}T\d{6})', filename)
        if match:
            start_raw, stop_raw = match.groups()
            # convert to datetime
            start_dt = datetime.datetime.strptime(start_raw, "%Y%m%dT%H%M%S")
            stop_dt = datetime.datetime.strptime(stop_raw, "%Y%m%dT%H%M%S")
            acquisition_date = start_dt.strftime("%Y-%m-%d")
            start_time = start_dt.strftime("%H:%M:%S")
            stop_time = stop_dt.strftime("%H:%M:%S")
            print(f"Acquisition Date: {acquisition_date}")
            print(f"Acquisition Start Time: {start_time}")
            print(f"Acquisition Stop Time : {stop_time}")

        # extract polygon using border points
        ds = xr.open_dataset(file_path)
        lat = ds['sar_grid_latitude'].values
        lon = ds['sar_grid_longitude'].values
        line = ds['sar_grid_line'].values
        sample = ds['sar_grid_sample'].values

        # Clean NaNs
        valid_mask = np.isfinite(lat) & np.isfinite(lon)
        lat = lat[valid_mask]
        lon = lon[valid_mask]
        line = line[valid_mask]
        sample = sample[valid_mask]

        # Top row
        min_line = line.min()
        top_mask = line == min_line
        # Bottom row
        max_line = line.max()
        bottom_mask = line == max_line
        # Left column
        min_sample = sample.min()
        left_mask = sample == min_sample
        # Right column
        max_sample = sample.max()
        right_mask = sample == max_sample

        # Extract border points in order
        # Top row
        top_idx = np.argsort(sample[top_mask])
        top_lat = lat[top_mask][top_idx]
        top_lon = lon[top_mask][top_idx]
        # Right column
        right_idx = np.argsort(line[right_mask])
        right_lat = lat[right_mask][right_idx]
        right_lon = lon[right_mask][right_idx]
        # Bottom row
        bottom_idx = np.argsort(sample[bottom_mask])[::-1]
        bottom_lat = lat[bottom_mask][bottom_idx]
        bottom_lon = lon[bottom_mask][bottom_idx]
        # Left column
        left_idx = np.argsort(line[left_mask])[::-1]
        left_lat = lat[left_mask][left_idx]
        left_lon = lon[left_mask][left_idx]

        # Combine all border points in clockwise order
        border_lats = np.concatenate([top_lat, right_lat, bottom_lat, left_lat])
        border_lons = np.concatenate([top_lon, right_lon, bottom_lon, left_lon])
        polygon = Polygon(zip(border_lons, border_lats))

        # save & create shapefiles
        gdf = gpd.GeoDataFrame({'filename': [filename]}, geometry=[polygon], crs="EPSG:4326")
        shapefile_base = os.path.splitext(filename)[0]
        shapefile_path = os.path.join(output_folder, shapefile_base + ".shp")
        gdf.to_file(shapefile_path)

        # zip shapefile
        zip_filename = f"{i:02d}_{shapefile_base}.zip"
        zip_path = os.path.join(output_folder, zip_filename)
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for ext in ['.shp', '.shx', '.dbf', '.cpg', '.prj']:
                file = os.path.join(output_folder, shapefile_base + ext)
                if os.path.exists(file):
                    zipf.write(file, arcname=os.path.basename(file))

        # remove temporary shapefile components
        for ext in ['.shp', '.shx', '.dbf', '.cpg', '.prj']:
            file = os.path.join(output_folder, shapefile_base + ext)
            if os.path.exists(file):
                os.remove(file)

    except Exception as e:
        print(f"Failed to process {filename}: {e}")
