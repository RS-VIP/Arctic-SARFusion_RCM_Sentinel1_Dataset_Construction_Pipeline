import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import geopandas as gpd
from shapely.geometry import MultiPoint
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
            print(f"Acquisition Date      : {acquisition_date}")
            print(f"Acquisition Start Time: {start_time}")
            print(f"Acquisition Stop Time : {stop_time}")

        # extract bounding box polygon
        ds = xr.open_dataset(file_path)
        lat = ds['sar_grid_latitude'].values
        lon = ds['sar_grid_longitude'].values
        lat_flat = lat.flatten()
        lon_flat = lon.flatten()
        valid_mask = np.isfinite(lat_flat) & np.isfinite(lon_flat)
        lat_flat = lat_flat[valid_mask]
        lon_flat = lon_flat[valid_mask]
        points = MultiPoint(list(zip(lon_flat, lat_flat)))
        hull = points.convex_hull  

        # save & create shapefiles
        gdf = gpd.GeoDataFrame({'filename': [filename]}, geometry=[hull], crs="EPSG:4326")
        shapefile_base = os.path.splitext(filename)[0]
        shapefile_path = os.path.join(output_folder, shapefile_base + ".shp")
        gdf.to_file(shapefile_path)
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