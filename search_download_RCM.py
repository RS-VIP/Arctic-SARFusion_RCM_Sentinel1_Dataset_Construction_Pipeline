# 1- Retrieve, Search, and Download RCM Scenes based on Sentinel-1s' shapefiles


"""
For each Sentinel-1 shapefile, this code extracts a buffered time range (10 hours) based on the filename.  
It then queries the EODMS API for overlapping RCM scenes with HH and HV polarizations within that time window and area.  
The matching scenes are saved as a GeoJSON file, and a new RCM order is submitted using the selected scene IDs.
And saving the RCM scenes geo information in an Excel file
"""

"""
Note: after submitting the orders(run the 1st part of the code), wait for the request approval email, and then download the scenes (run the 2nd & 3rd part of the code)
"""
# part 1 --------------------------------------------------------------------------------------------------------------------------------------
# Search for compatible RCM Scenes 
from getpass import getpass
import os
from datetime import datetime,timedelta
import geopandas as gpd
import zipfile
import tempfile
import re
from eodms_api_client import EodmsAPI
import pandas as pd
 
# Sentinel-1 shapefile path
shapefile_folder = "./train_dataset_shapefiles/"
records = []
best_matches_data = [] 
 
# get EODMS user credentials
username = input("EODMS username: ")
password = getpass("EODMS password: ")
 
# extract time range from Sentinel-1 filename
# create a time window of 10 hours (5h before, 5h after) to search for compatible RCM scenes
def extract_time_range(filename):
    match = re.search(r'_(\d{8}T\d{6})_(\d{8}T\d{6})_', filename)
    start = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
    end = datetime.strptime(match.group(2), "%Y%m%dT%H%M%S")
    return (start - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%S"), (end + timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%S")

for zip_file in sorted(os.listdir(shapefile_folder), key=lambda f: int(re.match(r'\d+', f).group())):
    zip_path = os.path.join(shapefile_folder, zip_file)
    print(f"\n{zip_file}")
 
    # extract time range and prefix
    start_date, end_date = extract_time_range(zip_file)
    print(f"Buffered Time Range: {start_date} to {end_date}")
    prefix_match = re.match(r"(\d+)_", zip_file)
    prefix = prefix_match.group(1) if prefix_match else "unknown"
 
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        shp_file = [f for f in os.listdir(tmpdir) if f.endswith('.shp')][0]
        shp_path = os.path.join(tmpdir, shp_file)
        geojson_path = shp_path.replace('.shp', '.geojson')
        gdf = gpd.read_file(shp_path)
        gdf.to_file(geojson_path, driver="GeoJSON")
 
        # connect to EODMS API and query
        client = EodmsAPI(collection='RCM', username=username, password=password)
        client.query(start=start_date, end=end_date, geometry=geojson_path)
        print(f"Found {len(client.results)} scenes.")
        if len(client.results) == 0:
            records.append({                    
                "shapefile": zip_file,
                "time_window": f"{start_date} – {end_date}",
                "num_RCM_scenes": 0
            })
            print("No scenes found")
            continue
 
        # filter by polarization (Only HH, HV), Product Type (GRD), swath width (350km), LUT (Ice)
        filtered = client.results[
            (client.results["Polarization in Product"].str.contains("HH") & client.results["Polarization in Product"].str.contains("HV")) &
            (client.results["Beam Mode Description"].str.contains("350km Swath", case=False, na=False)) &
            (client.results["Product Type"] == "GRD") &
            (client.results["LUT Applied"] == "Ice")
        ]
        
        print(f"The number of scenes with HH & HV channels: {len(filtered)}")
        
        records.append({
            "shapefile"  : zip_file,
            "time_window" : f"{start_date} – {end_date}",
            "num_RCM_scenes" : len(filtered)
        })
        print("\n--- Full Metadata for Filtered Scenes ---")
        print(filtered.to_string(index=False))

        if not filtered.empty:
            # Use the accurate Sentinel-1 footprint polygon (not bounding box)
            s1_geom = gdf.unary_union
    
            # RCM footprints (already Shapely polygons in filtered["geometry"])
            rcm_gdf = gpd.GeoDataFrame(filtered, geometry="geometry", crs=gdf.crs)
    
            # Compute intersection area
            rcm_gdf["intersection_area"] = rcm_gdf.geometry.apply(lambda rcm_geom: rcm_geom.intersection(s1_geom).area)
            rcm_gdf["rcm_area"] = rcm_gdf.geometry.apply(lambda geom: geom.area)
            rcm_gdf["overlap_percent"] = (rcm_gdf["intersection_area"] / rcm_gdf["rcm_area"]) * 100
    
            # Remove scenes with 0 overlap
            rcm_gdf = rcm_gdf[rcm_gdf["intersection_area"] > 0]
        else:
            print("No HH/HV scenes found in the filtered results.")

        # Identify the best overlapping RCM scene 
        if not rcm_gdf.empty:
            best_match = rcm_gdf.loc[rcm_gdf["intersection_area"].idxmax()]
            s1_area = s1_geom.area
            s1_overlap_percent = (best_match["intersection_area"] / s1_area) * 100
            print("\n Best Overlapping RCM Scene (by polygon intersection):")
            print(best_match[["Granule", "overlap_percent"]])

            # Save best match data for Excel
            meta_cols = [
                "Granule", "Beam Mnemonic", "Beam Mode Type", "Beam Mode Description", "Spatial Resolution", "Polarization Data Mode", "Polarization", "Polarization in Product", 
                "Incidence Angle (Low)", "Incidence Angle (High)", "Orbit Direction", "LUT Applied", "Product Format", "Product Type"
            ]
            print(best_match)
            row_data = {
                "Sentinel-1 shapefile": zip_file,
                "Overlap % (RCM with S1)": best_match["overlap_percent"],
                "Overlap % (S1 with RCM)": s1_overlap_percent,
            }
            for col in meta_cols:
                row_data[col] = best_match.get(col, "")

            best_matches_data.append(row_data)
        else:
            print("\n No overlapping RCM scenes found with valid intersection.")

        # Submit order
        best_record_id = best_match["EODMS RecordId"]
        order_ids = client.order([best_record_id])
        print(f"\n Submitted order for best match: Record ID {best_record_id}")
        print(f"Order ID: {order_ids[0]}")

        # Save order IDs
        order_dir = "./order_ids"
        os.makedirs(order_dir, exist_ok=True)
        order_file_path = os.path.join(order_dir, f"{prefix}_RCM.txt")
        with open(order_file_path, "w") as f:
            f.write(str(order_ids[0]))
 

# Save all best matching RCM scenes information to an Excel file
df_best = pd.DataFrame(best_matches_data)
df_best.to_excel("match_rcm_scenes_summary.xlsx", index=False)
# --------------------------------------------------------------------------------------------------------------------------------------



# part 2 --------------------------------------------------------------------------------------------------------------------------------------
# Download the ordered RCM scenes
import time
from pathlib import Path

username = input("EODMS username: ")
password = getpass("EODMS password: ")
client = EodmsAPI(collection='RCM', username=username, password=password)
order_dir = "./order_ids"
rcm_download_root = "./RCM"
os.makedirs(rcm_download_root, exist_ok=True)

for order_file in os.listdir(order_dir):
    order_file_path = os.path.join(order_dir, order_file)
    order_id = int(Path(order_file_path).read_text().strip())

    prefix = order_file.split("_")[0]
    folder = os.path.join(rcm_download_root, f"{prefix}_RCM")
    os.makedirs(folder, exist_ok=True)

    if list(Path(folder).glob("*.zip")):
        print(f"Already downloaded: {prefix}")
        continue

    print(f"Downloading order {order_id} to {folder}")
    try:
        client.download([order_id], folder)
    except Exception as e:
        print(f"Failed once: {e}\nRetrying in 10s...")
        time.sleep(10)
        client.download([order_id], folder)

    if list(Path(folder).glob("*.zip")):
        print(f"Download complete: {prefix}")
    else:
        print(f"No file found in {folder}")

# --------------------------------------------------------------------------------------------------------------------------------------

# part 3
# unzip RCM compress files
for folder_name in os.listdir(rcm_download_root):
    folder_path = os.path.join(rcm_download_root, folder_name)
    if not os.path.isdir(folder_path):
        continue
    print(f"{folder_path}")
    for zip_name in os.listdir(folder_path):
        if zip_name.endswith(".zip"):
            zip_path = os.path.join(folder_path, zip_name)
            extract_folder = os.path.join(folder_path, zip_name.replace(".zip", ""))
            if not os.path.exists(extract_folder):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_folder)
                    print(f"Extracted:{zip_name}")
                os.remove(zip_path)