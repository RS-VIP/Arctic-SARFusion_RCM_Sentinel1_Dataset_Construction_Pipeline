# 4- Co-registering RCM with Sentinel-1


# import libraries
import numpy as np
import os
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
import xml.etree.ElementTree as ET
import xarray as xr
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from best_match_overlap import get_best_RCM_match

rcm_download_root = "./RCM_test"

def create_sentinel_mask_dirs(best_RCM_match, rcm_download_root):
    for rcm_path in best_RCM_match:
        parts = os.path.normpath(rcm_path).split(os.sep)
        rcm_folder = parts[-3]
        rcm_scene = parts[-2]
        scene_base = os.path.join(rcm_download_root, rcm_folder, rcm_scene)
        sentinel_mask_dir = os.path.join(scene_base, rcm_scene, "sentinel_mask")
        os.makedirs(sentinel_mask_dir, exist_ok=True)
        print(f"[RCM] {rcm_folder} | [SCENE] {rcm_scene} → {sentinel_mask_dir}")


# Extract RCM sparse geolocation tie points (line, pixel) to geographic (lat, lon) from product.xml
def parse_rcm_tie_points(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'ns': root.tag.split('}')[0].strip('{')}
    lines, pixels, lats, lons = [], [], [], []
    for tp in root.findall(".//ns:geolocationGrid/ns:imageTiePoint", namespaces=ns):
        img = tp.find("ns:imageCoordinate", namespaces=ns)
        geo = tp.find("ns:geodeticCoordinate", namespaces=ns)
        lines.append(float(img.find("ns:line",   namespaces=ns).text))
        pixels.append(float(img.find("ns:pixel", namespaces=ns).text))
        lats.append(float(geo.find("ns:latitude",  namespaces=ns).text))
        lons.append(float(geo.find("ns:longitude", namespaces=ns).text))
    return np.array(lines), np.array(pixels), np.array(lats), np.array(lons)



def coregister_s1_rcm(best_RCM_match, sentinel_footprints):
    # Load S1 NetCDF and interpolate lat/lon to every pixel of the full S1 image grid
    matched_rcm_set     = {os.path.normpath(p).split(os.sep)[-3] for p in best_RCM_match}
    matched_rcm_folders = sorted(matched_rcm_set, key=lambda f: int(f.split('_')[0]))
    print(matched_rcm_folders)
    for sentinel_scene in sentinel_footprints:
        print(sentinel_scene)
        rcm_folder = sentinel_scene['rcm_folder']
        if rcm_folder not in matched_rcm_folders:
            continue
        sentinel_nc_path = os.path.join("/home/n2azad/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3", sentinel_scene['base_name'])

        print(f"Using .nc file: {sentinel_nc_path}")
        ds = xr.open_dataset(sentinel_nc_path, engine="h5netcdf")
        s1_data = ds["nersc_sar_primary"].values
        rows_s1, cols_s1 = s1_data.shape
        lines_s1 = ds["sar_grid_line"].values
        samples_s1 = ds["sar_grid_sample"].values
        lats_s1 = ds["sar_grid_latitude"].values
        lons_s1 = ds["sar_grid_longitude"].values
        ds.close()

        s1_grid_row, s1_grid_col = np.meshgrid(np.arange(rows_s1), np.arange(cols_s1), indexing='ij')
        points_s1 = np.column_stack((lines_s1, samples_s1))
        s1_pixel_coords = np.column_stack((s1_grid_row.ravel(), s1_grid_col.ravel()))
        lat_s1_grid = griddata(points_s1, lats_s1, s1_pixel_coords, method='cubic').reshape(rows_s1, cols_s1)
        lon_s1_grid = griddata(points_s1, lons_s1, s1_pixel_coords, method='cubic').reshape(rows_s1, cols_s1)

        # Interpolate S1 lat/lon to RCM pixel locations
        # Resample RCM image onto Sentinel-1 pixel grid and attach GCPs
        order_path = os.path.join("./RCM_test", rcm_folder)
        for scene_folder in os.listdir(order_path):
            scene_path = os.path.join(order_path, scene_folder, scene_folder)

            '''
            interpolation functions that map geographic coordinates (lat, lon) back to RCM image pixel coordinates.  
            This inverse mapping lets us find, for each pixel in the S1 grid (with known lat/lon), the corresponding location in the RCM image.  
            '''
            rcm_xml_path = os.path.join(scene_path, "metadata", "product.xml")
            print(rcm_xml_path)
            lines_rcm, pixels_rcm, lats_rcm, lons_rcm = parse_rcm_tie_points(rcm_xml_path)
            print(f"  pairing S1 = {sentinel_nc_path}  ↔  RCM = {order_path}")
            points_rcm_geo = np.column_stack((lats_rcm, lons_rcm))
            latlon_coords  = np.column_stack((lat_s1_grid.ravel(), lon_s1_grid.ravel()))
            rcm_row_map    = griddata(points_rcm_geo, lines_rcm,   latlon_coords,  method='cubic').reshape(rows_s1, cols_s1)
            rcm_col_map    = griddata(points_rcm_geo, pixels_rcm,  latlon_coords,  method='cubic').reshape(rows_s1, cols_s1) 

            '''
            With the coordinate maps (`rcm_row_map`, `rcm_col_map`), we use `map_coordinates` to resample the original RCM image to the S1 pixel grid.  
            We then prepare Ground Control Points (GCPs) from the Sentinel-1 sparse grid, which contain georeferencing info (pixel to lat/lon).  
            Finally, we save the resampled RCM image **directly with the GCPs attached** to enable georeferencing without intermediate files.
            '''
            imagery_path = os.path.join(scene_path, "calibrated_imagery")
            if not os.path.isdir(imagery_path):
                continue

            for tif_name in os.listdir(imagery_path):
                if not tif_name.lower().endswith(".tif"):
                    continue
                rcm_raster_path = os.path.join(imagery_path, tif_name)
                with rasterio.open(rcm_raster_path) as rcm_src:
                    rcm_data    = rcm_src.read(1)
                    rcm_profile = rcm_src.profile

                # Resample RCM data to Sentinel-1 grid coordinates (assumes rcm_row_map, rcm_col_map, rows_s1, cols_s1 are ready)
                coords    = np.vstack([rcm_row_map.ravel(), rcm_col_map.ravel()])
                resampled = map_coordinates(rcm_data, coords, order=1, mode='nearest').reshape(rows_s1, cols_s1)

                # Prepare GCPs from Sentinel-1 sparse grid
                gcps = []
                for line, sample, lat, lon in zip(lines_s1, samples_s1, lats_s1, lons_s1):
                    row = int(round(line))
                    col = int(round(sample))
                    if 0 <= row < rows_s1 and 0 <= col < cols_s1:
                        gcps.append(GroundControlPoint(row=row, col=col, x=lon, y=lat))

                # Update profile for new resampled image
                profile = rcm_profile.copy()
                profile.update({
                    "height": rows_s1,
                    "width":  cols_s1,
                    "dtype":  resampled.dtype,
                    "count":  1,
                })
                profile.pop('nodata', None)

                out_path = os.path.join( imagery_path, f"{os.path.splitext(tif_name)[0]}_resampled_to_s1_grid_gcps.tif")
                print(f"{out_path}  →  shape: {resampled.shape}")

                # Write resampled RCM image directly with GCPs
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(resampled, 1)
                    dst.gcps = (gcps, CRS.from_epsg(4326))
                print(f"Resampled RCM image saved with GCPs: {out_path}")

                # Create and save binary valid-pixel mask
                with rasterio.open(out_path) as src:
                    img = src.read(1)
                    profile = src.profile.copy()

                mask = np.where(np.isnan(img), 0, 1).astype(np.uint8)
                profile.update( dtype='uint8', count=1, nodata=0, compress='lzw')
                sentinel_mask_folder = os.path.join(scene_path, "sentinel_mask")
                mask_filename = os.path.splitext(os.path.basename(out_path))[0] + "_valid_mask.tif"
                mask_path = os.path.join(sentinel_mask_folder, mask_filename)
                with rasterio.open(mask_path, "w", **profile) as dst:
                    dst.write(mask, 1)



if __name__ == "__main__":
    print("Starting script", flush=True)

    best_RCM_match, sentinel_footprints, _, _ = get_best_RCM_match()

    create_sentinel_mask_dirs(best_RCM_match, rcm_download_root)

    coregister_s1_rcm(best_RCM_match, sentinel_footprints)
    print("Finished Coregirtration", flush=True)


