# 5- Adding RCM imagery and geo information to the AI4Arctic dataset

"""
This code loads raw and calibrated RCM SAR data, non-overlap masks, and metadata for each matched Sentinel-1 scene and appends them to the corresponding Sentinel-1 NetCDF file.  
It extracts and stores RCM tie-point geolocation (lat/lon), pixel coordinates, and incidence angle data from XML metadata.  
"""

# import libraries
# import libraries
import numpy as np
import os
import xarray as xr
import rasterio
import xml.etree.ElementTree as ET
from best_match_overlap import get_best_RCM_match



def load_geotiff(filepath):
    with rasterio.open(filepath) as src:
        return src.read(1)

def append_rcm_to_nc(best_RCM_match, sentinel_footprints, rcm_download_root):
    matched_rcm_folders = set()
    for rcm_path in best_RCM_match:
        parts = os.path.normpath(rcm_path).split(os.sep)
        rcm_folder = parts[-2] 
        matched_rcm_folders.add(rcm_folder)

    matched_pairs = []

    for sentinel_scene in sentinel_footprints:
        rcm_folder = sentinel_scene['rcm_folder']
        base_name = sentinel_scene['base_name']
        nc_path = os.path.join("/home/n2azad/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3", base_name)

        matched_pairs.append((sentinel_scene, nc_path))

    print(matched_pairs)
    output_dir = "new_test_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    for sentinel_scene, nc_path in matched_pairs:
        rcm_folder = sentinel_scene['rcm_folder']
        base_name = sentinel_scene['base_name']
        print(rcm_folder, base_name)
    
        rcm_scene_path = next((p for p in best_RCM_match if rcm_folder in p), None)
        parts = os.path.normpath(rcm_scene_path).split(os.sep)
        rcm_scene = parts[-1]
        full_rcm_path = os.path.join(rcm_download_root, rcm_folder, rcm_scene)
    
        cal_imagery_folder = os.path.join(full_rcm_path, rcm_scene, 'calibrated_imagery')
        sar_RCM_HH_cor_cal = next((os.path.join(cal_imagery_folder, f) for f in os.listdir(cal_imagery_folder) if f.endswith("_nan_sigma0_HH_dB_resampled_to_s1_grid_gcps.tif")), None)
        sar_RCM_HV_cor_cal = next((os.path.join(cal_imagery_folder, f) for f in os.listdir(cal_imagery_folder) if f.endswith("_nan_sigma0_HV_dB_resampled_to_s1_grid_gcps.tif")), None)
        orig_sar_RCM_HH = next((os.path.join(cal_imagery_folder, f) for f in os.listdir(cal_imagery_folder) if f.endswith("_nan_sigma0_HH_dB.tif")), None)
        orig_sar_RCM_HV = next((os.path.join(cal_imagery_folder, f) for f in os.listdir(cal_imagery_folder) if f.endswith("_nan_sigma0_HV_dB.tif")), None)
        mask_sentinel = next((os.path.join(full_rcm_path, rcm_scene, "sentinel_mask", f) for f in os.listdir(os.path.join(full_rcm_path, rcm_scene, "sentinel_mask")) if f.endswith("_nan_sigma0_HH_dB_resampled_to_s1_grid_gcps_valid_mask.tif")), None)

        orig_sar_RCM_HH = np.flipud(load_geotiff(orig_sar_RCM_HH))
        orig_sar_RCM_HV = np.flipud(load_geotiff(orig_sar_RCM_HV))
        sar_RCM_HH_cor_cal = load_geotiff(sar_RCM_HH_cor_cal)
        sar_RCM_HV_cor_cal = load_geotiff(sar_RCM_HV_cor_cal)
        mask_sentinel = load_geotiff(mask_sentinel)
        
        ds = xr.open_dataset(nc_path)
        ds["orig_sar_RCM_HH"] = xr.DataArray(orig_sar_RCM_HH, dims=["sar_lines_rcm", "sar_samples_rcm"], attrs={"id": rcm_scene})
        ds["orig_sar_RCM_HV"] = xr.DataArray(orig_sar_RCM_HV, dims=["sar_lines_rcm", "sar_samples_rcm"], attrs={"id": rcm_scene})
        ds["sar_RCM_HH_cor_cal"] = xr.DataArray(sar_RCM_HH_cor_cal, dims=["sar_lines", "sar_samples"], attrs={"id": rcm_scene, "description": "Sigma0 in dB"})
        ds["sar_RCM_HV_cor_cal"] = xr.DataArray(sar_RCM_HV_cor_cal, dims=["sar_lines", "sar_samples"], attrs={"id": rcm_scene, "description": "Sigma0 in dB"})
        ds["mask_sentinel"] = xr.DataArray(mask_sentinel, dims=["sar_lines", "sar_samples"])
    
        product_xml = os.path.join(full_rcm_path, rcm_scene, 'metadata', 'product.xml')
        tree = ET.parse(product_xml)
        root = tree.getroot()
        ns = {'ns': root.tag.split('}')[0].strip('{')}
        tie_points = root.findall(".//ns:geolocationGrid/ns:imageTiePoint", namespaces=ns)
        lines, pixels, lats, lons = [], [], [], []
        for pt in tie_points:
            lines.append(float(pt.find(".//ns:imageCoordinate/ns:line", ns).text))
            pixels.append(float(pt.find(".//ns:imageCoordinate/ns:pixel", ns).text))
            lats.append(float(pt.find(".//ns:geodeticCoordinate/ns:latitude", ns).text))
            lons.append(float(pt.find(".//ns:geodeticCoordinate/ns:longitude", ns).text))

        angle_path = os.path.join(full_rcm_path, rcm_scene, 'metadata', 'calibration', 'incidenceAngles.xml')
        angle_tree = ET.parse(angle_path)
        angle_root = angle_tree.getroot()
        angle_ns = {'ns': angle_root.tag.split('}')[0].strip('{')}
        angles = np.array([float(a.text) for a in angle_root.findall(".//ns:angles", angle_ns)])    
        pixel_first = int(angle_root.find(".//ns:pixelFirstAnglesValue", angle_ns).text)
        step_size   = int(angle_root.find(".//ns:stepSize", angle_ns).text)
        pixel_idx   = pixel_first + np.arange(angles.size) * step_size
    
        dim_name = "sar_grid_points_rcm"
        ds["sar_grid_line_rcm"] = xr.DataArray(np.array(lines), dims=[dim_name])
        ds["sar_grid_sample_rcm"] = xr.DataArray(np.array(pixels), dims=[dim_name])
        ds["sar_grid_latitude_rcm"] = xr.DataArray(np.array(lats), dims=[dim_name])
        ds["sar_grid_longitude_rcm"] = xr.DataArray(np.array(lons), dims=[dim_name])

        dim_ang = "sar_pixel_rcm"
        ds = ds.assign_coords({dim_ang: pixel_idx})
        ds["sar_grid_incidenceangle_rcm"] = xr.DataArray(angles, dims=[dim_ang])

        out_path = os.path.join(output_dir, f"{base_name}")
        ds.to_netcdf(out_path, engine="h5netcdf")





if __name__ == "__main__":
    print("Starting script", flush=True)

    rcm_download_root = "./RCM_test"  

    best_RCM_match, sentinel_footprints, _, _ = get_best_RCM_match()
    print(f"Found {len(best_RCM_match)} folders", flush=True)

    append_rcm_to_nc(best_RCM_match, sentinel_footprints, rcm_download_root)

    print("Finished appending RCM to NetCDF", flush=True)





# #===========================================================================
# # moving RCM_test to new_train_dataset
# import os
# import shutil

# src_root = "./new_test_dataset"
# dst_root = "./new_train_dataset"

# # ensure destination exists
# os.makedirs(dst_root, exist_ok=True)

# for folder in os.listdir(src_root):
#     src_path = os.path.join(src_root, folder)
#     dst_path = os.path.join(dst_root, folder)

#     if os.path.exists(src_path):
#         print(f"Moving: {folder}")
#         shutil.move(src_path, dst_path)
#     else:
#         print(f"Not found: {folder}")

# print("Done.")