#%% [markdown]

import xarray as xr
import glob
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
from Enhance_dataset import Enhance_image
from Enhance_dataset import filter_outliers
import rasterio
from rasterio.crs import CRS
from rasterio.control import GroundControlPoint
from netCDF4 import Dataset
import os

def save_rasters(image, gcps, filename, crs_epsg=4326):
    """
    Save an image as a GeoTIFF with GCPs.
    """
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=image.shape[0],
        width=image.shape[1],
        count=1,
        dtype=image.dtype,
        crs=CRS.from_epsg(crs_epsg),
        nodata=np.nan
    ) as dst:
        dst.write(image, 1)
        dst.gcps = (gcps, CRS.from_epsg(crs_epsg))


files = glob.glob("C:/temp/new_train_dataset/*.nc")
for scene_file in files:
    print(f"Processing file: {scene_file}")
    # Use h5netcdf engine to read the file
    # try:
    scene = xr.open_dataset(scene_file)
    # except FileNotFoundError:
    #     print(scene_file)


    # --- Prepare GCPs from Sentinel-1 sparse grid  
    rows_s1, cols_s1 = scene["nersc_sar_primary"].shape
    lines_s1 = scene["sar_grid_line"].values
    samples_s1 = scene["sar_grid_sample"].values
    lats_s1 = scene["sar_grid_latitude"].values
    lons_s1 = scene["sar_grid_longitude"].values
    gcps = []
    for line, sample, lat, lon in zip(lines_s1, samples_s1, lats_s1, lons_s1):
        row = int(round(line))
        col = int(round(sample))
        if 0 <= row < rows_s1 and 0 <= col < cols_s1:
            gcps.append(GroundControlPoint(row=row, col=col, x=lon, y=lat))
    
    # --- Load images and check shapes
    polarization = 'HH' # Change to 'HV' for HV polarization#
    print(f"Processing {polarization} band")
    if polarization == 'HH':
        S1 = scene['nersc_sar_primary'].values
        RCM = scene['sar_RCM_HH_cor_cal'].values
        orig_RCM = scene['orig_sar_RCM_HH'].values
    else:
        S1 = scene['nersc_sar_secondary'].values
        RCM = scene['sar_RCM_HV_cor_cal'].values
        orig_RCM = scene['orig_sar_RCM_HV'].values
    assert S1.shape == RCM.shape, f"Shapes do not match: {S1.shape} != {RCM.shape}"
    ic(S1.shape, RCM.shape, orig_RCM.shape)

    # --- Save GeoTIFFs with GCPs
    # Sentinel 1
    output_dir = f"./Geocoded_images/{os.path.split(scene_file)[1].split('.nc')[0][:67]}"
    os.makedirs(output_dir, exist_ok=True)
    save_rasters(S1, gcps, output_dir + f"/S1_{polarization}.tif", crs_epsg=4326)
    save_rasters(RCM, gcps, output_dir + f"/RCM_{polarization}.tif", crs_epsg=4326)

    # masks
    mask_S1 = np.isnan(RCM) | np.isnan(S1)

    # Define common limits to stretch histograms
    clips = filter_outliers(orig_RCM.copy(), bins=2**16-1, bth=0.001, uth=0.999, 
                                train_pixels=np.asarray(np.where(~np.isnan(orig_RCM))))
    clips_ = filter_outliers(RCM.copy(), bins=2**16-1, bth=0.001, uth=0.999, 
                                train_pixels=np.asarray(np.where(~mask_S1)))
    clips[0] = min(clips[0], clips_[0]); clips[1] = max(clips[1], clips_[1]) # use the same clips for both images
    clips_ = filter_outliers(S1.copy(), bins=2**16-1, bth=0.001, uth=0.999, 
                                train_pixels=np.asarray(np.where(~np.isnan(S1))))
    clips[0] = min(clips[0], clips_[0]); clips[1] = max(clips[1], clips_[1]) # use the same clips for both images

    idx = np.where(np.isnan(orig_RCM)); x = idx[0][:2]; y = idx[1][:2]
    orig_RCM[x[0], y[0]] = clips[0]; orig_RCM[x[1], y[1]] = clips[1] 
    idx = np.where(mask_S1); x = idx[0][:2]; y = idx[1][:2]
    RCM[x[0], y[0]] = clips[0]; RCM[x[1], y[1]] = clips[1] 
    idx = np.where(np.isnan(S1)); x = idx[0][:2]; y = idx[1][:2]
    S1[x[0], y[0]] = clips[0]; S1[x[1], y[1]] = clips[1] 


    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0,0].imshow(Enhance_image(orig_RCM, np.isnan(orig_RCM), clips=clips), cmap='gray')
    axes[0,0].set_title('Original RCM')
    axes[0,1].imshow(Enhance_image(RCM, mask_S1, clips=clips), cmap='gray')
    axes[0,1].set_title('co-registered RCM')
    axes[0,2].imshow(Enhance_image(S1, np.isnan(S1), clips=clips), cmap='gray')
    axes[0,2].set_title('S1')

    # axes[0,0].imshow(Enhance_image(orig_RCM, np.isnan(orig_RCM)), cmap='gray')
    # axes[0,0].set_title('Original RCM')
    # axes[0,1].imshow(Enhance_image(RCM, mask_S1), cmap='gray')
    # axes[0,1].set_title('RCM')
    # axes[0,2].imshow(Enhance_image(S1, np.isnan(S1)), cmap='gray')
    # axes[0,2].set_title('S1')

    # hist, bins  = np.histogram(orig_RCM[~np.isnan(orig_RCM)], bins=2000)
    # axes[1,0].plot(bins[1:], hist/(hist.sum()))
    # axes[1,0].set_title("orig RCM hist")
    # hist, bins  = np.histogram(RCM[~mask_S1], bins=2000)
    # axes[1,1].plot(bins[1:], hist/(hist.sum()))
    # axes[1,1].set_title("co-registered RCM hist")
    # hist, bins  = np.histogram(S1[~mask_S1], bins=2000)
    # axes[1,2].plot(bins[1:], hist/(hist.sum()))
    # axes[1,2].set_title("S1 hist")
    plt.tight_layout()
    plt.show()
    # plt.savefig(output_dir + f"/{polarization}_GRD_geometry&dist.png")
    # plt.close()
