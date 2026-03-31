#%% [markdown]

import xarray as xr
import glob
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.crs import CRS
from rasterio.control import GroundControlPoint
from netCDF4 import Dataset
import os
import itertools

def filter_outliers(img, bins=2**16-1, bth=0.001, uth=0.999, train_pixels=None):
    
    if len(img.shape) == 2:
        rows, cols = img.shape
        bands = 1
        img = img[:,:,np.newaxis]
    else:
        rows, cols, bands = img.shape

    if train_pixels is None:
        h = np.arange(0, rows)
        w = np.arange(0, cols)
        train_pixels = np.asarray(list(itertools.product(h, w))).transpose()

    min_value, max_value = [], []
    for band in range(bands):
        hist, bins = np.histogram(img[train_pixels[0], train_pixels[1], band].ravel(), bins=bins) # select training pixels
        cum_hist = np.cumsum(hist) / hist.sum()

        # # See outliers cut values
        # plt.plot(bins[1:], hist)
        # plt.plot(bins[1:], cum_hist)
        # plt.stem(bins[len(cum_hist[cum_hist<bth])], 0.5)
        # plt.stem(bins[len(cum_hist[cum_hist<uth])], 0.5)
        # plt.title("band %d"%(band))
        # plt.show()

        min_value.append(bins[len(cum_hist[cum_hist<bth])])
        max_value.append(bins[len(cum_hist[cum_hist<uth])])
        
    return [np.array(min_value), np.array(max_value)]

def median_filter(img, clips, mask):
    kernel_size = 10

    outliers = ((img < clips[0]) + (img > clips[1]))
    if len(img.shape) == 3:
        outliers *= np.expand_dims(mask, axis=2)
    else: outliers *= mask
    # plt.imshow(outliers[:,:,0], cmap='gray')
    # plt.imshow(outliers[:,:,1], cmap='gray')
    # plt.show()
    out_idx = np.asarray(np.where(outliers))

    img_ = img.copy()
    for i in range(out_idx.shape[1]):
        x = out_idx[0][i]
        y = out_idx[1][i]
        a = x - kernel_size//2 if x - kernel_size//2 >=0 else 0
        c = y - kernel_size//2 if y - kernel_size//2 >=0 else 0
        b = x + kernel_size//2 if x + kernel_size//2 <= img.shape[0] else img.shape[0]
        d = y + kernel_size//2 if y + kernel_size//2 <= img.shape[1] else img.shape[1]
        win = img[a:b, c:d][mask[a:b, c:d]==True]
        img_[x, y] = np.median(win, axis=0)
        # img_[x, y] = np.mean(win, axis=0)
    
    return img_

def Enhance_image(img, land_nan_mask, output_folder='', clips=None):

    # fig, axs = plt.subplots(2, 1, figsize=(16, 8))
    # hist, bins  = np.histogram(img[~land_nan_mask], bins=10000)
    # axs[0].plot(bins[1:], hist/(hist.sum())); axs[0].set_title("hist")

    if clips is None:
        clips = filter_outliers(img.copy(), bins=2**16-1, bth=0.001, uth=0.999, 
                                train_pixels=np.asarray(np.where(~land_nan_mask)))
    img = median_filter(img, clips, ~land_nan_mask)

    # hist, bins  = np.histogram(img[~land_nan_mask], bins=10000)
    # axs[1].plot(bins[1:], hist/(hist.sum())); axs[1].set_title("(no outliers)-hist")
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(output_folder + 'histogram.png')
    # plt.close()

    min_ = img[~land_nan_mask].min(0)
    max_ = img[~land_nan_mask].max(0)
    img = np.uint8(255*((img - min_) / (max_ - min_)))
    
    img[land_nan_mask] = 255

    return img

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
