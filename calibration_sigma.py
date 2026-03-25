# import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import rasterio
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d
from best_match_overlap import get_best_RCM_match


# # converting Tif images to float32, replace 0 values with NaN
# def convert_tif_to_nan(best_RCM_match):
#     for folder_path in best_RCM_match:
#         folder_name = os.path.basename(folder_path)        
#         order_folder = os.path.basename(os.path.dirname(folder_path))
#         folder_root = folder_path
#         imagery_path = os.path.join(folder_root, 'imagery')
#         if not os.path.isdir(imagery_path):
#             print(f"Skipping: {imagery_path} (not found)")
#             continue
#         print(f"\nProcessing: {imagery_path}")

#         for tif_file in os.listdir(imagery_path):
#             if not tif_file.endswith(".tif"):
#                 continue
#             tif_path = os.path.join(imagery_path, tif_file)

#             with rasterio.open(tif_path) as src:
#                 arr = src.read(1).astype(np.float32)  # convert to float32
#                 profile = src.profile.copy()

#             arr[arr == 0] = np.nan                   # replace 0s with NaN
#             profile.update(dtype='float32', nodata=np.nan)  # update metadata

#             base, ext = os.path.splitext(tif_file)
#             new_filename = f"{base}_nan.tif"
#             new_path = os.path.join(imagery_path, new_filename)

#             # save
#             with rasterio.open(new_path, 'w', **profile) as dst:
#                 dst.write(arr, 1)
#             print(f"Saved: {new_filename}")



"""
This code performs radiometric calibration of RCM HH and HV TIFF images into Sigma⁰ (σ⁰) backscatter in decibels using look-up tables (LUTs) from XML metadata.  
It reads the digital number (DN) values, applies calibration gains through interpolation, converts to dB scale, and prints summary statistics for each polarization.  
"""
def calibrate_sigma0(tiff_path, lut_path):
    with rasterio.open(tiff_path) as src:
        dn = src.read(1).astype(np.float32)
        tags = src.tags()
        profile = src.profile
        if src.nodata is not None:
            dn = np.ma.masked_equal(dn, src.nodata)
    tree = ET.parse(lut_path)
    root = tree.getroot()
    ns = {'ns': root.tag.split('}')[0].strip('{')}
    gains_elem = root.find("ns:gains", ns)
    gains_text = gains_elem.text.strip()
    gains = np.array(list(map(float, gains_text.split())))
    step_size_elem = root.find("ns:stepSize", ns)
    step_size = int(step_size_elem.text.strip())
    height, width = dn.shape
    num_values = len(gains)
    
    if step_size < 0:
        start_row = int(root.find("ns:pixelFirstLutValue", ns).text.strip())
        lut_rows = np.arange(start_row, start_row + step_size * num_values, step_size)
        gains = gains[::-1]
    else:
        lut_rows = np.arange(0, height, step_size)
    lut_rows = lut_rows[:len(gains)]
    interp_func = interp1d(lut_rows, gains[:len(lut_rows)], kind='linear', fill_value='extrapolate')
    gains_full = interp_func(np.arange(height))
    lut_2d = np.tile(gains_full[:, np.newaxis], (1, width))
    sigma0_linear = (dn ** 2) / lut_2d
    sigma0_dB = 10 * np.log10(sigma0_linear + 1e-10)
    return sigma0_dB, profile, tags
 
 
def run_calibration_and_save(best_RCM_match):
    for folder_path in best_RCM_match:
        folder_name = os.path.basename(folder_path)        
        order_folder = os.path.basename(os.path.dirname(folder_path))
        folder_root = folder_path
        imagery_path = os.path.join(folder_root, 'imagery')
        calibration_path = os.path.join(folder_root, 'metadata', 'calibration')
        print(f"Processing folder: {folder_path}", flush=True)

        hh_files = [f for f in os.listdir(imagery_path) if 'HH' in f and f.endswith('_HH_nan.tif')]
        print(hh_files, flush=True)
        hv_files = [f for f in os.listdir(imagery_path) if 'HV' in f and f.endswith('_HV_nan.tif')]
        print(hv_files, flush=True)

        hh_path = os.path.join(imagery_path, hh_files[0])
        hv_path = os.path.join(imagery_path, hv_files[0])
        lut_hh_path = os.path.join(calibration_path, "lutSigma_HH.xml")
        lut_hv_path = os.path.join(calibration_path, "lutSigma_HV.xml")

        # Calibrate HH, HV
        sigma0_dB_HH, profile_HH, tags_HH = calibrate_sigma0(hh_path, lut_hh_path)
        sigma0_dB_HV, profile_HV, tags_HV = calibrate_sigma0(hv_path, lut_hv_path)

        print(f"\n[HH] {order_folder, folder_name}", flush=True)
        # print(f" - Min: {sigma0_dB_HH.min():.2f} dB")
        # print(f" - Max: {sigma0_dB_HH.max():.2f} dB")
        # print(f" - Mean: {sigma0_dB_HH.mean():.2f} dB")
        print(f"\n[HV] {order_folder, folder_name}", flush=True)
        # print(f" - Min: {sigma0_dB_HV.min():.2f} dB")
        # print(f" - Max: {sigma0_dB_HV.max():.2f} dB")
        # print(f" - Mean: {sigma0_dB_HV.mean():.2f} dB")

        calibrated_dir = os.path.join(folder_root, 'calibrated_imagery')
        os.makedirs(calibrated_dir, exist_ok=True)
        # Save HH
        profile_HH.update(dtype='float32', count=1, compress='lzw')
        save_path_HH = os.path.join(calibrated_dir, f"{folder_name}_nan_sigma0_HH_dB.tif")
        with rasterio.open(save_path_HH, 'w', **profile_HH) as dst:
            dst.write(sigma0_dB_HH.astype(np.float32), 1)
            dst.update_tags(**tags_HH)
        # Save HV
        profile_HV.update(dtype='float32', count=1, compress='lzw')
        save_path_HV = os.path.join(calibrated_dir, f"{folder_name}_nan_sigma0_HV_dB.tif")
        with rasterio.open(save_path_HV, 'w', **profile_HV) as dst:
            dst.write(sigma0_dB_HV.astype(np.float32), 1)
            dst.update_tags(**tags_HV)

        # # Plot calibrated images
        # fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        # im1 = axs[0].imshow(sigma0_dB_HH, cmap='gray', vmin=-30, vmax=0)
        # axs[0].set_title(f"Calibrated HH - {order_folder}")
        # axs[0].axis('off')
        # fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04, label='σ⁰ (dB)')
        # im2 = axs[1].imshow(sigma0_dB_HV, cmap='gray', vmin=-30, vmax=0)
        # axs[1].set_title(f"Calibrated HV - {order_folder}")
        # axs[1].axis('off')
        # fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04, label='σ⁰ (dB)')
        # plt.suptitle(f"Radiometrically Calibrated RCM σ⁰ Images - {order_folder}/{folder_name}", fontsize=14)
        # plt.tight_layout()
        # plt.show()


if __name__ == "__main__":
    print("Starting script", flush=True)

    best_RCM_match, _, _, _ = get_best_RCM_match()
    print(f"Found {len(best_RCM_match)} folders", flush=True)
    # convert_tif_to_nan(best_RCM_match)

    run_calibration_and_save(best_RCM_match)
    print("Finished calibration", flush=True)



