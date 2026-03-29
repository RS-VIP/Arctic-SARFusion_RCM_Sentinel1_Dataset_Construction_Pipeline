# 6- checking the updated dataset

import xarray as xr
import numpy as np
import os

DATA_PATH = "/home/n2azad/projects/rrg-dclausi/ai4arctic/dataset/RCM_Arctic_dataset"

variables_to_check = [
    "nersc_sar_primary", "nersc_sar_secondary",
    "sar_RCM_HH_cor_cal", "sar_RCM_HV_cor_cal"
]

required_vars = [
    "nersc_sar_primary", "nersc_sar_secondary", "polygon_icechart", "distance_map",
    "sar_grid_line", "sar_grid_sample", "sar_grid_latitude", "sar_grid_longitude",
    "sar_grid_incidenceangle", "sar_grid_height", "btemp_6_9h", "btemp_6_9v",
    "btemp_7_3h", "btemp_7_3v", "btemp_10_7h", "btemp_10_7v", "btemp_18_7h",
    "btemp_18_7v", "btemp_23_8h", "btemp_23_8v", "btemp_36_5h", "btemp_36_5v",
    "btemp_89_0h", "btemp_89_0v", "amsr2_swath_map", "swath_segmentation",
    "u10m_rotated", "v10m_rotated", "t2m", "skt", "tcwv", "tclw",
    "orig_sar_RCM_HH", "orig_sar_RCM_HV", "sar_RCM_HH_cor_cal", "sar_RCM_HV_cor_cal",
    "mask_sentinel", "sar_grid_line_rcm", "sar_grid_sample_rcm", "sar_grid_latitude_rcm",
    "sar_grid_longitude_rcm", "sar_grid_incidenceangle_rcm"
]

def diagnose(arr):
    if arr is None: return "❌ Missing"
    if arr.size == 0: return "❌ Empty"
    if np.all(np.isnan(arr)): return "❌ NaN"
    v = arr[~np.isnan(arr)]
    if v.size == 0: return "❌ No valid"
    if np.nanmin(v) == np.nanmax(v): return "❌ Constant"
    if arr.ndim != 2: return f"⚠ Shape {arr.shape}"
    if min(arr.shape) < 1000: return f"⚠ Small {arr.shape}"
    return "✔"

total = incomplete = 0
problem_scenes = []

for root, _, files in os.walk(DATA_PATH):
    for f in files:
        if not f.endswith(".nc"):
            continue

        total += 1
        path = os.path.join(root, f)
        print(f"\nChecking: {f}")

        with xr.open_dataset(path, engine="h5netcdf") as ds:
            vars_in_file = set(ds.variables)

            missing = [v for v in required_vars if v not in vars_in_file]
            if missing:
                incomplete += 1
                print(f"❌ Missing ({len(missing)}):", ", ".join(missing))

            has_problem = False
            for name in variables_to_check:
                arr = ds[name].values if name in vars_in_file else None
                diagnosis = diagnose(arr)
                if diagnosis != "✔":
                    has_problem = True
                    print(f"{name}: {diagnosis}")

            if has_problem:
                problem_scenes.append(f)

print("\n===== SUMMARY =====")
print(f"Total: {total}")
print(f"Incomplete: {incomplete}")
print(f"Complete: {total - incomplete}")
if problem_scenes:
    print(f"\nChannel issues: {len(problem_scenes)}")
    print("\nProblem scenes:")
    print("\n".join(problem_scenes))