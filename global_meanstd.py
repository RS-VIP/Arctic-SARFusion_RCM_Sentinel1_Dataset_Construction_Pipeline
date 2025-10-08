import os
import numpy as np
import xarray as xr
from tqdm import tqdm


data_dir = "/home/n2azad/projects/rrg-dclausi/ai4arctic/dataset/new_RCM_ai4arctic_train"
files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".nc")]
print(f"Found {len(files)} files.")

variables = ["sar_RCM_HH_cor_cal", "sar_RCM_HV_cor_cal", "sar_grid_incidenceangle_rcm"]


# compute global mean, min, max
sum_values = {var: 0.0 for var in variables}
count_values = {var: 0 for var in variables}
min_values = {var: np.inf for var in variables}
max_values = {var: -np.inf for var in variables}

for f in files:
    try:
        ds = xr.open_dataset(f)
        for var in variables:
            if var in ds.variables:
                data = ds[var].values
                data = data[np.isfinite(data)]
                if data.size > 0:
                    sum_values[var] += np.sum(data)
                    count_values[var] += data.size
                    min_values[var] = min(min_values[var], np.min(data))
                    max_values[var] = max(max_values[var], np.max(data))
        ds.close()
    except Exception as e:
        print(f"Error reading {f}: {e}")

means = {var: sum_values[var] / count_values[var] for var in variables}
print(means)
# compute std using global mean
sum_sq_diff = {var: 0.0 for var in variables}

for f in files:
    try:
        ds = xr.open_dataset(f)
        for var in variables:
            if var in ds.variables:
                data = ds[var].values
                data = data[np.isfinite(data)]
                if data.size > 0:
                    diff = data - means[var]
                    sum_sq_diff[var] += np.sum(diff ** 2)
        ds.close()
    except Exception as e:
        print(f"Error reading {f}: {e}")

stds = {var: np.sqrt(sum_sq_diff[var] / count_values[var]) for var in variables}
print(stds)
final = {}
for var in variables:
    final[var] = {
        'mean': np.float64(means[var]),
        'std': np.float64(stds[var]),
        'min': np.float64(min_values[var]),
        'max': np.float64(max_values[var]),
    }


print("\nNewly computed statistics:")
for k, v in final.items():
    print(f"{k}: {v}")


meanstd_path = "/home/n2azad/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_train_v3/global_meanstd.npy"
meanstd_output_path = "./global_meanstd.npy"
# minmax_path = "global_minmax.npy"

try:
    meanstd = np.load(meanstd_path, allow_pickle=True).item()
except FileNotFoundError:
    meanstd = {}

# try:
#     minmax = np.load(minmax_path, allow_pickle=True).item()
# except FileNotFoundError:
#     minmax = {}

# Add new results
for var, stats in tqdm(final.items(), desc="Updating NPY file", unit="var"):
    meanstd[var] = {'mean': stats['mean'], 'std': stats['std']}
    # minmax[var] = {'min': stats['min'], 'max': stats['max']}

np.save(meanstd_output_path, meanstd)
# np.save(minmax_path, minmax)






