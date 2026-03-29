# 2- plot the polygon of the best matched RCM scene and the ovelapping area of RCM and Sentinel-1


# import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import xarray as xr
from shapely.geometry import Polygon
from shapely.ops import unary_union
import xml.etree.ElementTree as ET
import re


rcm_download_root = "./RCM_test"
sentinel1_data_folder = "/home/n2azad/projects/rrg-dclausi/ai4arctic/dataset/ai4arctic_raw_test_v3"
shapefile_folder = "./test_dataset_shapefiles"

def get_best_RCM_match():
    """
    This code loops through downloaded RCM scenes to extract geolocation tie-points from their product.xml files.  
    It identifies tie-points along the image borders and constructs a polygon representing the scene footprint.  
    """

    # Calculate RCM polygons using tie-points from product.xml files
    rcm_polygons = []
    for order_folder in os.listdir(rcm_download_root):
        order_path = os.path.join(rcm_download_root, order_folder)

        for scene_folder in os.listdir(order_path):
            scene_path = os.path.join(order_path, scene_folder)
            product_xml_path = os.path.join(scene_path, scene_folder, "metadata", "product.xml")
            if not os.path.exists(product_xml_path):
                continue                                    

            # parse all tie-points 
            tree = ET.parse(product_xml_path)
            root = tree.getroot()
            ns = {"ns": root.tag.split('}')[0].strip('{')}
            tie_pts = root.findall(".//ns:geolocationGrid/ns:imageTiePoint", namespaces=ns)
            lines, pixels, lats, lons = [], [], [], []
            for tp in tie_pts:
                img = tp.find("ns:imageCoordinate", namespaces=ns)
                geo = tp.find("ns:geodeticCoordinate", namespaces=ns)
                lines.append(float(img.find("ns:line",   namespaces=ns).text))
                pixels.append(float(img.find("ns:pixel", namespaces=ns).text))
                lats.append(float(geo.find("ns:latitude",  namespaces=ns).text))
                lons.append(float(geo.find("ns:longitude", namespaces=ns).text))

            # identify border tie-points 
            min_line, max_line = min(lines), max(lines)
            min_pix,  max_pix  = min(pixels), max(pixels)

            border_mask = [(ln in (min_line, max_line) or px in (min_pix,  max_pix)) for ln, px in zip(lines, pixels)]
            border_pts = [(lon, lat) for lon, lat, is_b in zip(lons, lats, border_mask) if is_b]

            if len(border_pts) < 3:                         
                continue

            cx = np.mean([p[0] for p in border_pts])
            cy = np.mean([p[1] for p in border_pts])
            border_pts_sorted = sorted(border_pts, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))

            footprint = Polygon(border_pts_sorted)
            rcm_polygons.append({
                "order_folder": order_folder,
                "scene_folder": scene_folder,
                "footprint": footprint
            })

    # unique_folders = sorted({p["order_folder"] for p in rcm_polygons})
    # for folder in unique_folders:
    #     print(f"\nFolder: {folder}")
    #     for poly in rcm_polygons:
    #         if poly["order_folder"] == folder:
    #             print(poly)



    """
    This code matches Sentinel-1 shapefiles to their corresponding NetCDF (.nc) files and extracts valid SAR grid latitude/longitude points.  
    It identifies the outer border points of each scene to calculate a polygon representing the Sentinel-1 footprint.  
    The resulting footprint is visualized and stored for later comparison with RCM scenes.
    """

    sentinel_footprints = []
    available_rcm_folders = {
        f.split("_")[0]
        for f in os.listdir(rcm_download_root)
        if os.path.isdir(os.path.join(rcm_download_root, f)) and f.endswith("_RCM")
    }

    for shapefile_zip in os.listdir(shapefile_folder):
        match = re.match(r"(\d+)_([^.]+)\.zip", shapefile_zip)
        if not match:
            continue                       
        shapefile_number = match.group(1)  
        shapefile_base   = match.group(2) 
        if shapefile_number not in available_rcm_folders:
            continue

        matched_nc = None
        for nc_file in os.listdir(sentinel1_data_folder):
            if nc_file.endswith(".nc") and shapefile_base in nc_file:
                matched_nc = os.path.join(sentinel1_data_folder, nc_file)
                break
        if matched_nc is None:
            print(f"No matching .nc found for {shapefile_zip}")
            continue

        # print(f"\nFolder: {shapefile_number}  |  Sentinel file: {os.path.basename(matched_nc)}")

        ds = xr.open_dataset(matched_nc)
        lat = ds['sar_grid_latitude'].values
        lon = ds['sar_grid_longitude'].values
        line = ds['sar_grid_line'].values      
        sample = ds['sar_grid_sample'].values  

        valid = np.isfinite(lat) & np.isfinite(lon)
        lat, lon, line, sample = lat[valid], lon[valid], line[valid], sample[valid]

        top_mask = line == line.min()
        bottom_mask = line == line.max()
        left_mask = sample == sample.min()
        right_mask = sample == sample.max()

        top_idx = np.argsort(sample[top_mask])               # left -> right
        right_idx = np.argsort(line[right_mask])             # top  -> bottom
        bottom_idx = np.argsort(sample[bottom_mask])[::-1]   # right -> left
        left_idx = np.argsort(line[left_mask])[::-1]         # bottom -> top

        border_lats = np.concatenate([
            lat[top_mask][top_idx],
            lat[right_mask][right_idx],
            lat[bottom_mask][bottom_idx],
            lat[left_mask][left_idx]
        ])
        border_lons = np.concatenate([
            lon[top_mask][top_idx],
            lon[right_mask][right_idx],
            lon[bottom_mask][bottom_idx],
            lon[left_mask][left_idx]
        ])

        sentinel_polygon = Polygon(zip(border_lons, border_lats))
        matching_rcm_folder = next(f for f in available_rcm_folders if f == shapefile_number)

        sentinel_footprints.append({
            'shapefile': shapefile_zip,
            'rcm_folder': f"{matching_rcm_folder}_RCM",
            'base_name': os.path.basename(matched_nc),
            'footprint': sentinel_polygon})

        # print(f"  Sentinel: {os.path.basename(matched_nc)}")
        # print(f"  Footprint: {sentinel_footprints[-1]}") 



    """
    This code compares each Sentinel-1 footprint with RCM scene polygons to calculate their spatial overlap.  
    It selects the RCM scene with the highest intersection area for each Sentinel-1 file and records the overlap percentages.  
    """
    best_RCM_match = []
    all_overlaps = [] 

    rcm_polygon_lookup = {}
    for rcm_scene in rcm_polygons:
        rcm_path = os.path.join(
            rcm_download_root,
            rcm_scene['order_folder'],
            rcm_scene['scene_folder'],
            rcm_scene['scene_folder']
        )
        rcm_polygon_lookup[rcm_path] = rcm_scene['footprint']

    sentinel_lookup = {}
    for sentinel_scene in sentinel_footprints:
        sentinel_lookup[sentinel_scene['rcm_folder']] = sentinel_scene

    # calculate overlaps between Sentinel-1 and RCM scenes
    for sentinel_scene in sentinel_footprints:
        sentinel_footprint = sentinel_scene['footprint']
        sentinel_name = sentinel_scene['shapefile']
        sentinel_clean_name = re.sub(r'^\d+_', '', sentinel_name)
        sentinel_area = sentinel_footprint.area

        overlaps = []
        
        for rcm_scene in rcm_polygons:
            if rcm_scene['order_folder'] != sentinel_scene['rcm_folder']:
                continue

            rcm_polygon = rcm_scene['footprint']

            intersection_area = rcm_polygon.intersection(sentinel_footprint).area
            rcm_area = rcm_polygon.area

            row = {
                'sentinel_shape_file': sentinel_name,
                'sentinel_file': sentinel_clean_name,
                'rcm_folder': rcm_scene['order_folder'],
                'rcm_scene': rcm_scene['scene_folder'],
                'overlap_area': intersection_area,
                'overlap_percent_rcm': (intersection_area / rcm_area) * 100,
                'overlap_percent_sentinel': (intersection_area / sentinel_area) * 100
            }

            overlaps.append(row)
            all_overlaps.append(row)

        overlaps_sorted = sorted(overlaps, key=lambda x: x['overlap_area'], reverse=True)
        # print(overlaps_sorted)
        if overlaps_sorted:
            best_match = overlaps_sorted[:1]
            print(f"\nBest RCM match for Sentinel-1: {sentinel_name}")
            for i, match in enumerate(best_match, 1):
                print(f"\n    RCM Scene: {match['rcm_folder']}/{match['rcm_scene']}")
                print(f"    Overlap Area: {match['overlap_area']:.4f}")
                print(f"    Overlap % of RCM: {match['overlap_percent_rcm']:.2f}%")
                print(f"    Overlap % of Sentinel-1: {match['overlap_percent_sentinel']:.2f}%")
                best_path = os.path.join(rcm_download_root, match['rcm_folder'], match['rcm_scene'], match['rcm_scene'])
                best_RCM_match.append(best_path)

    return best_RCM_match, sentinel_footprints, rcm_polygon_lookup, sentinel_lookup




def plot_overlap(best_RCM_match, sentinel_lookup, rcm_polygon_lookup):
    plotpng_dir = "./plotpng_test"
    os.makedirs(plotpng_dir, exist_ok=True)

    for rcm_path in best_RCM_match:
        # print(f"\nProcessing: {rcm_path}")

        parts = rcm_path.split(os.sep)
        rcm_folder = parts[-3]
        rcm_scene = parts[-2]

        rcm_footprint = rcm_polygon_lookup.get(rcm_path)
        if rcm_footprint is None:
            raise ValueError(f"No precomputed RCM footprint found for {rcm_path}")

        matched_sentinel = sentinel_lookup.get(rcm_folder)
        if matched_sentinel is None:
            raise ValueError(f"No matching Sentinel-1 footprint found for {rcm_folder}")

        sentinel_footprint = matched_sentinel['footprint']
        # print("Matching Sentinel-1 image:", matched_sentinel['shapefile'])

        overlap_area = rcm_footprint.intersection(sentinel_footprint)
        rcm_non_overlap = rcm_footprint.difference(sentinel_footprint)
        sentinel_non_overlap = sentinel_footprint.difference(rcm_footprint)

        # print("Overlap area:", overlap_area)
        # print("RCM non-overlap area:", rcm_non_overlap)
        # print("Sentinel-1 non-overlap area:", sentinel_non_overlap)
        # print(rcm_folder)

        fig, ax = plt.subplots(figsize=(8, 8))

        if not overlap_area.is_empty:
            if overlap_area.geom_type == 'Polygon':
                ax.fill(*overlap_area.exterior.xy, color='cyan', label='Overlapping Area')
            else:
                for geom in overlap_area.geoms:
                    ax.fill(*geom.exterior.xy, color='cyan')

        if not rcm_non_overlap.is_empty:
            if rcm_non_overlap.geom_type == 'Polygon':
                ax.fill(*rcm_non_overlap.exterior.xy, color='lightgreen', label='RCM Non-Overlapping Area')
            else:
                for geom in rcm_non_overlap.geoms:
                    ax.fill(*geom.exterior.xy, color='lightgreen')

        if not sentinel_non_overlap.is_empty:
            if sentinel_non_overlap.geom_type == 'Polygon':
                ax.fill(*sentinel_non_overlap.exterior.xy, color='peachpuff', label='Sentinel-1 Non-Overlapping Area')
            else:
                for geom in sentinel_non_overlap.geoms:
                    ax.fill(*geom.exterior.xy, color='peachpuff')

        ax.plot(*sentinel_footprint.exterior.xy, color='blue', linewidth=1.5, label='Sentinel-1 Footprint')
        ax.plot(*rcm_footprint.exterior.xy, color='teal', linestyle='--', linewidth=1.5,
                label=f"RCM Footprint: {rcm_folder}/{rcm_scene}")

        ax.legend(loc='lower center', fontsize=10)
        ax.set_title("RCM & Sentinel-1 Overlap and Non-Overlap Regions", fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(plotpng_dir, f"{rcm_folder}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)



if __name__ == "__main__":
    best_RCM_match, sentinel_footprints, rcm_polygon_lookup, sentinel_lookup = get_best_RCM_match()
    plot_overlap(best_RCM_match, sentinel_lookup, rcm_polygon_lookup)