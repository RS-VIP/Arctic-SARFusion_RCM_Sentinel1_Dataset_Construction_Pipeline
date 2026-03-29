# RCM shapefile creator 

import os
import zipfile
import tempfile
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

def make_rcm_shapefile(root_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for scene_name in sorted(os.listdir(root_folder)):
        scene_path = os.path.join(root_folder, scene_name)
        if not os.path.isdir(scene_path):
            continue

        # find product.xml
        product_xml = None
        for root, _, files in os.walk(scene_path):
            if "product.xml" in files:
                product_xml = os.path.join(root, "product.xml")
                break

        if product_xml is None:
            print(f"{scene_name}: product.xml not found")
            continue

        try:
            # extract footprint
            tree = ET.parse(product_xml)
            root = tree.getroot()
            ns = {"ns": root.tag.split('}')[0].strip('{')}

            tie_pts = root.findall(".//ns:geolocationGrid/ns:imageTiePoint", ns)

            lines, pixels, lats, lons = [], [], [], []

            for tp in tie_pts:
                img = tp.find("ns:imageCoordinate", ns)
                geo = tp.find("ns:geodeticCoordinate", ns)

                lines.append(float(img.find("ns:line", ns).text))
                pixels.append(float(img.find("ns:pixel", ns).text))
                lats.append(float(geo.find("ns:latitude", ns).text))
                lons.append(float(geo.find("ns:longitude", ns).text))

            if len(lines) == 0:
                raise RuntimeError("No tie-points found")

            min_line, max_line = min(lines), max(lines)
            min_pix, max_pix   = min(pixels), max(pixels)

            border_pts = [
                (lon, lat)
                for ln, px, lon, lat in zip(lines, pixels, lons, lats)
                if ln in (min_line, max_line) or px in (min_pix, max_pix)
            ]

            cx = np.mean([p[0] for p in border_pts])
            cy = np.mean([p[1] for p in border_pts])

            polygon = Polygon(
                sorted(border_pts, key=lambda p: np.arctan2(p[1]-cy, p[0]-cx))
            )

            # save shapefile
            tmp_dir = tempfile.mkdtemp()
            shp_path = os.path.join(tmp_dir, f"{scene_name}.shp")

            gdf = gpd.GeoDataFrame(
                [{"scene_id": scene_name, "geometry": polygon}],
                crs="EPSG:4326"
            )
            gdf.to_file(shp_path, driver="ESRI Shapefile")

            # zip shapefile
            zip_out = os.path.join(output_folder, f"{scene_name}.zip")
            with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in os.listdir(tmp_dir):
                    z.write(os.path.join(tmp_dir, f), arcname=f)

            shutil.rmtree(tmp_dir)

            print(f"{scene_name}: shapefile saved")

        except Exception as e:
            print(f"{scene_name}: failed → {e}")


if __name__ == "__main__":
    root_folder = "./RCM_test"
    output_folder = "./RCM_test_shapefiles"
    make_rcm_shapefile(root_folder, output_folder)
