from __future__ import annotations

import json
from typing import Iterable

import rasterio
from rasterio import features
from shapely.geometry import shape, mapping


def vectorize_binary_raster(binary_raster_path: str, output_geojson: str, min_area_pixels: int = 16) -> None:
    feats = []
    with rasterio.open(binary_raster_path) as src:
        arr = src.read(1)
        for geom, val in features.shapes(arr, transform=src.transform):
            if int(val) != 1:
                continue
            poly = shape(geom)
            if poly.is_empty or (poly.area / abs(src.transform.a * src.transform.e)) < min_area_pixels:
                continue
            feats.append(
                {
                    "type": "Feature",
                    "properties": {"class": "skylight_candidate"},
                    "geometry": mapping(poly),
                }
            )

    collection = {"type": "FeatureCollection", "features": feats}
    with open(output_geojson, "w", encoding="utf-8") as f:
        json.dump(collection, f)


def polygons_to_geojson(polygons: Iterable, output_geojson: str, properties: dict | None = None) -> None:
    if properties is None:
        properties = {}
    feats = [{"type": "Feature", "properties": properties, "geometry": mapping(poly)} for poly in polygons]
    with open(output_geojson, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
