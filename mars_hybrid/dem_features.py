from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from rasterio import features
from rasterio.transform import Affine
from scipy import ndimage
from skimage import measure
from shapely.geometry import Polygon, shape


@dataclass
class DemFeatureResult:
    slope: np.ndarray
    curvature: np.ndarray
    depth: np.ndarray
    roughness: np.ndarray
    valid_mask: np.ndarray
    pit_truth: np.ndarray
    thresholds: Dict[str, float]


def _nan_gaussian(data: np.ndarray, sigma: float, valid: np.ndarray) -> np.ndarray:
    weighted = ndimage.gaussian_filter(np.where(valid, data, 0.0), sigma=sigma)
    weights = ndimage.gaussian_filter(valid.astype(np.float32), sigma=sigma)
    out = np.divide(weighted, weights, out=np.full_like(weighted, np.nan), where=weights > 1e-6)
    return out


def _local_std(data: np.ndarray, valid: np.ndarray, size: int = 7) -> np.ndarray:
    kernel = np.ones((size, size), dtype=np.float32)
    weights = ndimage.convolve(valid.astype(np.float32), kernel, mode="nearest")
    mean = ndimage.convolve(np.where(valid, data, 0.0), kernel, mode="nearest")
    mean = np.divide(mean, weights, out=np.full_like(mean, np.nan), where=weights > 0)

    sq = ndimage.convolve(np.where(valid, data**2, 0.0), kernel, mode="nearest")
    sq = np.divide(sq, weights, out=np.full_like(sq, np.nan), where=weights > 0)
    var = np.maximum(sq - mean**2, 0.0)
    return np.sqrt(var)


def compute_dem_features(z: np.ndarray, nodata: float) -> DemFeatureResult:
    valid = z != nodata
    zf = np.where(valid, z, np.nan)

    z_blur = _nan_gaussian(zf, sigma=6.0, valid=valid)
    depth = z_blur - zf

    z_small = _nan_gaussian(zf, sigma=1.5, valid=valid)
    gy, gx = np.gradient(z_small)
    slope = np.sqrt(gx**2 + gy**2)

    z_filled = np.where(valid, zf, np.nanmean(zf))
    curvature = ndimage.laplace(z_filled)
    curvature[~valid] = np.nan

    roughness = _local_std(zf, valid, size=7)

    depth_thr = float(np.nanpercentile(depth[valid], 90.0))
    curv_thr = float(np.nanpercentile(curvature[valid], 20.0))
    slope_thr = float(np.nanpercentile(slope[valid], 60.0))

    pit_truth = (depth > depth_thr) & (curvature < curv_thr) & (slope < slope_thr) & valid

    return DemFeatureResult(
        slope=slope,
        curvature=curvature,
        depth=depth,
        roughness=roughness,
        valid_mask=valid,
        pit_truth=pit_truth,
        thresholds={"depth": depth_thr, "curvature": curv_thr, "slope": slope_thr},
    )


def extract_pit_components(mask: np.ndarray, transform: Affine, min_area_pixels: int = 20) -> Tuple[list[Polygon], np.ndarray]:
    labels = measure.label(mask, connectivity=2)
    polygons: list[Polygon] = []

    for region_id in range(1, labels.max() + 1):
        region_mask = labels == region_id
        if int(region_mask.sum()) < min_area_pixels:
            continue
        shapes = features.shapes(region_mask.astype(np.uint8), mask=region_mask, transform=transform)
        for geom, value in shapes:
            if value != 1:
                continue
            poly = shape(geom)
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)

    return polygons, labels
