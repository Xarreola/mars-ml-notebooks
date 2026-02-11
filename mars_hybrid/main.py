from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds

from .dem_features import compute_dem_features, extract_pit_components
from .inference import InferenceConfig, run_full_scene_inference
from .jp2_features import FeatureConfig, compute_feature_stack, feature_names
from .training import fit_logistic_model
from .vectorize import polygons_to_geojson, vectorize_binary_raster


def collect_training_samples(
    jp2_path: str,
    dem_features,
    dem_transform,
    dem_crs,
    max_samples: int = 250_000,
    tile_size: int = 1024,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)
    x_parts = []
    y_parts = []

    valid = dem_features.valid_mask
    labels = dem_features.pit_truth

    with rasterio.open(jp2_path) as jp2:
        for row in range(0, valid.shape[0], tile_size):
            for col in range(0, valid.shape[1], tile_size):
                h = min(tile_size, valid.shape[0] - row)
                w = min(tile_size, valid.shape[1] - col)
                dem_win = Window(col, row, w, h)

                valid_tile = valid[row : row + h, col : col + w]
                if not valid_tile.any():
                    continue

                bounds = rasterio.windows.bounds(dem_win, dem_transform)
                jp2_win = from_bounds(*bounds, transform=jp2.transform)
                img = jp2.read(
                    1,
                    window=jp2_win,
                    out_shape=(h, w),
                    resampling=Resampling.bilinear,
                    boundless=True,
                    fill_value=0,
                )
                feats = compute_feature_stack(img, FeatureConfig())
                mask = valid_tile & np.isfinite(feats).all(axis=-1)
                if not mask.any():
                    continue

                xv = feats[mask]
                yv = labels[row : row + h, col : col + w][mask].astype(np.uint8)

                if xv.shape[0] > 8000:
                    idx = rng.choice(xv.shape[0], size=8000, replace=False)
                    xv = xv[idx]
                    yv = yv[idx]
                x_parts.append(xv)
                y_parts.append(yv)

    if not x_parts:
        raise RuntimeError("No valid training samples extracted from DEM-valid region.")

    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    if x.shape[0] > max_samples:
        idx = rng.choice(x.shape[0], size=max_samples, replace=False)
        x = x[idx]
        y = y[idx]

    return x.astype(np.float32), y.astype(np.uint8)


def run_pipeline(jp2_path: str, dem_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        dem_nodata = dem_src.nodata if dem_src.nodata is not None else -9999.0
        dem_result = compute_dem_features(dem, dem_nodata)
        dem_transform = dem_src.transform

    pit_polys, _ = extract_pit_components(dem_result.pit_truth, dem_transform, min_area_pixels=20)
    polygons_to_geojson(pit_polys, str(out / "dem_pit_truth.geojson"), properties={"source": "dem_truth"})

    x_train, y_train = collect_training_samples(
        jp2_path=jp2_path,
        dem_features=dem_result,
        dem_transform=dem_transform,
        dem_crs=None,
    )
    train_res = fit_logistic_model(x_train, y_train)

    prob_tif = str(out / "pit_probability_full.tif")
    bin_tif = str(out / "pit_binary_full.tif")
    run_full_scene_inference(
        jp2_path=jp2_path,
        model=train_res.model,
        probability_out=prob_tif,
        binary_out=bin_tif,
        feature_config=FeatureConfig(),
        infer_config=InferenceConfig(tile_size=1024, threshold=0.55),
    )

    vectorize_binary_raster(bin_tif, str(out / "skylight_candidates.geojson"), min_area_pixels=20)

    report = {
        "training_samples": train_res.n_train,
        "training_auc": train_res.auc,
        "pit_positive_fraction": train_res.class_balance,
        "dem_thresholds": dem_result.thresholds,
        "jp2_features": list(feature_names()),
        "outputs": {
            "probability_raster": prob_tif,
            "binary_raster": bin_tif,
            "candidate_geojson": str(out / "skylight_candidates.geojson"),
            "dem_truth_geojson": str(out / "dem_pit_truth.geojson"),
        },
    }
    with open(out / "run_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid DEM–HiRISE skylight and lava tube detection pipeline")
    p.add_argument("--jp2", default="/work/imagery/hirise/ESP_037232_1770/jp2/ESP_037232_1770_RED.JP2")
    p.add_argument("--dem", default="/work/stereo/final/roi_full/roi_-DEM_warped_to_JP2.tif")
    p.add_argument("--out", default="/out")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.jp2, args.dem, args.out)
