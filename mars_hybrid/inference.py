from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.windows import Window

from .jp2_features import FeatureConfig, compute_feature_stack


@dataclass
class InferenceConfig:
    tile_size: int = 1024
    threshold: float = 0.55


def run_full_scene_inference(
    jp2_path: str,
    model,
    probability_out: str,
    binary_out: str,
    feature_config: FeatureConfig | None = None,
    infer_config: InferenceConfig | None = None,
) -> None:
    if feature_config is None:
        feature_config = FeatureConfig()
    if infer_config is None:
        infer_config = InferenceConfig()

    with rasterio.open(jp2_path) as src:
        profile = src.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="float32", compress="deflate", nodata=np.nan)
        bprofile = profile.copy()
        bprofile.update(dtype="uint8", nodata=0)

        with rasterio.open(probability_out, "w", **profile) as dst_prob, rasterio.open(binary_out, "w", **bprofile) as dst_bin:
            for row in range(0, src.height, infer_config.tile_size):
                for col in range(0, src.width, infer_config.tile_size):
                    h = min(infer_config.tile_size, src.height - row)
                    w = min(infer_config.tile_size, src.width - col)
                    window = Window(col, row, w, h)
                    img = src.read(1, window=window)
                    feats = compute_feature_stack(img, feature_config)
                    x = feats.reshape(-1, feats.shape[-1])
                    prob = model.predict_proba(x)[:, 1].reshape(h, w).astype(np.float32)
                    pred = (prob >= infer_config.threshold).astype(np.uint8)
                    dst_prob.write(prob, 1, window=window)
                    dst_bin.write(pred, 1, window=window)
