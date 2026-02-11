from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage


@dataclass
class FeatureConfig:
    log_sigmas: tuple[float, ...] = (1.0, 2.5)
    texture_size: int = 7
    normalize_clip: tuple[float, float] = (1.0, 99.0)


def normalize_intensity(image: np.ndarray, clip: tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    lo, hi = np.percentile(image, clip)
    denom = max(hi - lo, 1e-6)
    norm = np.clip((image - lo) / denom, 0.0, 1.0)
    return norm.astype(np.float32)


def compute_feature_stack(image: np.ndarray, config: FeatureConfig | None = None) -> np.ndarray:
    if config is None:
        config = FeatureConfig()

    img = image.astype(np.float32)
    norm = normalize_intensity(img, config.normalize_clip)

    gx = ndimage.sobel(norm, axis=1)
    gy = ndimage.sobel(norm, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)

    logs = [ndimage.gaussian_laplace(norm, sigma=s) for s in config.log_sigmas]

    mean = ndimage.uniform_filter(norm, size=config.texture_size)
    sq_mean = ndimage.uniform_filter(norm**2, size=config.texture_size)
    texture_std = np.sqrt(np.maximum(sq_mean - mean**2, 0.0))

    shadow_core = norm < 0.22
    bright_rim = ndimage.maximum_filter(norm, size=5) > 0.78
    shadow_proxy = shadow_core.astype(np.float32) * bright_rim.astype(np.float32)

    stack = np.stack([norm, grad_mag, texture_std, shadow_proxy, *logs], axis=-1)
    return stack.astype(np.float32)


def feature_names(config: FeatureConfig | None = None) -> Iterable[str]:
    if config is None:
        config = FeatureConfig()
    names = ["norm_intensity", "sobel_grad", "texture_std", "shadow_proxy"]
    names.extend([f"log_sigma_{s}" for s in config.log_sigmas])
    return names
