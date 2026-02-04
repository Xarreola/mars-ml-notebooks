#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mars ML script refactor for Colab execution."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter
from skimage.filters import laplace, sobel
from skimage.morphology import skeletonize

# ---------------------------
# Config (Colab-friendly)
# ---------------------------
product_type = "IMAGERY"
JP2_PATH = "/content/drive/MyDrive/HIRISE_DTMS/ESP_037232_1770_MIRB.JP2"
OUT_DIR_LOCAL = "/content/outputs"
OUT_DIR_DRIVE = "/content/drive/MyDrive/MARS/outputs"


@dataclass
class OutputPaths:
    local: str
    drive: str


def ensure_output_dirs(paths: OutputPaths) -> None:
    os.makedirs(paths.local, exist_ok=True)
    os.makedirs(paths.drive, exist_ok=True)


def percentile_stretch(img: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    if img.ndim == 2:
        lo, hi = np.percentile(img[np.isfinite(img)], [p_low, p_high])
        stretched = (img - lo) / (hi - lo + 1e-6)
        return np.clip(stretched, 0, 1).astype(np.float32)

    stretched = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[0]):
        channel = img[c]
        lo, hi = np.percentile(channel[np.isfinite(channel)], [p_low, p_high])
        stretched[c] = np.clip((channel - lo) / (hi - lo + 1e-6), 0, 1)
    return stretched.astype(np.float32)


def normalize_channel(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mask = np.isfinite(arr)
    if not np.any(mask):
        return np.zeros_like(arr, dtype=np.float32)
    vmin = np.nanpercentile(arr[mask], 1)
    vmax = np.nanpercentile(arr[mask], 99)
    out = (arr - vmin) / (vmax - vmin + eps)
    return np.clip(out, 0, 1).astype(np.float32)


def load_jp2(path: str) -> Tuple[np.ndarray, rasterio.Affine, Dict]:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        transform = src.transform
        profile = src.profile
    if data.shape[0] == 1:
        data = data[0]
    return data, transform, profile


def save_preview(image: np.ndarray, out_path: str, title: str) -> None:
    plt.figure(figsize=(6, 6))
    if image.ndim == 2:
        plt.imshow(image, cmap="gray")
    else:
        if image.shape[0] > 3:
            image = image[:3]
        plt.imshow(np.moveaxis(image, 0, -1))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_to_outputs(paths: OutputPaths, filename: str, image: np.ndarray, title: str) -> None:
    save_preview(image, os.path.join(paths.local, filename), title)
    save_preview(image, os.path.join(paths.drive, filename), title)


def imagery_fields(image: np.ndarray) -> Dict[str, np.ndarray]:
    if image.ndim == 2:
        gray = image
    else:
        gray = np.mean(image[:3], axis=0)

    gray_n = normalize_channel(gray)
    grad_mag = sobel(gray_n)
    lap = laplace(gray_n)

    mean = uniform_filter(gray_n, size=7)
    mean_sq = uniform_filter(gray_n ** 2, size=7)
    local_std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0.0))

    return {
        "gray_n": gray_n,
        "grad_mag": normalize_channel(grad_mag),
        "laplace": normalize_channel(lap),
        "local_std": normalize_channel(local_std),
    }


def dem_fields(dem: np.ndarray, transform: rasterio.Affine) -> Dict[str, np.ndarray]:
    dx = transform.a
    dy = -transform.e
    gy, gx = np.gradient(dem, dy, dx)
    slope_rad = np.arctan(np.sqrt(gx ** 2 + gy ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    curvature = laplace(dem.astype(np.float32))
    return {
        "dem_n": normalize_channel(dem),
        "slope": normalize_channel(slope_deg),
        "curvature": normalize_channel(curvature),
    }


def generate_hybrid_sites(field: np.ndarray, seed_xy: Tuple[int, int]) -> np.ndarray:
    h, w = field.shape
    seed_x, seed_y = seed_xy
    rng = np.random.default_rng(0)

    xs = np.arange(100, w, 200)
    ys = np.arange(100, h, 200)
    gx, gy = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

    jitter = rng.normal(scale=40.0, size=grid_pts.shape)
    grid_pts = grid_pts + jitter

    mask = (
        (grid_pts[:, 0] >= 0) & (grid_pts[:, 0] < w) &
        (grid_pts[:, 1] >= 0) & (grid_pts[:, 1] < h)
    )
    sites = [p.tolist() for p in grid_pts[mask]]

    finite = field[np.isfinite(field)]
    zmin, zmax = float(finite.min()), float(finite.max())
    levels = np.linspace(zmin, zmax, 12)

    fig, ax = plt.subplots()
    contours = ax.contour(field, levels=levels)
    plt.close(fig)

    for segs in contours.allsegs:
        for seg in segs:
            if seg.size == 0:
                continue
            step = max(1, seg.shape[0] // 40)
            for x, y in seg[::step]:
                if 0 <= x < w and 0 <= y < h:
                    sites.append([x, y])

    sites = np.array(sites, dtype=float)
    if sites.size == 0:
        sites = np.array([[seed_x, seed_y]], dtype=float)

    d = np.hypot(sites[:, 0] - seed_x, sites[:, 1] - seed_y)
    if np.min(d) > 1.5:
        sites = np.vstack([sites, [seed_x, seed_y]])

    return sites


def voronoi_field(field: np.ndarray, seed_xy: Tuple[int, int]) -> np.ndarray:
    from scipy.spatial import cKDTree

    h, w = field.shape
    points = generate_hybrid_sites(field, seed_xy)

    tree = cKDTree(points)
    yy, xx = np.indices((h, w))
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    _, idx = tree.query(coords, k=1)
    labels = idx.reshape(h, w)

    mean_val = np.full(points.shape[0], np.nan, dtype=np.float32)
    for i in range(points.shape[0]):
        mask = labels == i
        if np.any(mask):
            mean_val[i] = np.nanmean(field[mask])

    return normalize_channel(mean_val[labels])


def build_attractiveness(fields: Dict[str, np.ndarray], vor_field: np.ndarray) -> np.ndarray:
    if product_type == "DEM":
        proxy = 0.6 * fields["slope"] + 0.4 * fields["curvature"]
    else:
        proxy = 0.5 * fields["grad_mag"] + 0.3 * fields["laplace"] + 0.2 * fields["local_std"]

    attractiveness = normalize_channel(proxy + 0.5 * vor_field)
    return attractiveness


def ml_iteration_stub(attractiveness: np.ndarray) -> np.ndarray:
    print("[ML stub] Place model training/inference here. Using attractiveness as proxy output.")
    return attractiveness


def physarum_growth(attractiveness: np.ndarray, steps: int = 1200) -> np.ndarray:
    h, w = attractiveness.shape
    food_field = 1.0 - normalize_channel(attractiveness)

    trail = np.zeros_like(food_field, dtype=np.float32)
    rng = np.random.default_rng(42)

    n_agents = max(2000, (h * w) // 5000)
    agents_x = rng.uniform(0, w - 1, size=n_agents)
    agents_y = rng.uniform(0, h - 1, size=n_agents)
    agents_theta = rng.uniform(0, 2 * np.pi, size=n_agents)

    sensor_dist = 4.0
    sensor_angle = np.deg2rad(35.0)
    turn_angle = np.deg2rad(25.0)
    step_size = 1.0

    for _ in range(steps):
        field = 0.6 * trail + 2.5 * food_field

        fx = agents_x + np.cos(agents_theta) * sensor_dist
        fy = agents_y + np.sin(agents_theta) * sensor_dist
        lx = agents_x + np.cos(agents_theta + sensor_angle) * sensor_dist
        ly = agents_y + np.sin(agents_theta + sensor_angle) * sensor_dist
        rx = agents_x + np.cos(agents_theta - sensor_angle) * sensor_dist
        ry = agents_y + np.sin(agents_theta - sensor_angle) * sensor_dist

        fx_i = np.clip(fx.round().astype(int), 0, w - 1)
        fy_i = np.clip(fy.round().astype(int), 0, h - 1)
        lx_i = np.clip(lx.round().astype(int), 0, w - 1)
        ly_i = np.clip(ly.round().astype(int), 0, h - 1)
        rx_i = np.clip(rx.round().astype(int), 0, w - 1)
        ry_i = np.clip(ry.round().astype(int), 0, h - 1)

        sf = field[fy_i, fx_i]
        sl = field[ly_i, lx_i]
        sr = field[ry_i, rx_i]

        turn_left = (sl > sf) & (sl > sr)
        turn_right = (sr > sf) & (sr > sl)

        agents_theta = agents_theta + turn_left * turn_angle
        agents_theta = agents_theta - turn_right * turn_angle
        agents_theta = agents_theta + rng.normal(scale=0.03, size=n_agents)

        agents_x = np.clip(agents_x + np.cos(agents_theta) * step_size, 0, w - 1)
        agents_y = np.clip(agents_y + np.sin(agents_theta) * step_size, 0, h - 1)

        xi = agents_x.round().astype(int)
        yi = agents_y.round().astype(int)
        trail[yi, xi] += 1.0

        trail *= 0.96
        trail = uniform_filter(trail, size=1)

    return normalize_channel(trail)


def overlay_skeleton(base: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    if base.ndim == 2:
        base_rgb = np.stack([base, base, base], axis=0)
    else:
        base_rgb = base[:3].copy()

    overlay = base_rgb.copy()
    overlay[:, skeleton > 0] = np.array([1.0, 0.2, 0.2])[:, None]
    return overlay


def main() -> None:
    if not os.path.exists(JP2_PATH):
        print(
            f"JP2 file not found at {JP2_PATH}. "
            "Mount Drive at /content/drive and check the path."
        )
        sys.exit(1)

    outputs = OutputPaths(local=OUT_DIR_LOCAL, drive=OUT_DIR_DRIVE)
    ensure_output_dirs(outputs)

    data, transform, _ = load_jp2(JP2_PATH)
    stretched = percentile_stretch(data)
    save_to_outputs(outputs, "jp2_preview.png", stretched, "JP2 Preview")

    if product_type == "DEM":
        if data.ndim != 2:
            raise ValueError("DEM mode expects a single-band JP2.")
        fields = dem_fields(data, transform)
        proxy_field = normalize_channel(fields["slope"] + fields["curvature"])
    else:
        fields = imagery_fields(data)
        proxy_field = normalize_channel(
            0.5 * fields["grad_mag"] + 0.3 * fields["laplace"] + 0.2 * fields["local_std"]
        )

    # Pipeline order: SAI/proxy -> Voronoi -> scoring -> ML stub -> moss-like output
    seed_xy = (proxy_field.shape[1] // 2, proxy_field.shape[0] // 2)
    vor_field = voronoi_field(proxy_field, seed_xy)
    attractiveness = build_attractiveness(fields, vor_field)
    _ = ml_iteration_stub(attractiveness)

    density = physarum_growth(attractiveness)
    threshold = np.percentile(density, 90)
    mask = (density >= threshold).astype(np.uint8)
    skeleton = skeletonize(mask > 0).astype(np.uint8)

    save_to_outputs(outputs, "attractiveness.png", attractiveness, "Attractiveness")
    save_to_outputs(outputs, "density.png", density, "Physarum Density")

    overlay = overlay_skeleton(stretched, skeleton)
    save_to_outputs(outputs, "skeleton_overlay.png", overlay, "Skeleton Overlay")

    print("Preview written to:", outputs.local, "and", outputs.drive)


if __name__ == "__main__":
    main()
