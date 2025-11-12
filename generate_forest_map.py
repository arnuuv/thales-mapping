#!/usr/bin/env python3
"""
Procedural fictitious forest map generator (with rivers) → PNG + optional GeoTIFF

Outputs:
  - forest_rgb.png        (styled 8‑bit RGB rendering)
  - mask_water.png        (water mask)
  - dem.png               (hillshade-like preview)
  - forest_rgb.tif        (GeoTIFF, if rasterio is installed)

Notes
-----
* No external noise libs required (custom fractal value-noise / FBM).
* Rivers are carved from a synthetic DEM using D8 flow + flow accumulation.
* Landcover (forest density/grass/rock/water) is driven by elevation & moisture
  (moisture derived from distance-to-river + orographic precipitation proxy).
* Coordinates are synthetic; GeoTIFF is written with 1 m pixel size unless you
  change PIXEL_SIZE.

Quick start
-----------
python generate_forest_map.py --size 2048 --seed 7 --pixel-size 1

Then drag forest_rgb.tif into QGIS. Reproject/clip as you like.
"""
from __future__ import annotations
import argparse
import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.transform import from_origin
    RASTERIO_OK = True
except Exception:
    RASTERIO_OK = False

# ------------------------------
# Utility: simple FBM value noise
# ------------------------------

def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(t):
    return t * t * (3 - 2 * t)


def value_noise(width: int, height: int, grid: int, rng: np.random.Generator) -> np.ndarray:
    """Value noise: generate a coarse random grid then bilinearly upsample.
    grid: grid cell size in pixels (>= 2).
    """
    gw = (width // grid) + 2
    gh = (height // grid) + 2
    g = rng.random((gh, gw), dtype=np.float32)

    # create coordinates
    y = np.arange(height, dtype=np.float32)
    x = np.arange(width, dtype=np.float32)
    yy = (y[:, None] / grid)
    xx = (x[None, :] / grid)

    xi = np.floor(xx).astype(int)
    yi = np.floor(yy).astype(int)

    xf = smoothstep(xx - xi)
    yf = smoothstep(yy - yi)

    # sample corners
    g00 = g[yi,     xi]
    g10 = g[yi,     xi+1]
    g01 = g[yi+1,   xi]
    g11 = g[yi+1,   xi+1]

    # bilinear blend
    nx0 = lerp(g00, g10, xf)
    nx1 = lerp(g01, g11, xf)
    n = lerp(nx0, nx1, yf)
    return n.astype(np.float32)


def fbm(width: int, height: int, octaves=6, lacunarity=2.0, gain=0.5, base_grid=128, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    amp = 1.0
    freq = 1.0
    total = np.zeros((height, width), dtype=np.float32)
    norm = 0.0
    for o in range(octaves):
        grid = max(2, int(base_grid / (freq)))
        n = value_noise(width, height, grid, rng)
        total += n * amp
        norm += amp
        amp *= gain
        freq *= lacunarity
    total /= max(1e-6, norm)
    return total

# ------------------------------
# Terrain & hydrology
# ------------------------------

def make_dem(w: int, h: int, seed: int) -> np.ndarray:
    """Synthetic DEM (0..1)."""
    # base ridges & valleys
    base = fbm(w, h, octaves=6, base_grid=256, seed=seed)
    detail = fbm(w, h, octaves=4, base_grid=64, seed=seed+1)
    dem = 0.75 * base + 0.25 * detail

    # shape the island/region with a soft radial mask so edges become lowlands
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    dx = (xx - cx) / (0.6 * w)
    dy = (yy - cy) / (0.6 * h)
    rad = np.clip(np.sqrt(dx*dx + dy*dy), 0, 1)
    bowl = 1 - rad**2
    dem = dem * 0.8 + bowl * 0.2

    # normalize
    dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)
    return dem.astype(np.float32)


def d8_flow(dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute D8 flow directions and slopes.
    Returns (dir_idx, slope) where dir_idx in [0..7] for 8 neighbors (clockwise),
    or -1 for pits.
    """
    h, w = dem.shape
    # neighbor offsets (E, NE, N, NW, W, SW, S, SE) — clockwise
    offs = np.array([
        (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1), (1,0), (1,1)
    ])
    dir_idx = -np.ones((h, w), dtype=np.int8)
    slope = np.zeros((h, w), dtype=np.float32)

    for i in range(1, h-1):
        for j in range(1, w-1):
            z = dem[i, j]
            best = 0.0
            best_k = -1
            for k, (di, dj) in enumerate(offs):
                ni = i + di; nj = j + dj
                dz = z - dem[ni, nj]
                if dz > best:
                    best = dz
                    best_k = k
            dir_idx[i, j] = best_k
            slope[i, j] = best
    return dir_idx, slope

from collections import deque

def flow_accumulation(dir_idx: np.ndarray) -> np.ndarray:
    """
    Fast D8 flow accumulation using a topological (Kahn) pass.
    Each cell contributes 1 to itself, then pushes its flow to its single
    downstream neighbor. O(N). Handles pits (cells with dir=-1) naturally.
    """
    h, w = dir_idx.shape
    offs = np.array([
        (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1), (1,0), (1,1)
    ], dtype=np.int32)

    # Precompute downstream coordinate and indegree (number of upstream cells)
    downstream = np.full((h, w, 2), -1, dtype=np.int32)
    indeg = np.zeros((h, w), dtype=np.int32)

    for i in range(1, h-1):
        for j in range(1, w-1):
            k = dir_idx[i, j]
            if k >= 0:
                di, dj = offs[k]
                ti, tj = i + di, j + dj
                downstream[i, j, 0] = ti
                downstream[i, j, 1] = tj
                indeg[ti, tj] += 1

    # Initialize with one unit of flow per cell
    acc = np.ones((h, w), dtype=np.float32)

    # Start with sources (indegree 0). Ignore borders/pits with dir=-1 for queueing.
    q = deque()
    for i in range(1, h-1):
        for j in range(1, w-1):
            if indeg[i, j] == 0 and dir_idx[i, j] != -1:
                q.append((i, j))

    # Kahn-style propagation
    while q:
        i, j = q.popleft()
        ti, tj = downstream[i, j]
        if ti >= 0:
            acc[ti, tj] += acc[i, j]
            indeg[ti, tj] -= 1
            if indeg[ti, tj] == 0 and dir_idx[ti, tj] != -1:
                q.append((ti, tj))

    return acc



def cheap_blur(img: np.ndarray, k: int = 5) -> np.ndarray:
    """Box blur using rolling window; avoids SciPy dependency."""
    if k <= 1:
        return img
    out = img.copy()
    for _ in range(k):
        out = (np.roll(out, 1, 0) + out + np.roll(out, -1, 0)) / 3.0
        out = (np.roll(out, 1, 1) + out + np.roll(out, -1, 1)) / 3.0
    return out


def river_mask_from_dem(dem: np.ndarray, seed: int, target_density=0.005) -> np.ndarray:
    """Create a boolean mask for rivers using flow accumulation + threshold.
    target_density controls how many pixels become rivers.
    """
    dir_idx, _ = d8_flow(dem)
    acc = flow_accumulation(dir_idx)

    # Normalize accumulation and pick threshold by quantile
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-6)
    q = 1.0 - target_density
    thr = np.quantile(acc, q)
    rivers = acc >= thr

    # Thicken main channels with a little blur + threshold
    rivf = cheap_blur(rivers.astype(np.float32), k=2)
    rivers = rivf > 0.3

    # Mask out high ridges (optional): only allow rivers below 0.8 elevation
    rivers &= (dem < 0.8)
    return rivers

# ------------------------------
# Biome & styling
# ------------------------------

def make_moisture(dem: np.ndarray, rivers: np.ndarray) -> np.ndarray:
    # Distance-to-river via iterative diffusion (cheap proxy)
    moist = rivers.astype(np.float32)
    moist = cheap_blur(moist, k=6)

    # Orographic effect: wind from west → more rain on west-facing slopes
    # Approximate slope facing by x-gradient
    gx = np.gradient(dem, axis=1)
    orographic = np.clip(-gx, 0, None)  # west-facing positive
    orographic = (orographic - orographic.min()) / (orographic.max() - orographic.min() + 1e-6)

    m = 0.6 * moist + 0.4 * orographic
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    return m


def shade(dem: np.ndarray, azimuth_deg=315, altitude_deg=45) -> np.ndarray:
    """Simple Lambertian hillshade (0..1)."""
    az = math.radians(azimuth_deg)
    alt = math.radians(altitude_deg)
    # Gradients
    gy, gx = np.gradient(dem)
    # Surface normal components
    slope = np.sqrt(gx*gx + gy*gy)
    aspect = np.arctan2(-gx, gy)
    hs = (np.sin(alt) * (1 - slope) + np.cos(alt) * slope * np.cos(az - aspect))
    hs = np.clip(hs, 0, 1)
    return hs


def palette_forest(elev: np.ndarray, moist: np.ndarray, rivers: np.ndarray) -> np.ndarray:
    """Map elevation & moisture to RGB colors with forest emphasis."""
    h, w = elev.shape

    # Base: water
    r = np.zeros((h, w), dtype=np.float32)
    g = np.zeros((h, w), dtype=np.float32)
    b = np.zeros((h, w), dtype=np.float32)

    # Classify
    water = rivers | (elev < 0.15)

    # Wetlands near water
    distwet = cheap_blur(water.astype(np.float32), k=4)

    # Forest density grows with moisture, mid elevations
    forest_score = moist * (1.0 - np.clip(np.abs(elev - 0.45) / 0.45, 0, 1))

    # Grass/open areas at very low elev or dry places
    grass = (elev < 0.2) | (forest_score < 0.25)

    # Rock at high elevations
    rock = elev > 0.75

    # Colors (linear RGB 0..1)
    # water: deep blue → cyan near shallows
    r = np.where(water, 0.05 + 0.2*distwet, r)
    g = np.where(water, 0.2 + 0.5*distwet, g)
    b = np.where(water, 0.5 + 0.45*distwet, b)

    # grasslands
    r = np.where(grass & (~water), 0.35, r)
    g = np.where(grass & (~water), 0.55, g)
    b = np.where(grass & (~water), 0.25, b)

    # forest: darker greens, vary with density
    forest = (forest_score >= 0.25) & (~water)
    density = np.clip(forest_score, 0, 1)
    r = np.where(forest, 0.08 + 0.10 * density, r)
    g = np.where(forest, 0.25 + 0.55 * density, g)
    b = np.where(forest, 0.08 + 0.10 * density, b)

    # rock/snow caps
    r = np.where(rock & (~water), 0.65, r)
    g = np.where(rock & (~water), 0.62, g)
    b = np.where(rock & (~water), 0.60, b)

    # Return RGB
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 1)

# ------------------------------
# IO & main
# ------------------------------

@dataclass
class Params:
    size: int = 2048
    seed: int = 3
    pixel_size: float = 1.0  # meters per pixel
    river_density: float = 0.003   # fraction of pixels becoming rivers


def save_png(arr: np.ndarray, path: str):
    arr8 = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if arr8.ndim == 2:
        mode = 'L'
    else:
        mode = 'RGB'
    Image.fromarray(arr8, mode=mode).save(path)

def write_geotiff_rgb(rgb: np.ndarray, px: float, path: str):
    if not RASTERIO_OK:
        return False
    h, w, _ = rgb.shape

    # Web Mercator meters; top-left at (0,  h*px). Pixel size = px meters.
    transform = from_origin(0.0, h * px, px, px)

    arr8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=3,
        dtype=rasterio.uint8,
        transform=transform,
        crs="EPSG:3857",     # meters; consistent with px meaning “meters per pixel”
        # Make it Preview/QGIS friendly:
        tiled=False,         # no tiling
        compress="none",     # no compression
        interleave="pixel",  # RGBRGB… per pixel
        photometric="RGB",
        BIGTIFF="NO",
    ) as dst:
        dst.write(arr8[:, :, 0], 1)
        dst.write(arr8[:, :, 1], 2)
        dst.write(arr8[:, :, 2], 3)
    return True




def main(p: Params):
    w = h = p.size
    print(f"Generating DEM {w}x{h} seed={p.seed}…")
    dem = make_dem(w, h, p.seed)

    print("Hydrology (rivers)…")
    rivers = river_mask_from_dem(dem, p.seed+10, target_density=p.river_density)

    print("Moisture & biome…")
    moist = make_moisture(dem, rivers)

    print("Style & shade…")
    hs = shade(dem)
    rgb = palette_forest(dem, moist, rivers)

    # Apply hillshade as lightness multiplier
    rgb = np.clip(rgb * (0.6 + 0.4 * hs[..., None]), 0, 1)

    print("Saving previews (PNG)…")
    save_png(rgb, 'forest_rgb.png')
    save_png(hs, 'dem.png')
    save_png(rivers.astype(np.float32), 'mask_water.png')

    print("Writing GeoTIFF (if rasterio available)…")
    ok = write_geotiff_rgb(rgb, p.pixel_size, 'forest_rgb.tif')
    if ok:
        print("✓ GeoTIFF written: forest_rgb.tif (EPSG:3857, 1 band per color)")
    else:
        print("(rasterio not installed) — skipped GeoTIFF. Install rasterio to enable.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=2048, help='square output size (px)')
    ap.add_argument('--seed', type=int, default=3, help='random seed')
    ap.add_argument('--pixel-size', type=float, default=1.0, help='meters per pixel (GeoTIFF only)')
    ap.add_argument('--river-density', type=float, default=0.003, help='fraction of pixels that become rivers')
    args = ap.parse_args()
    p = Params(size=args.size, seed=args.seed, pixel_size=args.pixel_size, river_density=args.river_density)
    main(p)
