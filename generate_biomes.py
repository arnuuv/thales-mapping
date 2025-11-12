#!/usr/bin/env python3
"""
Procedural fictitious biome generator (FOREST / MOUNTAIN)
Fast, detailed, QGIS‑friendly layered outputs written to separate folders.

Usage examples
--------------
# Forest, 2048 px, quick
python generate_biomes.py --biome forest --size 2048 --seed 7 --outdir output/forest

# Mountain, 2048 px
python generate_biomes.py --biome mountain --size 2048 --seed 11 --outdir output/mountain

What you get in outdir/
-----------------------
- rgb.tif            (3‑band baseline GeoTIFF, EPSG:3857)
- dem.tif            (1‑band DEM 0..1)
- hillshade.tif      (1‑band 0..1)
- slope.tif          (1‑band 0..1)
- water.tif          (1‑band water mask 0/1)
- cover.tif          (1‑band landcover classes: see legend below)
- preview.png        (styled PNG for quick look)

Landcover legend (cover.tif)
----------------------------
FOREST biome:
0=water, 1=grass, 2=forest_light, 3=forest_dense, 4=rock/snow

MOUNTAIN biome:
0=water, 1=alpine_meadow, 2=subalpine_forest, 3=bare_rock, 4=snow/ice

Notes
-----
* No GDAL dependency; only numpy, pillow, rasterio.
* Rivers via fast D8 flow accumulation (topological pass).
* Extra micro‑detail added with high‑freq noise + ambient light.
* Baseline GeoTIFF (no tiling/compression) for macOS Preview and QGIS.
"""
from __future__ import annotations
import argparse
import math
from collections import deque
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

# ------------------ Noise & helpers ------------------

def smoothstep(t):
    return t * t * (3 - 2 * t)


def value_noise(width: int, height: int, grid: int, rng: np.random.Generator) -> np.ndarray:
    gw = (width // grid) + 2
    gh = (height // grid) + 2
    g = rng.random((gh, gw), dtype=np.float32)
    y = np.arange(height, dtype=np.float32)[:, None] / grid
    x = np.arange(width, dtype=np.float32)[None, :] / grid
    yi = np.floor(y).astype(int)
    xi = np.floor(x).astype(int)
    fy = smoothstep(y - yi)
    fx = smoothstep(x - xi)
    g00 = g[yi, xi]; g10 = g[yi, xi+1]; g01 = g[yi+1, xi]; g11 = g[yi+1, xi+1]
    n0 = g00*(1-fx) + g10*fx
    n1 = g01*(1-fx) + g11*fx
    return (n0*(1-fy) + n1*fy).astype(np.float32)


def fbm(w: int, h: int, octaves=6, lacunarity=2.0, gain=0.5, base_grid=128, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    freq = 1.0
    for _ in range(octaves):
        grid = max(2, int(base_grid / freq))
        total += value_noise(w, h, grid, rng) * amp
        amp *= gain
        freq *= lacunarity
    total /= (1.0 - gain**octaves) / (1.0 - gain) if gain != 1 else octaves
    return total

# Ridged multifractal for sharp mountains

def ridged_fbm(w: int, h: int, octaves=6, base_grid=128, seed=0) -> np.ndarray:
    n = fbm(w, h, octaves=octaves, base_grid=base_grid, seed=seed)
    ridged = 1.0 - np.abs(2*n - 1.0)  # peaks become bright
    for k in range(2):  # add a bit more shape
        ridged = np.clip(0.6*ridged + 0.4*(1.0 - np.abs(2*fbm(w,h,octaves=4,base_grid=base_grid//2,seed=seed+100+k)-1.0)), 0, 1)
    return ridged

# ------------------ Terrain & hydrology ------------------

def make_dem(biome: str, w: int, h: int, seed: int) -> np.ndarray:
    if biome == 'mountain':
        base = ridged_fbm(w, h, octaves=7, base_grid=256, seed=seed)
        macro = ridged_fbm(w, h, octaves=4, base_grid=512, seed=seed+1)
        dem = 0.7*base + 0.3*macro
        # emphasize relief
        dem = dem**1.1
    else:  # forest
        base = fbm(w, h, octaves=6, base_grid=256, seed=seed)
        detail = fbm(w, h, octaves=5, base_grid=64, seed=seed+1)
        dem = 0.65*base + 0.35*detail

    # region mask so edges sink → coastlines
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w/2, h/2
    rad = np.sqrt(((xx-cx)/(0.62*w))**2 + ((yy-cy)/(0.62*h))**2)
    bowl = 1 - np.clip(rad, 0, 1)**2
    dem = 0.82*dem + 0.18*bowl

    # add micro detail
    micro = fbm(w, h, octaves=3, base_grid=24, gain=0.6, seed=seed+77)
    dem = np.clip(dem + 0.06*(micro-0.5), 0, 1)
    dem = (dem - dem.min())/(dem.max()-dem.min()+1e-6)
    return dem.astype(np.float32)


def d8_flow(dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = dem.shape
    offs = np.array([(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)], dtype=np.int32)
    dir_idx = -np.ones((h, w), dtype=np.int8)
    slope = np.zeros((h, w), dtype=np.float32)
    for i in range(1, h-1):
        zi = dem[i]
        for j in range(1, w-1):
            z = zi[j]
            best = 0.0; best_k = -1
            # manual unroll for speed
            for k in range(8):
                di, dj = offs[k]; dz = z - dem[i+di, j+dj]
                if dz > best: best = dz; best_k = k
            dir_idx[i, j] = best_k
            slope[i, j] = best
    return dir_idx, slope


def flow_accumulation(dir_idx: np.ndarray) -> np.ndarray:
    h, w = dir_idx.shape
    offs = np.array([(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)], dtype=np.int32)
    downstream = np.full((h, w, 2), -1, dtype=np.int32)
    indeg = np.zeros((h, w), dtype=np.int32)
    for i in range(1, h-1):
        for j in range(1, w-1):
            k = dir_idx[i, j]
            if k >= 0:
                di, dj = offs[k]; ti, tj = i+di, j+dj
                downstream[i, j, 0] = ti; downstream[i, j, 1] = tj
                indeg[ti, tj] += 1
    acc = np.ones((h, w), dtype=np.float32)
    q = deque()
    for i in range(1, h-1):
        for j in range(1, w-1):
            if indeg[i, j] == 0 and dir_idx[i, j] != -1:
                q.append((i, j))
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
    if k <= 1: return img
    out = img
    for _ in range(k):
        out = (np.roll(out,1,0)+out+np.roll(out,-1,0))/3.0
        out = (np.roll(out,1,1)+out+np.roll(out,-1,1))/3.0
    return out


def river_mask(dem: np.ndarray, target_density=0.004) -> np.ndarray:
    dir_idx, _ = d8_flow(dem)
    acc = flow_accumulation(dir_idx)
    acc = (acc - acc.min())/(acc.max()-acc.min()+1e-6)
    thr = np.quantile(acc, 1.0 - target_density)
    rivers = acc >= thr
    rivers = cheap_blur(rivers.astype(np.float32), k=2) > 0.25
    rivers &= (dem < 0.88)
    return rivers

# ------------------ Biome styling ------------------

def hillshade(dem: np.ndarray, az=315.0, alt=45.0) -> np.ndarray:
    azr = math.radians(az); altr = math.radians(alt)
    gy, gx = np.gradient(dem)
    slope = np.hypot(gx, gy)
    aspect = np.arctan2(-gx, gy)
    hs = np.sin(altr)*(1-slope) + np.cos(altr)*slope*np.cos(azr - aspect)
    return np.clip(hs, 0, 1)


def slope_norm(dem: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(dem)
    s = np.hypot(gx, gy)
    return np.clip(s / (s.max() + 1e-6), 0, 1)


def palette_forest(dem, moisture, water) -> Tuple[np.ndarray, np.ndarray]:
    h, w = dem.shape
    rgb = np.zeros((h,w,3), np.float32)
    cover = np.zeros((h,w), np.uint8)

    forest_score = moisture * (1.0 - np.clip(np.abs(dem - 0.45)/0.45, 0, 1))
    grass = (dem < 0.22) | (forest_score < 0.22)
    forest_light = (forest_score >= 0.22) & (forest_score < 0.55) & ~water
    forest_dense = (forest_score >= 0.55) & ~water
    rock = (dem > 0.78) & ~water

    # water first
    rgb[...,0] = np.where(water, 0.05, 0)
    rgb[...,1] = np.where(water, 0.40, 0)
    rgb[...,2] = np.where(water, 0.80, 0)
    cover = np.where(water, 0, cover)

    # grass
    rgb[...,0] = np.where(grass & ~water, 0.30, rgb[...,0])
    rgb[...,1] = np.where(grass & ~water, 0.55, rgb[...,1])
    rgb[...,2] = np.where(grass & ~water, 0.20, rgb[...,2])
    cover = np.where(grass & ~water, 1, cover)

    # forest light/dense
    dens = np.clip(forest_score, 0, 1)
    for_mask = forest_light | forest_dense
    rgb[...,0] = np.where(for_mask, 0.08 + 0.10*dens, rgb[...,0])
    rgb[...,1] = np.where(for_mask, 0.25 + 0.55*dens, rgb[...,1])
    rgb[...,2] = np.where(for_mask, 0.08 + 0.10*dens, rgb[...,2])
    cover = np.where(forest_light, 2, cover)
    cover = np.where(forest_dense, 3, cover)

    # rock / snow
    rgb[...,0] = np.where(rock, 0.65, rgb[...,0])
    rgb[...,1] = np.where(rock, 0.62, rgb[...,1])
    rgb[...,2] = np.where(rock, 0.60, rgb[...,2])
    cover = np.where(rock, 4, cover)
    return rgb, cover


def palette_mountain(dem, moisture, water) -> Tuple[np.ndarray, np.ndarray]:
    h, w = dem.shape
    rgb = np.zeros((h,w,3), np.float32)
    cover = np.zeros((h,w), np.uint8)

    # zones
    snow = (dem > 0.82)
    bare = (dem > 0.68) & ~snow
    subalpine = (dem > 0.48) & ~bare & ~snow
    meadow = (dem <= 0.48) & ~water

    # water
    rgb[...,0] = np.where(water, 0.06, 0)
    rgb[...,1] = np.where(water, 0.42, 0)
    rgb[...,2] = np.where(water, 0.78, 0)
    cover = np.where(water, 0, cover)

    # meadow
    rgb[...,0] = np.where(meadow, 0.36, rgb[...,0])
    rgb[...,1] = np.where(meadow, 0.62, rgb[...,1])
    rgb[...,2] = np.where(meadow, 0.28, rgb[...,2])
    cover = np.where(meadow, 1, cover)

    # subalpine forest
    rgb[...,0] = np.where(subalpine, 0.15, rgb[...,0])
    rgb[...,1] = np.where(subalpine, 0.42, rgb[...,1])
    rgb[...,2] = np.where(subalpine, 0.18, rgb[...,2])
    cover = np.where(subalpine, 2, cover)

    # bare rock
    rgb[...,0] = np.where(bare, 0.55, rgb[...,0])
    rgb[...,1] = np.where(bare, 0.52, rgb[...,1])
    rgb[...,2] = np.where(bare, 0.50, rgb[...,2])
    cover = np.where(bare, 3, cover)

    # snow/ice
    rgb[...,0] = np.where(snow, 0.95, rgb[...,0])
    rgb[...,1] = np.where(snow, 0.95, rgb[...,1])
    rgb[...,2] = np.where(snow, 0.97, rgb[...,2])
    cover = np.where(snow, 4, cover)

    return rgb, cover

# ------------------ IO ------------------

def ensure_outdir(path: str):
    import os
    os.makedirs(path, exist_ok=True)


def write_geotiff(data: np.ndarray, px: float, path: str, bands=1, photometric=None):
    if not RASTERIO_OK:
        return False
    import rasterio
    h, w = (data.shape[0], data.shape[1]) if bands==1 else (data.shape[0], data.shape[1])
    transform = from_origin(0.0, h*px, px, px)
    kwargs = dict(
        driver='GTiff', height=h, width=w, dtype=rasterio.uint8,
        transform=transform, crs='EPSG:3857', tiled=False, compress='none', BIGTIFF='NO'
    )
    if bands == 1:
        count=1
    else:
        count=data.shape[2]
    kwargs['count'] = count
    if photometric:
        kwargs['photometric'] = photometric
    with rasterio.open(path, 'w', **kwargs) as dst:
        if bands == 1:
            if data.dtype != np.uint8:
                arr = np.clip(data*255.0+0.5, 0, 255).astype(np.uint8)
            else:
                arr = data
            dst.write(arr, 1)
        else:
            arr = np.clip(data*255.0+0.5, 0, 255).astype(np.uint8)
            for i in range(arr.shape[2]):
                dst.write(arr[:,:,i], i+1)
    return True


def save_png(rgb: np.ndarray, path: str):
    Image.fromarray(np.clip(rgb*255+0.5,0,255).astype(np.uint8), 'RGB').save(path)

# ------------------ Main ------------------

@dataclass
class Params:
    biome: str
    size: int
    seed: int
    pixel_size: float
    river_density: float
    outdir: str


def main(p: Params):
    import os
    ensure_outdir(p.outdir)
    w = h = p.size

    print(f"Biome={p.biome} size={w} seed={p.seed}")
    dem = make_dem(p.biome, w, h, p.seed)
    rivers = river_mask(dem, target_density=p.river_density)

    # moisture proxy: blurred rivers + windward slopes
    moist = cheap_blur(rivers.astype(np.float32), k=6)
    gx = np.gradient(dem, axis=1)
    wind = np.clip(-gx, 0, None)
    wind = (wind - wind.min())/(wind.max()-wind.min()+1e-6)
    moisture = np.clip(0.6*moist + 0.4*wind, 0, 1)

    hs = hillshade(dem)
    sl = slope_norm(dem)

    if p.biome == 'mountain':
        rgb, cover = palette_mountain(dem, moisture, rivers)
    else:
        rgb, cover = palette_forest(dem, moisture, rivers)

    # Apply light
    rgb = np.clip(rgb*(0.62 + 0.38*hs[...,None]), 0, 1)

    # Write layers
    write_geotiff(rgb, p.pixel_size, os.path.join(p.outdir, 'rgb.tif'), bands=3, photometric='RGB')
    write_geotiff(dem, p.pixel_size, os.path.join(p.outdir, 'dem.tif'))
    write_geotiff(hs,  p.pixel_size, os.path.join(p.outdir, 'hillshade.tif'))
    write_geotiff(sl,  p.pixel_size, os.path.join(p.outdir, 'slope.tif'))
    write_geotiff(rivers.astype(np.float32), p.pixel_size, os.path.join(p.outdir, 'water.tif'))
    write_geotiff(cover.astype(np.uint8), p.pixel_size, os.path.join(p.outdir, 'cover.tif'))

    # Preview PNG
    save_png(rgb, os.path.join(p.outdir, 'preview.png'))

    print(f"✓ Wrote layered outputs to {p.outdir}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--biome', choices=['forest','mountain'], required=True)
    ap.add_argument('--size', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--pixel-size', type=float, default=1.0, help='meters per pixel in EPSG:3857')
    ap.add_argument('--river-density', type=float, default=0.004)
    ap.add_argument('--outdir', type=str, default='output/biome')
    args = ap.parse_args()

    p = Params(biome=args.biome, size=args.size, seed=args.seed, pixel_size=args.pixel_size,
               river_density=args.river_density, outdir=args.outdir)
    main(p)
