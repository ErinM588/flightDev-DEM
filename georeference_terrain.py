"""
georeference_terrain.py
========================

Polished pipeline that takes the two manually-georeferenced GeoTIFFs
produced in QGIS/ArcGIS and emits two final, fully WGS84-tagged,
perfectly co-registered outputs:

  island_map.tif : 3-band RGB GeoTIFF, EPSG:4326
  island_dem.tif : 1-band float32 DEM (m above MSL), EPSG:4326, warped
                   onto the ortho's *exact* pixel grid so the two stack
                   pixel-for-pixel.

Why this version is reliable
----------------------------
Earlier iterations tried to infer the heightmap-to-ortho relationship
from image content (phase correlation, chamfer matching of coastlines,
etc.) and the inferred alignment was never quite right. This version
sidesteps inference entirely: the user has already done the alignment
by hand in QGIS/ArcGIS, and the two input GeoTIFFs already encode it
as their internal affine transforms — both expressed in a shared
"ortho-pixel" coordinate frame. We simply:

  1) Read both rasters and their stored transforms.
  2) Use rasterio.warp.reproject to put the heightmap onto the ortho's
     exact grid (the user-supplied transforms tell reproject where each
     heightmap pixel goes in ortho space).
  3) Convert the ortho's "pixel-space" transform into a true WGS84
     geographic transform anchored at the takeoff lat/lon and the red
     dot in the ortho.
  4) Write both outputs with that geographic transform and EPSG:4326.

Elevation mapping
-----------------
The heightmap pixel values are NOT 0–255; the actual range present in
the source file is narrower. The mapping is:

  min observed pixel value  ->  0 m   (sea level)
  max observed pixel value  ->  max_elevation_m (322 m by default)
  interpolation             :  linear

Inputs (defaults in parentheses)
---------------------------------
  --ortho      (IslandMap_NorthUp.tif)   manually-georeferenced color GeoTIFF
  --heightmap  (HeightMap_Georef.tif)    manually-georeferenced height GeoTIFF
  --params     (params.yaml)             takeoff lat/lon + ortho ground extent

Outputs (in --out-dir, default current directory)
-------------------------------------------------
  island_map.tif
  island_dem.tif
  alignment_qc.png   (with --qc)

Run
---
    $ python georeference_terrain.py
    $ python georeference_terrain.py --qc
    $ python georeference_terrain.py --help
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rasterio
import yaml
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6378137.0
DEFAULT_NODATA = -9999.0
DEFAULT_MAX_ELEVATION_M = 322.0

# Both input GeoTIFFs share a coordinate system (the ortho's pixel grid)
# but carry no CRS. reproject() requires *some* CRS for both sides; we
# assign a shared dummy so it relies purely on the stored transforms.
SHARED_FAKE_CRS = "EPSG:3857"

WGS84 = "EPSG:4326"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrthoParams:
    takeoff_lat: float
    takeoff_lon: float
    extent_w_m: float
    extent_h_m: float


@dataclass(frozen=True)
class PipelineResult:
    out_ortho_path: Path
    out_dem_path: Path
    ortho_transform_wgs84: Affine
    ortho_bounds_wgs84: Tuple[float, float, float, float]
    ortho_gsd_m: Tuple[float, float]
    red_dot_row_col: Tuple[int, int]
    valid_dem_fraction: float
    elevation_range_m: Tuple[float, float]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def find_reddest_pixel(img_bgr: np.ndarray) -> Tuple[int, int]:
    """Center of the reddest 3x3 region in a BGR image.
    Vectorized equivalent of georeferencing.find_reddest_kernel.
    Returns (row, col).
    """
    b = img_bgr[:, :, 0].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)
    redness = r - 0.5 * (b + g)
    sums = cv2.boxFilter(redness, ddepth=-1, ksize=(3, 3), normalize=False)
    cy, cx = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(cy), int(cx)


def offsets_to_lonlat(
    takeoff_lat: float, takeoff_lon: float,
    east_m: float, north_m: float,
) -> Tuple[float, float]:
    """Local east/north meters -> (lon, lat). Spherical-Earth approx,
    consistent with georeferencing.offsets_to_decimal_degrees.
    Returns (lon, lat) in rasterio's (x, y) order.
    """
    lat_rad = math.radians(takeoff_lat)
    lon_rad = math.radians(takeoff_lon)
    dlat = north_m / EARTH_RADIUS_M
    dlon = east_m / (EARTH_RADIUS_M * math.cos(lat_rad))
    return math.degrees(lon_rad + dlon), math.degrees(lat_rad + dlat)


def build_ortho_wgs84_transform(
    params: OrthoParams, ortho_w_px: int, ortho_h_px: int,
    red_dot_row: int, red_dot_col: int,
) -> Tuple[Affine, Tuple[float, float]]:
    """Build a WGS84 Affine for the ortho, anchored on the red dot pixel.
    Returns (transform, (gsd_x_m, gsd_y_m)).
    """
    gsd_x = params.extent_w_m / ortho_w_px
    gsd_y = params.extent_h_m / ortho_h_px

    nw_east_m = (0 - red_dot_col) * gsd_x
    nw_north_m = -(0 - red_dot_row) * gsd_y
    nw_lon, nw_lat = offsets_to_lonlat(
        params.takeoff_lat, params.takeoff_lon, nw_east_m, nw_north_m,
    )

    deg_per_m_lat = math.degrees(1.0 / EARTH_RADIUS_M)
    deg_per_m_lon = math.degrees(
        1.0 / (EARTH_RADIUS_M * math.cos(math.radians(params.takeoff_lat)))
    )
    pixel_size_lon = gsd_x * deg_per_m_lon
    pixel_size_lat = gsd_y * deg_per_m_lat
    T = Affine(pixel_size_lon, 0.0, nw_lon, 0.0, -pixel_size_lat, nw_lat)
    return T, (gsd_x, gsd_y)


def heightmap_to_elevation_m(
    hm_band: np.ndarray, max_elevation_m: float,
) -> Tuple[np.ndarray, float, float, float]:
    """Linearly map heightmap pixel intensities to meters above MSL.

    The heightmap is NOT assumed to span 0..255. The actual minimum
    pixel value is treated as sea level (0 m) and the actual maximum
    as max_elevation_m. Linear interpolation in between.

    Returns (elevation_array, hm_min, hm_max, scale_m_per_unit).
    """
    hm_min = float(hm_band.min())
    hm_max = float(hm_band.max())
    if hm_max == hm_min:
        return np.zeros_like(hm_band, dtype=np.float32), hm_min, hm_max, 0.0
    scale = max_elevation_m / (hm_max - hm_min)
    elev = (hm_band.astype(np.float32) - hm_min) * scale
    return elev, hm_min, hm_max, scale


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_params(path: Path) -> OrthoParams:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return OrthoParams(
        takeoff_lat=float(data["takeoff"]["lat"]),
        takeoff_lon=float(data["takeoff"]["lon"]),
        extent_w_m=float(data["image"]["width_m"]),
        extent_h_m=float(data["image"]["height_m"]),
    )


def read_ortho(path: Path) -> Tuple[np.ndarray, Affine, int, int]:
    """Read a (potentially 4-band uint16) color GeoTIFF into uint8 RGB."""
    with rasterio.open(path) as ds:
        arr = ds.read(); T = ds.transform; h, w = ds.height, ds.width
    rgb = arr[:3]
    if rgb.dtype != np.uint8:
        maxv = rgb.max()
        rgb = (rgb.astype(np.float32) / max(1, maxv) * 255.0).astype(np.uint8)
    return rgb.transpose(1, 2, 0), T, h, w


def read_heightmap_band(path: Path) -> Tuple[np.ndarray, Affine, int, int]:
    with rasterio.open(path) as ds:
        arr = ds.read(); T = ds.transform; h, w = ds.height, ds.width
    return arr[0].astype(np.float32), T, h, w


def write_ortho_geotiff(
    rgb_hwc: np.ndarray, transform: Affine, path: Path,
    tags: Optional[dict] = None,
) -> None:
    h, w = rgb_hwc.shape[:2]
    bands = rgb_hwc.transpose(2, 0, 1).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=3, dtype="uint8",
        crs=WGS84, transform=transform, photometric="RGB",
        compress="lzw", tiled=False,
    ) as dst:
        dst.write(bands)
        for i, name in enumerate(("Red", "Green", "Blue"), start=1):
            dst.set_band_description(i, name)
        if tags:
            dst.update_tags(**tags)


def write_dem_geotiff(
    elev: np.ndarray, transform: Affine, path: Path,
    nodata: float = DEFAULT_NODATA, tags: Optional[dict] = None,
) -> None:
    h, w = elev.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
        crs=WGS84, transform=transform, nodata=nodata,
        compress="lzw", tiled=False,
    ) as dst:
        dst.write(elev.astype(np.float32), 1)
        dst.set_band_description(1, "Elevation (m above MSL)")
        if tags:
            dst.update_tags(**tags)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    ortho_path: Path, heightmap_path: Path, params_path: Path,
    out_dir: Path,
    max_elevation_m: float = DEFAULT_MAX_ELEVATION_M,
    nodata: float = DEFAULT_NODATA,
    resampling: Resampling = Resampling.bilinear,
    qc: bool = False,
    verbose: bool = True,
) -> PipelineResult:

    def _log(msg: str) -> None:
        if verbose: print(msg)

    out_dir.mkdir(parents=True, exist_ok=True)

    _log("Loading inputs...")
    params = load_params(params_path)
    ortho_rgb, ortho_T_src, oh, ow = read_ortho(ortho_path)
    hm_band, hm_T_src, hh, hw = read_heightmap_band(heightmap_path)
    _log(f"  ortho:    {ow} x {oh}   transform: "
         f"{tuple(round(v, 4) for v in ortho_T_src[:6])}")
    _log(f"  heightmap:{hw} x {hh}   transform: "
         f"{tuple(round(v, 4) for v in hm_T_src[:6])}")

    ortho_bgr = cv2.cvtColor(ortho_rgb, cv2.COLOR_RGB2BGR)
    red_row, red_col = find_reddest_pixel(ortho_bgr)
    _log(f"  red dot at ortho pixel (row={red_row}, col={red_col})")

    ortho_T_wgs, (gsd_x, gsd_y) = build_ortho_wgs84_transform(
        params, ow, oh, red_row, red_col,
    )
    _log(f"  ortho GSD: {gsd_x:.4f} x {gsd_y:.4f} m/px")

    elev_src, hm_min, hm_max, scale_m = heightmap_to_elevation_m(
        hm_band, max_elevation_m,
    )
    _log(f"  elevation mapping:")
    _log(f"    pixel min = {int(hm_min)}  -> 0 m  (sea level)")
    _log(f"    pixel max = {int(hm_max)}  -> {max_elevation_m:.1f} m")
    _log(f"    scale     = {scale_m:.6f} m / pixel-unit")
    _log(f"    source elevation range (full heightmap): "
         f"[{float(elev_src.min()):.2f}, {float(elev_src.max()):.2f}] m")

    elev_aligned = np.full((oh, ow), nodata, dtype=np.float32)
    reproject(
        source=elev_src,
        destination=elev_aligned,
        src_transform=hm_T_src, src_crs=SHARED_FAKE_CRS,
        dst_transform=ortho_T_src, dst_crs=SHARED_FAKE_CRS,
        src_nodata=None, dst_nodata=nodata,
        resampling=resampling,
    )
    valid_mask = elev_aligned != nodata
    valid_frac = float(valid_mask.sum() / elev_aligned.size)
    valid_min = float(elev_aligned[valid_mask].min()) if valid_mask.any() else 0.0
    valid_max = float(elev_aligned[valid_mask].max()) if valid_mask.any() else 0.0
    _log(f"  warped to ortho grid ({resampling.name} resampling):")
    _log(f"    coverage  : {valid_frac*100:.1f}% of ortho pixels")
    _log(f"    range     : [{valid_min:.2f}, {valid_max:.2f}] m")
    if valid_max < max_elevation_m - 1 or valid_min > 1:
        _log(f"    (range narrower than source [0, {max_elevation_m:.0f}] m "
             "because the absolute peak / sea-level pixels lie outside "
             "the ortho's geographic footprint)")

    out_ortho = out_dir / "island_map.tif"
    write_ortho_geotiff(ortho_rgb, ortho_T_wgs, out_ortho, tags={
        "SOURCE_IMAGE": str(ortho_path),
        "RED_DOT_PIXEL": f"row={red_row},col={red_col}",
        "TAKEOFF_LATLON": f"{params.takeoff_lat},{params.takeoff_lon}",
        "GSD_X_M": f"{gsd_x:.6f}",
        "GSD_Y_M": f"{gsd_y:.6f}",
    })
    _log(f"  wrote {out_ortho}")

    out_dem = out_dir / "island_dem.tif"
    write_dem_geotiff(elev_aligned, ortho_T_wgs, out_dem, nodata=nodata, tags={
        "SOURCE_HEIGHTMAP": str(heightmap_path),
        "ALIGNED_TO": out_ortho.name,
        "ELEVATION_UNITS": "meters above MSL",
        "MAX_ELEVATION_M": f"{max_elevation_m}",
        "ELEVATION_MAPPING": (
            f"pixel {int(hm_min)} -> 0 m (sea level); "
            f"pixel {int(hm_max)} -> {max_elevation_m:.1f} m; "
            f"linear; {scale_m:.6f} m/unit"
        ),
        "RESAMPLING": resampling.name,
    })
    _log(f"  wrote {out_dem}")

    left, top = ortho_T_wgs * (0, 0)
    right, bottom = ortho_T_wgs * (ow, oh)
    ortho_bounds = (left, bottom, right, top)

    if qc:
        qc_path = out_dir / "alignment_qc.png"
        _write_qc(ortho_rgb, elev_aligned, nodata, qc_path)
        _log(f"  wrote {qc_path}")

    return PipelineResult(
        out_ortho_path=out_ortho,
        out_dem_path=out_dem,
        ortho_transform_wgs84=ortho_T_wgs,
        ortho_bounds_wgs84=ortho_bounds,
        ortho_gsd_m=(gsd_x, gsd_y),
        red_dot_row_col=(red_row, red_col),
        valid_dem_fraction=valid_frac,
        elevation_range_m=(valid_min, valid_max),
    )


def _write_qc(
    ortho_rgb_hwc: np.ndarray, elev_aligned: np.ndarray,
    nodata: float, path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    masked = np.ma.masked_equal(elev_aligned, nodata)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(ortho_rgb_hwc); axes[0].set_title("Ortho (input)"); axes[0].axis("off")
    im = axes[1].imshow(masked, cmap="terrain")
    axes[1].set_title("Aligned DEM (m above MSL)"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.04, label="m")
    axes[2].imshow(ortho_rgb_hwc)
    axes[2].imshow(masked, cmap="terrain", alpha=0.55)
    axes[2].set_title("Overlay (55%)"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> int:
    p = argparse.ArgumentParser(
        description="Produce co-registered WGS84 GeoTIFFs from the "
                    "user-supplied georeferenced ortho and heightmap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ortho",     type=Path, default=Path("IslandMap_NorthUp.tif"))
    p.add_argument("--heightmap", type=Path, default=Path("HeightMap_Georef.tif"))
    p.add_argument("--params",    type=Path, default=Path("params.yaml"))
    p.add_argument("--out-dir",   type=Path, default=Path("."))
    p.add_argument("--max-elevation", type=float, default=DEFAULT_MAX_ELEVATION_M,
                   help="Meters above MSL represented by the heightmap's max pixel.")
    p.add_argument("--nodata",    type=float, default=DEFAULT_NODATA)
    p.add_argument("--resampling",
                   choices=["bilinear", "nearest", "cubic"], default="bilinear",
                   help="Resampling method when warping heightmap onto ortho grid.")
    p.add_argument("--qc",        action="store_true",
                   help="Write alignment_qc.png for visual inspection.")
    p.add_argument("--quiet",     action="store_true")
    args = p.parse_args()

    for f in (args.ortho, args.heightmap, args.params):
        if not f.exists():
            p.error(f"Input file not found: {f}")

    resampling_map = {
        "bilinear": Resampling.bilinear,
        "nearest":  Resampling.nearest,
        "cubic":    Resampling.cubic,
    }
    result = run_pipeline(
        ortho_path=args.ortho,
        heightmap_path=args.heightmap,
        params_path=args.params,
        out_dir=args.out_dir,
        max_elevation_m=args.max_elevation,
        nodata=args.nodata,
        resampling=resampling_map[args.resampling],
        qc=args.qc,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  ortho out        : {result.out_ortho_path}")
        print(f"  DEM out          : {result.out_dem_path}")
        print(f"  red dot pixel    : row={result.red_dot_row_col[0]}, "
              f"col={result.red_dot_row_col[1]}")
        print(f"  ortho GSD        : {result.ortho_gsd_m[0]:.4f} x "
              f"{result.ortho_gsd_m[1]:.4f} m/px")
        l, b, r, t = result.ortho_bounds_wgs84
        print(f"  ortho WGS84 bbox : ({l:.6f}, {b:.6f}, {r:.6f}, {t:.6f})")
        print(f"  DEM coverage     : {result.valid_dem_fraction*100:.1f}%")
        print(f"  elevation range  : {result.elevation_range_m[0]:.1f} - "
              f"{result.elevation_range_m[1]:.1f} m")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
