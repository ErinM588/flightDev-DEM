"""
dem.py
======

A small, focused DEM sampler for the local-meter coordinate frame used
by the rest of the georeferencing pipeline.

The DEM is read once into memory and lookups happen in local east/north
meters relative to the takeoff origin, returning elevation in meters
above mean sea level (the same datum the DEM was written with).

The spherical-Earth meters-to-degrees conversion intentionally mirrors
``georeferencing.offsets_to_decimal_degrees`` so this module doesn't
have to import from there — keeping the geometry code free of any
rasterio / GIS dependency.

Usage
-----
    >>> from dem import DEMSampler
    >>> dem = DEMSampler("island_dem.tif", takeoff_lat=-35.3633285,
    ...                  takeoff_lon=-149.1652232)
    >>> dem.sample(east_m=0.0, north_m=0.0)
    71.2       # elevation at the takeoff origin, m above MSL
    >>> dem.sample(east_m=10_000.0, north_m=0.0) is None
    True       # outside DEM footprint
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import rasterio


_EARTH_RADIUS_M = 6378137.0  # WGS84 equatorial radius; matches georeferencing.py


@dataclass(frozen=True)
class _CachedDEM:
    """Pre-read DEM data + inverse-affine coefficients for hot-path sampling."""
    data: np.ndarray            # 2D float32 array, shape (height, width)
    width: int
    height: int
    # Inverse affine flattened to scalars: (col, row) = inv_T * (lon, lat).
    # rasterio's Affine maps (col, row) -> (x, y); the inverse maps the other way.
    inv_a: float; inv_b: float; inv_c: float
    inv_d: float; inv_e: float; inv_f: float
    nodata: float               # use ``float('-inf')`` as a sentinel when source has none


class DEMSampler:
    """Sample a georeferenced DEM in local-meter coordinates.

    Parameters
    ----------
    dem_path : str or Path
        A single-band float32 GeoTIFF in EPSG:4326 (the output of
        ``georeference_terrain.py`` is the expected shape).
    takeoff_lat, takeoff_lon : float
        The local origin in decimal degrees. The DEM should have been
        georeferenced relative to this same origin.

    Notes
    -----
    - All lookups are bilinearly interpolated. Points within half a
      pixel of the DEM edge return ``None`` to avoid extrapolating.
    - NoData cells (matching ``ds.nodata`` from the source file) cause
      a ``None`` return. This is what tells the ray-marcher "I have no
      information here — fall back to flat ground or give up".
    - The class assumes a north-up DEM and the spherical-Earth
      meters-to-degrees approximation, both consistent with the rest
      of the project.
    """

    def __init__(
        self,
        dem_path: Union[str, Path],
        takeoff_lat: float,
        takeoff_lon: float,
    ) -> None:
        with rasterio.open(dem_path) as ds:
            data = ds.read(1).astype(np.float32)
            inv = ~ds.transform
            nodata = ds.nodata if ds.nodata is not None else float("-inf")
            self._d = _CachedDEM(
                data=data,
                width=ds.width,
                height=ds.height,
                inv_a=inv.a, inv_b=inv.b, inv_c=inv.c,
                inv_d=inv.d, inv_e=inv.e, inv_f=inv.f,
                nodata=float(nodata),
            )
        self._takeoff_lat = float(takeoff_lat)
        self._takeoff_lon = float(takeoff_lon)
        self._cos_takeoff_lat = math.cos(math.radians(takeoff_lat))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, east_m: float, north_m: float) -> Optional[float]:
        """Return the elevation in meters above MSL at the given local
        offset from the takeoff origin, or ``None`` if the point is
        outside the DEM or hits a NoData cell.

        Bilinearly interpolated between the four enclosing pixel
        centers.
        """
        d = self._d

        # local meters -> (lon, lat) (spherical-Earth approximation)
        lon = self._takeoff_lon + math.degrees(
            east_m / (_EARTH_RADIUS_M * self._cos_takeoff_lat)
        )
        lat = self._takeoff_lat + math.degrees(north_m / _EARTH_RADIUS_M)

        # (lon, lat) -> (col_f, row_f) via inverse affine
        col_f = d.inv_a * lon + d.inv_b * lat + d.inv_c
        row_f = d.inv_d * lon + d.inv_e * lat + d.inv_f

        # Need a full bilinear stencil: col_f in [0, width-1), row_f in [0, height-1).
        if not (0.0 <= col_f < d.width - 1 and 0.0 <= row_f < d.height - 1):
            return None

        c0 = int(col_f)
        r0 = int(row_f)
        fc = col_f - c0
        fr = row_f - r0

        v00 = d.data[r0,     c0    ]
        v01 = d.data[r0,     c0 + 1]
        v10 = d.data[r0 + 1, c0    ]
        v11 = d.data[r0 + 1, c0 + 1]
        if (v00 == d.nodata or v01 == d.nodata
                or v10 == d.nodata or v11 == d.nodata):
            return None

        v_top    = (1.0 - fc) * v00 + fc * v01
        v_bottom = (1.0 - fc) * v10 + fc * v11
        return float((1.0 - fr) * v_top + fr * v_bottom)

    # ------------------------------------------------------------------
    # Optional helpers (not required by ray-marching, but handy for QC)
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of the underlying DEM array."""
        return (self._d.height, self._d.width)

    def bounds_local_m(self) -> tuple[float, float, float, float]:
        """Approximate (east_min, north_min, east_max, north_max) of
        the DEM footprint expressed in local meters from the takeoff
        origin. Useful for diagnostic prints; do not rely on this for
        precise clipping near the equator/poles."""
        d = self._d
        # forward affine on the four corners
        T = (d.inv_a, d.inv_b, d.inv_c, d.inv_d, d.inv_e, d.inv_f)
        # invert the inverse to get the forward transform coefficients
        # rather than carrying both around. Simpler: use rasterio again
        # on the cached values:
        from affine import Affine
        forward = ~Affine(*T)
        lons = []
        lats = []
        for (col, row) in [(0, 0), (d.width, 0), (d.width, d.height), (0, d.height)]:
            x, y = forward * (col, row)
            lons.append(x)
            lats.append(y)
        # convert each corner back to local meters
        easts = []
        norths = []
        for lon, lat in zip(lons, lats):
            de = (lon - self._takeoff_lon) * math.radians(1.0)
            dn = (lat - self._takeoff_lat) * math.radians(1.0)
            easts.append(de * _EARTH_RADIUS_M * self._cos_takeoff_lat)
            norths.append(dn * _EARTH_RADIUS_M)
        return min(easts), min(norths), max(easts), max(norths)
