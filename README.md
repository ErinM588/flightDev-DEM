# flightdev — UAV Georeferencing with Terrain-Aware Ray Casting

A Python pipeline that converts UAV camera frames into geo-located map overlays in real time, using platform telemetry, camera field-of-view geometry, and an optional Digital Elevation Model (DEM) for terrain-corrected accuracy.

The project was developed and tested against an **Unreal Engine 5** synthetic island scene driven by **ArduPilot SITL**, but the georeferencing core is simulator-agnostic and works with any MAVLink-compatible platform.

---

## What it does

For every camera frame captured during flight the pipeline:

1. Reads platform pose from a paired telemetry file (position, roll, pitch, yaw, gimbal angles).
2. Computes the four ground-plane corners of the camera's field of view, either against a flat z = 0 plane (fast, approximate) or against a real terrain surface via DEM ray marching (accurate).
3. Builds a homography that maps image pixels to local ground coordinates in meters.
4. Warps the camera frame onto a pre-loaded orthophoto base map using that homography.
5. Draws the footprint polygon and aircraft heading on the annotated map and displays it frame by frame.

---

## Repository layout

```
flightdev/
├── camera_calculator.py      Core ray geometry — FOV rays, rotation, ground intersection
├── georeferencing.py         High-level pipeline: pose + camera → Footprint
├── dem.py                    DEMSampler: load island_dem.tif, sample elevation by local meters
├── main.py                   Runtime loop: load frames, drive georef, warp onto base map
├── params.yaml               All tuneable parameters (camera, image, DEM, stream)
├── pyproject.toml            Python package manifest and dependency list
│
├── island_map.tif            Georeferenced color ortho, EPSG:4326 (WGS84)
├── island_dem.tif            Georeferenced DEM (float32, m above MSL), EPSG:4326
├── IslandMap_NorthUp.png     Source ortho PNG (used as the base map at runtime)
│
├── georeference_terrain.py   Utility: bake island_map.tif and island_dem.tif from
│                             the manually-georeferenced QGIS/ArcGIS outputs
│
└── test_dem_integration.py   Validation script: runs 6 reference frames through both
                              flat-ground and DEM paths, prints comparison table,
                              writes test_comparison.png
```

---

## Setup

**Requires Python ≥ 3.12.**  Install with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
# uv
uv sync

# plain pip
pip install affine dronekit future geopy opencv-python pymavlink pyyaml rasterio vector3d
```

All runtime dependencies are declared in `pyproject.toml`.

---

## Quick start

### Running on a batch of frames

Place paired `frame_NNN.jpg` / `frame_NNN.txt` files under `test_data/`, then:

```bash
python main.py
```

Each frame is displayed on the annotated base map in a window; press any key to advance.

### Validating the DEM integration

```bash
python test_dem_integration.py
```

This runs the 6 bundled reference frames through both the flat-ground and DEM-aware paths and prints a comparison table. It also writes `test_comparison.png` (side-by-side annotated map) and `test_results.txt`.

### Regenerating the GeoTIFF terrain assets

If you replace the source heightmap or ortho, rebuild the GeoTIFFs from the QGIS/ArcGIS-georeferenced inputs:

```bash
python georeference_terrain.py --ortho IslandMap_NorthUp.tif \
                                --heightmap HeightMap_Georef.tif \
                                --qc
```

This writes `island_map.tif`, `island_dem.tif`, and `alignment_qc.png`.

---

## Configuration — `params.yaml`

```yaml
takeoff:
  lat: -35.3633285   # WGS84 decimal degrees — local ENU origin
  lon: -149.1652232  # ⚠ sign should be +149.165... for Canberra SITL — see known issues
  alt: 0.1           # metres above MSL at takeoff

image:
  width_px: 1431     # base map pixel dimensions
  height_px: 1132
  width_m: 2130      # ground extent of the base map
  height_m: 1685
  gsd: 1.49          # ground sample distance, m/px (informational only)

camera:
  hfov: 60           # horizontal field of view, degrees
  aspect: 1.333333   # width / height (used to derive vfov)

stream:
  url: "rtsp://localhost:8554/stream"   # live RTSP feed (unused by batch runner)

dem:
  path: "island_dem.tif"   # float32 GeoTIFF, EPSG:4326
  step_m: 5.0              # ray-march step size in metres; smaller = more accurate, slower
  max_range_m: 10000.0     # give up marching after this distance
  strict: true             # true  → reject whole frame if any corner ray misses the DEM
                           # false → fall back to flat-ground per failed corner
```

Remove or rename the `dem:` block entirely to revert to flat-ground mode with zero code changes.

---

## Coordinate system

All internal geometry uses a **local East-North-Up (ENU) frame** anchored at the takeoff point (red dot on the base map):

- **+x** = east (right on the map)
- **+y** = north (up on the map)
- **+z** = up (altitude in metres)

Telemetry fields `n` (north), `e` (east), and `d` (down, i.e. altitude above home) are used directly. The `alt` field (AMSL metres) is available but not preferred; `d` keeps everything in the local frame.

Converting local metres to decimal degrees uses the spherical-Earth approximation from `georeferencing.offsets_to_decimal_degrees` with the WGS84 equatorial radius (6 378 137 m). This is accurate to better than 0.1 % for scenes up to a few kilometres across.

---

## Core modules

### `camera_calculator.py`

A Python port of [CameraCalculator.java](https://github.com/zelenmi6/thesis/blob/master/src/geometry/CameraCalculator.java). Given camera FOV angles, altitude, roll, pitch, and heading, it computes four corner rays and intersects them with the ground.

**Key public API:**

```python
CameraCalculator.getBoundingPolygon(
    FOVh, FOVv,           # horizontal / vertical FOV, radians
    altitude,             # camera altitude, metres (local z above takeoff)
    roll, pitch, heading, # platform orientation, radians
    *,
    dem_sampler=None,     # DEMSampler instance, or None for flat-ground
    strict=True,          # all-or-nothing failure mode (see DEM section)
    step_m=5.0,           # ray-march step size
    max_range_m=10000.0,  # ray-march cutoff
) -> list[Vector] | None
```

Two ground-intersection paths live inside `findRayGroundIntersection`:

- **Flat-ground** (`dem_sampler=None`): closed-form solve for the ray–plane intersection at z = 0. Original behaviour, preserved byte-for-byte.
- **DEM ray-marching** (`dem_sampler` provided): walks along each downward ray in `step_m` increments, comparing the ray's z position to the terrain elevation sampled from the DEM. On the first crossing, 8 bisection iterations refine the hit to ≈ `step_m / 256` along the ray (< 2 cm with the default 5 m step). Returns `None` if the ray points upward, exits DEM coverage, or exceeds `max_range_m`.

### `georeferencing.py`

Builds the full `Footprint` (corner positions, homography, center point) from a `PlatformPose`, `CameraModel`, and `ImageSize`.

**Key design decisions:**

- When DEM mode is active, the true platform yaw (`π/2 − pose.yaw`) is baked into `rotateRays` so that corner rays arrive at `getBoundingPolygon` already expressed in local ENU. The post-hoc 2-D `Affine.rotation(90 − yaw_deg)` used by the flat-ground path is skipped for DEM mode because the commutativity that makes it correct for a flat plane breaks down over terrain.
- `GeoReferencer.compute()` returns `Optional[Footprint]`. A `None` return means DEM strict mode rejected the polygon; callers (e.g. `main.py`) skip the frame rather than placing a misleading footprint on the map.

```python
georef = GeoReferencer(
    dem_sampler=dem,      # None → flat-ground
    step_m=5.0,
    max_range_m=10000.0,
    strict=True,
)
footprint = georef.compute(pose, camera, image_size)
if footprint is None:
    continue   # strict rejection — all-or-nothing
```

### `dem.py`

`DEMSampler` loads `island_dem.tif` once into memory and exposes:

```python
dem = DEMSampler("island_dem.tif", takeoff_lat=-35.363, takeoff_lon=-149.165)
elevation_m = dem.sample(east_m=200.0, north_m=-400.0)  # → float or None
```

Converts local ENU metres to (lon, lat) using the same spherical-Earth approximation as the rest of the project, applies a bilinear interpolation across the four nearest DEM pixels, and returns `None` for points outside the raster footprint or at NoData cells. The bilinear interpolation gives slightly smoother results than rasterio's default nearest-neighbour sampling (< 0.3 m difference, verified against `rasterio.sample`).

### `main.py`

The runtime loop. For each (image, telemetry) pair:

1. Parses the telemetry key-value file.
2. Combines platform and gimbal angles: `pitch = platform_pitch + gimbal_pitch`.
3. Creates a `PlatformPose` using `d` (height above home) as the altitude.
4. Calls `georef.compute()` and skips the frame on `None`.
5. Converts footprint corners from local metres to base-map pixels.
6. Warps the camera frame onto the base map with `cv2.getPerspectiveTransform` + `cv2.warpPerspective`.
7. Draws the footprint polygon, aircraft position, and heading arrow.

---

## DEM integration — how it improves accuracy

### The problem with flat-ground

The flat-ground path intersects every camera ray with the plane z = 0 regardless of actual terrain height. For a drone at 200 m above a 22 m MSL takeoff point flying over terrain that rises to 282 m MSL, the effective distance from camera to ground is only 135 m in some directions — far shorter than the 200 m the flat-ground formula assumes. This causes the computed footprint to be systematically too large and displaced from the true ground position.

### What the DEM fixes

With the DEM, each corner ray is marched against the actual terrain surface and terminates where it first touches ground. The footprint corners land at the correct geographic position and elevation. Tested against 6 reference frames from a 200 m circular orbit:

| Frame | Position | Flat area | DEM area | Ratio | Center shift |
|-------|----------|-----------|----------|-------|--------------|
| 531   | N+269 E−36   | 56 003 m² | 43 860 m² | 0.783 | 9.5 m  |
| 757   | N+180 E+179  | 56 201 m² | 37 219 m² | 0.662 | 23.5 m |
| 976   | N−48 E+197   | 56 199 m² | 35 237 m² | 0.627 | 26.0 m |
| 1197  | N−169 E±0    | 56 454 m² | 28 620 m² | **0.507** | **42.6 m** |
| 1417  | N−48 E−197   | 56 906 m² | 40 056 m² | 0.704 | 22.0 m |
| 1637  | N+179 E−180  | 56 363 m² | 37 778 m² | 0.670 | 23.7 m |
| **Mean** | | | | **0.659** | **24.6 m** |

Frame 1197 is the most extreme: the camera looks west over the rising interior of the island, where terrain approaches the camera altitude. The flat-ground footprint is nearly twice the correct size and placed 43 m from the true ground position.

### Residual limitation

The homography used for the image warp is still a 4-point projective transform, which assumes the ground between the corners is planar. For frames with large elevation variation across the footprint (e.g. a cliff in the middle of the frame), the interior pixels are still interpolated on a best-fit plane rather than the true terrain surface. Full per-pixel orthorectification would require ray-casting every pixel individually — see the next planned step.

---

## Terrain asset pipeline (`georeference_terrain.py`)

The `island_dem.tif` and `island_map.tif` GeoTIFFs are produced by `georeference_terrain.py` from two source files georeferenced manually in QGIS / ArcGIS:

- `IslandMap_NorthUp.tif` — the color ortho, in the ortho's own pixel-coordinate frame
- `HeightMap_Georef.tif` — the heightmap, georeferenced into the same pixel-coordinate frame

Both source files carry affine transforms that encode the heightmap-to-ortho alignment. `georeference_terrain.py` reads those transforms and uses `rasterio.warp.reproject` to warp the heightmap onto the ortho's exact pixel grid (with a shared dummy CRS, since neither file has a real geographic CRS). It then anchors everything to WGS84 using the red dot in the ortho (the local origin / takeoff position) and the `takeoff.lat` / `takeoff.lon` from `params.yaml`.

**Elevation mapping:** the heightmap pixel values do not span the full 0–255 range. The actual range in the source file (57–232) is mapped linearly: pixel 57 → 0 m (sea level), pixel 232 → 322 m (the scene's peak elevation). This scale (1.84 m per pixel unit) is embedded in the GeoTIFF tags for traceability.

Note: the DEM's valid elevation range within the ortho's geographic footprint is 1.8–282.3 m. The absolute peak (322 m) falls outside the ortho's footprint and therefore outside `island_dem.tif`.

---

## Telemetry format

Each `.txt` file is a simple `key: value` file (one field per line). All numeric values are floats. Required fields:

| Field | Unit | Description |
|-------|------|-------------|
| `n` | m | North offset from local origin |
| `e` | m | East offset from local origin |
| `d` | m | Altitude above home (used as `pose.alt`) |
| `roll` | rad | Platform roll |
| `pitch` | rad | Platform pitch |
| `yaw` | rad | Platform heading (0 = north, positive clockwise) |
| `gimbal_roll` | deg | Gimbal roll offset |
| `gimbal_pitch` | deg | Gimbal pitch offset (e.g. −75 = 75° below level) |
| `gimbal_yaw` | deg | Gimbal yaw offset |

Optional fields (`lat`, `lon`, `alt`, `airspeed`, `radius`) are parsed but not used by the core georeferencing logic.

Platform and gimbal angles are combined before being passed to `GeoReferencer`:
```python
pitch_rad = telemetry["pitch"] + math.radians(telemetry["gimbal_pitch"])
```

---

## Known issues and planned next steps

**Longitude sign in `params.yaml`**
`takeoff.lon` is currently `−149.1652232`. The SITL location near Canberra, Australia uses the positive value `+149.1652232`. The local ENU frame is self-consistent either way, so georeferencing and DEM lookups work correctly. However, any feature that converts local metres to absolute decimal degrees (e.g. annotating detections with lat/lon) will place results in the wrong hemisphere until this is corrected.

**4-point homography vs full orthorectification**
The image warp uses `cv2.getPerspectiveTransform` on the DEM-corrected corners. This fits a planar projection through four points. For frames with significant terrain variation *within* the footprint, a grid-warp approach using `cv2.remap` with a full NxN grid of DEM-sampled points would give better sub-footprint accuracy.

**Altitude semantics**
`pose.alt` is populated from the telemetry `d` field (height above home). The DEM stores elevations in metres above MSL (sea level). These are in the same frame only if the takeoff altitude is at or near MSL — which is true for the current SITL setup (`takeoff.alt: 0.1 m`). If you deploy on a platform that takes off from a significant elevation above sea level, you will need to add the takeoff MSL elevation to `d` before passing it to `GeoReferencer`.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `affine` | ≥ 2.4 | 2D affine transforms for ENU rotation |
| `dronekit` | ≥ 2.9 | MAVLink vehicle interface (live mode) |
| `future` | ≥ 1.0 | dronekit compatibility shim |
| `geopy` | ≥ 2.4 | Geographic utilities |
| `opencv-python` | ≥ 4.13 | Image I/O, homography, warpPerspective |
| `pymavlink` | ≥ 2.4 | MAVLink message parsing |
| `pyyaml` | ≥ 6.0 | params.yaml loading |
| `rasterio` | ≥ 1.3 | GeoTIFF read/write, coordinate transforms |
| `vector3d` | ≥ 1.1 | 3D vector math in camera_calculator |
