# flightdev — UAV Georeferencing with DEM-Aware Ray Casting

A Python pipeline that georeferences UAV camera frames against a Digital Elevation Model (DEM), computing terrain-accurate ground footprints and projecting camera imagery onto a base map in real time.

Developed and validated against an **Unreal Engine 5** synthetic island scene driven by **ArduPilot SITL**. The georeferencing core is simulator-agnostic and works with any MAVLink-compatible platform.

---

## Package contents

```
flightdev_dem_final/
├── README.md
├── pyproject.toml                 dependency manifest
├── params.yaml                    all runtime configuration
│
├── camera_calculator.py           ray geometry — FOV, rotation, flat-ground and DEM intersection
├── georeferencing.py              pose + camera → Footprint (corners, homography, center)
├── dem.py                         DEMSampler: bilinear elevation lookup in local ENU metres
├── main.py                        runtime loop: telemetry → footprint → base-map overlay
├── georeference_terrain.py        utility: bake island_map.tif / island_dem.tif from source data
│
├── IslandMap_NorthUp.png          base map (required at runtime by main.py and test script)
├── island_map.tif                 georeferenced colour ortho, EPSG:4326
├── island_dem.tif                 georeferenced DEM, float32 m above MSL, EPSG:4326
│
├── test_dem_integration.py        validation: flat-ground vs DEM over 6 reference frames
├── test_comparison.png            pre-computed side-by-side result of that validation
├── test_results.txt               per-frame metrics (area ratio, centre displacement)
│
└── test_data/
    ├── frame_531.jpg  …  frame_1637.jpg    six reference frames from a 200 m orbital pass
    └── frame_531.txt  …  frame_1637.txt    paired telemetry for each frame
```

---

## Setup

**Requires Python ≥ 3.12.** Install with [uv](https://docs.astral.sh/uv/) or pip:

```bash
# uv (recommended — reads pyproject.toml automatically)
uv sync

# plain pip
pip install affine dronekit future geopy opencv-python pymavlink pyyaml rasterio vector3d
```

---

## Running

### Batch processing your own frames

Place paired `frame_NNN.jpg` / `frame_NNN.txt` files in `test_data/`, then run from the package root:

```bash
python main.py
```

Each frame is displayed as it is processed; press any key to advance. The annotated base map accumulates all footprints across the run.

### Reproducing the validation results

```bash
python test_dem_integration.py
```

Runs the six bundled reference frames through both the flat-ground and DEM paths, prints a comparison table, and writes `test_comparison.png` and `test_results.txt`. The pre-computed versions of both outputs are already included in the package for inspection without re-running.

---

## Configuration — `params.yaml`

```yaml
takeoff:
  lat: -35.3633285    # WGS84 decimal degrees — anchor for the local ENU frame
  lon: -149.1652232   # ⚠ should be +149.165... for Canberra SITL (see Known issues)
  alt: 0.1            # metres above MSL at takeoff

image:
  width_px: 1431      # base map pixel dimensions (IslandMap_NorthUp.png)
  height_px: 1132
  width_m:  2130      # ground extent of the base map in metres
  height_m: 1685

camera:
  hfov: 60            # horizontal field of view, degrees
  aspect: 1.333333    # width / height aspect ratio (used to derive vfov)

stream:
  url: "rtsp://localhost:8554/stream"   # live RTSP source (not used by batch runner)

dem:
  path: "island_dem.tif"   # path to the float32 GeoTIFF DEM, relative to working directory
  step_m: 5.0              # ray-march step in metres — smaller is more accurate but slower
  max_range_m: 10000.0     # abandon a ray after this distance
  strict: true             # true  → skip frame if any corner ray misses the DEM
                           # false → fall back to flat-ground per failed corner
```

**Disabling the DEM:** remove or comment out the `dem:` block. `main.py` falls back to flat-ground automatically with no other changes required.

---

## How it works

### Coordinate system

All geometry uses a **local East-North-Up (ENU) frame** anchored at the takeoff point (the red dot on `IslandMap_NorthUp.png`):

| Axis | Direction |
|------|-----------|
| +x | East |
| +y | North |
| +z | Up (altitude in metres) |

Telemetry fields `n`, `e`, and `d` (height above home) feed directly into this frame. The `alt` field (AMSL) is available but not used; `d` is preferred because it keeps everything relative to the local origin.

### Georeferencing pipeline

For each frame, `georeferencing.py` builds a `Footprint`:

1. Converts combined platform + gimbal angles into a `PlatformPose`.
2. Passes pose and camera FOV to `CameraCalculator.getBoundingPolygon`, which rotates four corner rays into local ENU and intersects each with the ground.
3. With DEM active: each ray is marched in `step_m` increments; the first terrain crossing is refined with 8 bisection iterations (≈ 2 cm precision at default settings). The true platform yaw is baked into the 3D rotation matrix so rays arrive at the DEM already expressed in ENU.
4. With DEM disabled: the original closed-form ray–plane intersection at z = 0 is used. The flat-ground code path is preserved byte-for-byte.
5. The four ground corners are used to build a homography mapping image pixels to local ENU metres.

`GeoReferencer.compute()` returns `Optional[Footprint]`. A `None` return means `strict=True` and at least one corner ray missed the DEM; `main.py` skips that frame.

### Map projection

`main.py` converts each footprint corner from local ENU metres to base-map pixels using the ground extents in `params.yaml`, then uses `cv2.getPerspectiveTransform` + `cv2.warpPerspective` to project the camera frame onto the base map. This is an approximate rectification — exact at the four DEM-corrected corners, with planar interpolation between them. Frames with large terrain relief within the footprint will have residual interior distortion.

---

## DEM accuracy — validation results

Six reference frames from a 200 m circular orbit, gimbal pitched 75° below level:

| Frame | Position (N/E m) | Flat area | DEM area | Ratio | Centre shift |
|-------|------------------|-----------|----------|-------|--------------|
| 531   | +269 / −36       | 56 003 m² | 43 860 m² | 0.783 |  9.5 m |
| 757   | +180 / +179      | 56 201 m² | 37 219 m² | 0.662 | 23.5 m |
| 976   | −48  / +197      | 56 199 m² | 35 237 m² | 0.627 | 26.0 m |
| 1197  | −169 / ±0        | 56 454 m² | 28 620 m² | **0.507** | **42.6 m** |
| 1417  | −48  / −197      | 56 906 m² | 40 056 m² | 0.704 | 22.0 m |
| 1637  | +179 / −180      | 56 363 m² | 37 778 m² | 0.670 | 23.7 m |
| **Mean** | | | | **0.659** | **24.6 m** |

The DEM reduces footprint area by an average of 34% and shifts the centre by an average of 25 m compared to the flat-ground assumption. Frame 1197 is the most extreme case: the camera looks over the mountainous interior of the island where terrain approaches the camera altitude, making the flat-ground footprint nearly twice the correct size.

Flat-ground regression: 6/6 frames produce byte-identical corners to the original `camera_calculator.py` when `dem_sampler=None`.

---

## Telemetry format

Each `.txt` file is a `key: value` file (one field per line, all numeric values as floats):

| Field | Unit | Description |
|-------|------|-------------|
| `n` | m | North offset from local origin |
| `e` | m | East offset from local origin |
| `d` | m | Altitude above home — used as `pose.alt` |
| `roll` | rad | Platform roll |
| `pitch` | rad | Platform pitch |
| `yaw` | rad | Platform heading (0 = north, clockwise positive) |
| `gimbal_roll` | deg | Gimbal roll offset added to platform roll |
| `gimbal_pitch` | deg | Gimbal pitch offset added to platform pitch |
| `gimbal_yaw` | deg | Gimbal yaw offset added to platform yaw |

Additional fields (`lat`, `lon`, `alt`, `airspeed`, `radius`) are parsed and ignored.

---

## Regenerating terrain assets

`georeference_terrain.py` rebuilds `island_map.tif` and `island_dem.tif` from QGIS / ArcGIS outputs. It requires two source files **not included in this package**:

- `IslandMap_NorthUp.tif` — colour ortho georeferenced into the ortho's pixel-coordinate frame
- `HeightMap_Georef.tif` — heightmap georeferenced into the same frame

```bash
python georeference_terrain.py \
    --ortho IslandMap_NorthUp.tif \
    --heightmap HeightMap_Georef.tif \
    --qc
```

Outputs: `island_map.tif`, `island_dem.tif`, `alignment_qc.png`.

**Elevation mapping:** the source heightmap spans pixel values 57–232 (not the full 0–255 range). These are mapped linearly: 57 → 0 m (sea level), 232 → 322 m (peak). The scale (1.84 m per pixel unit) is embedded in the output GeoTIFF tags. The DEM's valid range within the ortho footprint is 1.8–282.3 m; the absolute 322 m peak falls outside the ortho's coverage.

---

## Known issues

**Longitude sign in `params.yaml`**
`takeoff.lon` is set to `−149.1652232`. The ArduPilot SITL location near Canberra uses the positive value `+149.1652232`. The local ENU frame is internally self-consistent either way, so georeferencing and DEM lookups are correct. However, any code that converts local metres to absolute decimal degrees will place results in the wrong hemisphere until this is fixed.

**Approximate map projection**
The base-map projection uses a 4-point perspective transform. It is exact at the four DEM-corrected corners but assumes planarity between them. For frames with large elevation variation within the footprint, a dense grid warp using `cv2.remap` would give better sub-footprint accuracy.

**Altitude frame assumption**
`pose.alt` is taken from the telemetry `d` field (height above home). The DEM stores elevations in metres above MSL. These are in the same frame only when the takeoff site is at or near sea level, which holds for this SITL configuration (`takeoff.alt: 0.1 m`). For deployments at significant elevation above MSL, add the takeoff MSL altitude to `d` before passing it to `GeoReferencer`.

---

## Dependencies

| Package | Min version | Purpose |
|---------|-------------|---------|
| `affine` | 2.4 | 2D affine transforms for ENU frame alignment |
| `dronekit` | 2.9 | MAVLink vehicle interface (live flight mode) |
| `future` | 1.0 | dronekit Python 3 compatibility shim |
| `geopy` | 2.4 | Geographic distance utilities |
| `opencv-python` | 4.13 | Image I/O, homography, perspective warp |
| `pymavlink` | 2.4 | MAVLink message parsing |
| `pyyaml` | 6.0 | `params.yaml` loading |
| `rasterio` | 1.3 | GeoTIFF read/write for DEM and terrain assets |
| `vector3d` | 1.1 | 3D vector arithmetic in `camera_calculator.py` |
