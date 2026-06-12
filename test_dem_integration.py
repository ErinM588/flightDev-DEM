"""
test_dem_integration.py
========================

Validation of the complete DEM-aware georeferencing pipeline against
6 reference frames from a simulated circular orbit at 200 m altitude.

For each frame the script:
  1. Runs the flat-ground path (original behaviour, no DEM)
  2. Runs the DEM path (strict=True; retries with strict=False on rejection)
  3. Prints a comparison table
  4. Draws both footprints on the island map (cyan = flat, yellow = DEM)
  5. Warps the camera image into both map spaces side-by-side

Outputs
-------
  test_comparison.png   island map with all 6 flat+DEM footprints overlaid
  test_results.txt      machine-readable summary table
"""

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

from camera_calculator import CameraCalculator
from dem import DEMSampler
from georeferencing import (
    CameraModel, GeoReferencer, ImageSize, PlatformPose,
    find_reddest_kernel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FRAMES_DIR = Path("test_data")
PARAMS_FILE = Path("params.yaml")
BASE_MAP    = Path("IslandMap_NorthUp.png")
OUT_MAP     = Path("test_comparison.png")
OUT_RESULTS = Path("test_results.txt")


def parse_kv(path: Path) -> dict:
    data = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            data[k.strip()] = float(v.strip())
        except ValueError:
            data[k.strip()] = v.strip()
    return data


def hfov_to_vfov(hfov_rad: float, aspect: float) -> float:
    return 2.0 * math.atan(math.tan(hfov_rad / 2.0) / aspect)


def poly_area(corners) -> float:
    """Shoelace area of a polygon given as [(x,y), ...]."""
    n = len(corners)
    a = 0.0
    for i in range(n):
        x1, y1 = corners[i][0], corners[i][1]
        x2, y2 = corners[(i + 1) % n][0], corners[(i + 1) % n][1]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def center_of(corners) -> tuple:
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def draw_poly(img, corners_px, color, thickness=2, filled=False):
    pts = np.array(corners_px, dtype=np.int32).reshape(-1, 1, 2)
    if filled:
        cv2.fillPoly(img, [pts], color)
    else:
        cv2.polylines(img, [pts], True, color, thickness)


# ---------------------------------------------------------------------------
# Regression check: flat-ground must match original CameraCalculator exactly
# ---------------------------------------------------------------------------

def regression_check(pose: PlatformPose, camera: CameraModel,
                     image: ImageSize) -> bool:
    """Verify flat-ground result is bit-identical to the original pipeline."""
    import sys, importlib.util
    spec = importlib.util.spec_from_file_location(
        "cc_orig", "/mnt/user-data/uploads/camera_calculator.py")
    cc_orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc_orig)
    OrigCC = cc_orig.CameraCalculator

    from georeferencing import GeoReferencer as GR
    flat_georef = GR()          # no DEM → original path
    fp_new = flat_georef.compute(pose, camera, image)

    # Replicate original pipeline manually
    total_pitch_deg = math.degrees(pose.pitch)
    pitch_conversion = -(90 + total_pitch_deg)
    hfov, vfov = camera.hfov, camera.vfov
    alt = pose.alt
    roll = pose.roll
    pitch_rad = math.radians(pitch_conversion)
    orig_bbox = OrigCC.getBoundingPolygon(hfov, vfov, alt, roll, pitch_rad, 0.0)

    from affine import Affine
    yaw_deg = math.degrees(pose.yaw)
    orig_corners = [Affine.rotation(90 - yaw_deg) * (p.x, p.y) for p in orig_bbox]

    new_corners = fp_new.corners_local
    return all(
        abs(nc[0] - oc[0]) < 1e-8 and abs(nc[1] - oc[1]) < 1e-8
        for nc, oc in zip(new_corners, orig_corners)
    )


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def main():
    # Load config
    with open(PARAMS_FILE) as f:
        params = yaml.safe_load(f)

    width_px = int(params["image"]["width_px"])
    height_px = int(params["image"]["height_px"])
    width_m  = float(params["image"]["width_m"])
    height_m = float(params["image"]["height_m"])
    scale_x  = width_px / width_m
    scale_y  = height_px / height_m

    hfov_rad = math.radians(float(params["camera"]["hfov"]))
    aspect   = float(params["camera"]["aspect"])
    vfov_rad = hfov_to_vfov(hfov_rad, aspect)

    # Base map + origin
    base_map = cv2.imread(str(BASE_MAP))
    orig_y, orig_x, _ = find_reddest_kernel(base_map)

    # DEM sampler
    dem_cfg = params["dem"]
    dem = DEMSampler(
        dem_cfg["path"],
        takeoff_lat=float(params["takeoff"]["lat"]),
        takeoff_lon=float(params["takeoff"]["lon"]),
    )

    # GeoReferencers
    flat_georef = GeoReferencer()           # no DEM — original flat-ground
    dem_strict  = GeoReferencer(            # DEM, strict=True (reject if any corner misses)
        dem_sampler=dem,
        step_m=float(dem_cfg["step_m"]),
        max_range_m=float(dem_cfg["max_range_m"]),
        strict=True,
    )
    dem_lenient = GeoReferencer(            # DEM, strict=False (flat fallback per corner)
        dem_sampler=dem,
        step_m=float(dem_cfg["step_m"]),
        max_range_m=float(dem_cfg["max_range_m"]),
        strict=False,
    )

    # Annotation canvases
    canvas_flat = base_map.copy()
    canvas_dem  = base_map.copy()

    results = []
    frames  = sorted(FRAMES_DIR.glob("frame_*.jpg"))

    print(f"\n{'Frame':<12} {'n':>7} {'e':>7} {'yaw':>7}  "
          f"{'regr':>5}  {'flat m²':>10}  "
          f"{'DEM m²':>10}  {'ratio':>6}  {'DEM-center Δ':>14}  {'status'}")
    print("-" * 110)

    for img_path in frames:
        telem     = parse_kv(img_path.with_suffix(".txt"))
        frame_img = cv2.imread(str(img_path))
        h, w      = frame_img.shape[:2]

        roll_rad  = telem.get("roll", 0.0)  + math.radians(telem.get("gimbal_roll",  0.0))
        pitch_rad = telem.get("pitch", 0.0) + math.radians(telem.get("gimbal_pitch", 0.0))
        yaw_rad   = telem.get("yaw", 0.0)   + math.radians(telem.get("gimbal_yaw",   0.0))
        north_m   = float(telem.get("n", 0.0))
        east_m    = float(telem.get("e", 0.0))

        pose   = PlatformPose(
            alt=float(telem.get("d", telem.get("alt", 0.0))),
            roll=roll_rad, pitch=pitch_rad, yaw=yaw_rad,
        )
        camera    = CameraModel(hfov=hfov_rad, vfov=vfov_rad)
        image_sz  = ImageSize(width=w, height=h)

        def m2px(x_m, y_m):
            return (
                int(round(orig_x + (east_m  + x_m) * scale_x)),
                int(round(orig_y - (north_m + y_m) * scale_y)),
            )

        # --- regression ---
        reg_ok = regression_check(pose, camera, image_sz)

        # --- flat-ground ---
        fp_flat = flat_georef.compute(pose, camera, image_sz)
        flat_area = poly_area(fp_flat.corners_local)
        flat_center = center_of(fp_flat.corners_local)

        # --- DEM strict ---
        fp_dem = dem_strict.compute(pose, camera, image_sz)
        if fp_dem is None:
            status = "STRICT-REJECT → lax"
            fp_dem = dem_lenient.compute(pose, camera, image_sz)
        else:
            status = "OK"

        # --- compute comparison metrics ---
        if fp_dem is not None:
            dem_area   = poly_area(fp_dem.corners_local)
            dem_center = center_of(fp_dem.corners_local)
            ratio      = dem_area / flat_area if flat_area > 0 else float("nan")
            delta_m    = math.hypot(
                dem_center[0] - flat_center[0],
                dem_center[1] - flat_center[1],
            )
            dem_corners_px = [m2px(p[0], p[1]) for p in fp_dem.corners_local]
        else:
            dem_area = dem_center = ratio = delta_m = None
            status = "FAILED"
            dem_corners_px = None

        flat_corners_px = [m2px(p[0], p[1]) for p in fp_flat.corners_local]

        # --- draw on canvases ---
        draw_poly(canvas_flat, flat_corners_px, (0, 255, 255), 2)          # cyan
        if dem_corners_px:
            draw_poly(canvas_dem, flat_corners_px, (0, 255, 255), 1)       # faint cyan
            draw_poly(canvas_dem, dem_corners_px,  (0, 200, 50),  2)       # green

        # Aircraft position dot
        ac_px = m2px(0, 0)
        cv2.circle(canvas_flat, ac_px, 5, (0, 0, 255), -1)
        cv2.circle(canvas_dem,  ac_px, 5, (0, 0, 255), -1)

        # --- print row ---
        row = {
            "frame": img_path.stem,
            "n": north_m, "e": east_m,
            "yaw_deg": math.degrees(yaw_rad),
            "regression_ok": reg_ok,
            "flat_area_m2": flat_area,
            "dem_area_m2": dem_area,
            "area_ratio": ratio,
            "center_delta_m": delta_m,
            "status": status,
        }
        results.append(row)

        dem_a_str   = f"{dem_area:10.0f}" if dem_area is not None else f"{'n/a':>10}"
        ratio_str   = f"{ratio:6.3f}"     if ratio   is not None else f"{'n/a':>6}"
        delta_str   = f"{delta_m:8.1f} m" if delta_m is not None else f"{'n/a':>10}"
        print(f"{img_path.stem:<12} {north_m:+7.0f} {east_m:+7.0f} {math.degrees(yaw_rad):+7.1f}°"
              f"  {'OK' if reg_ok else 'FAIL':>5}  {flat_area:10.0f}  "
              f"{dem_a_str}  {ratio_str}  {delta_str:>14}  {status}")

    # --- save annotated maps side-by-side ---
    # Add legend text
    for canvas, label in [(canvas_flat, "FLAT-GROUND"), (canvas_dem, "DEM-AWARE")]:
        cv2.putText(canvas, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(canvas, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
    # Put red dot + label for origin
    cv2.circle(canvas_dem, (orig_x, orig_y), 8, (0, 0, 255), -1)
    cv2.putText(canvas_dem, "origin", (orig_x + 10, orig_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

    combined = np.concatenate([canvas_flat, canvas_dem], axis=1)
    cv2.imwrite(str(OUT_MAP), combined)
    print(f"\nSaved annotated map → {OUT_MAP}")

    # --- save text results ---
    lines = ["frame,n,e,yaw_deg,regression_ok,flat_area_m2,dem_area_m2,area_ratio,center_delta_m,status"]
    for r in results:
        lines.append(
            f"{r['frame']},{r['n']:.1f},{r['e']:.1f},{r['yaw_deg']:.1f},"
            f"{r['regression_ok']},"
            f"{r['flat_area_m2']:.1f},"
            f"{r['dem_area_m2'] if r['dem_area_m2'] else 'n/a'},"
            f"{r['area_ratio'] if r['area_ratio'] else 'n/a'},"
            f"{r['center_delta_m'] if r['center_delta_m'] else 'n/a'},"
            f"{r['status']}"
        )
    OUT_RESULTS.write_text("\n".join(lines) + "\n")
    print(f"Saved results table → {OUT_RESULTS}")

    # --- summary ---
    ok_rows  = [r for r in results if r["dem_area_m2"]]
    reg_pass = sum(1 for r in results if r["regression_ok"])
    print(f"\n{'='*60}")
    print(f"Regression (flat-ground identical to original): {reg_pass}/{len(results)}")
    if ok_rows:
        avg_ratio = sum(r["area_ratio"] for r in ok_rows) / len(ok_rows)
        avg_delta = sum(r["center_delta_m"] for r in ok_rows) / len(ok_rows)
        print(f"DEM valid frames        : {len(ok_rows)}/{len(results)}")
        print(f"Mean footprint ratio    : {avg_ratio:.3f}  (DEM/flat; <1 = terrain correction reducing footprint)")
        print(f"Mean center displacement: {avg_delta:.1f} m  (DEM vs flat-ground footprint center)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
