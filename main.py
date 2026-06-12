import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from dem import DEMSampler
from georeferencing import CameraModel, GeoReferencer, ImageSize, PlatformPose, find_reddest_kernel


def parse_kv_file(file_path: Path) -> dict:
    """
    Parse simple key:value telemetry files into a dict.
    """
    data = {}
    with file_path.open("r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                data[key] = float(value)
            except ValueError:
                data[key] = value
    return data


def hfov_to_vfov(hfov_rad: float, aspect: float) -> float:
    """
    Convert horizontal FOV to vertical FOV given width/height aspect.
    """
    return 2 * math.atan(math.tan(hfov_rad / 2) / aspect)


def main():
    # TODO: build your list of (image_path, telemetry_path) pairs here.
    DATA_PATH = Path("test_data")
    
    # Zip together images and telemetry files based on naming convention (e.g., image_001.jpg <-> telemetry_001.txt)
    image_files = sorted(DATA_PATH.glob("*.jpg"))
    telemetry_files = sorted(DATA_PATH.glob("*.txt"))

    telem_data = []

    # For each line in texts files, parse to dict from {{key: value}} format, with value = float(value.strip())
    for telemetry_file in telemetry_files:
        telemetry_data = parse_kv_file(telemetry_file)
        telem_data.append(telemetry_data)
        print(f"Parsed telemetry from {telemetry_file.name}: {telemetry_data}")

    # Reassemble into img, data pairs for processing
    pairs = []

    def _strip_prefix(stem: str) -> str:
        if stem.startswith("frame_"):
            return stem[len("frame_"):]
        if stem.startswith("state_"):
            return stem[len("state_"):]
        return stem

    telemetry_by_stem = {
        _strip_prefix(p.stem): data for p, data in zip(telemetry_files, telem_data)
    }
    for image_path in image_files:
        stem = _strip_prefix(image_path.stem)
        if stem not in telemetry_by_stem:
            print(f"No telemetry match for {image_path.name}")
            continue
        pairs.append((image_path, telemetry_by_stem[stem]))
    
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    width_px = int(params["image"]["width_px"])
    height_px = int(params["image"]["height_px"])
    width_m = float(params["image"]["width_m"])
    height_m = float(params["image"]["height_m"])

    scale_x = width_px / width_m
    scale_y = height_px / height_m

    hfov_rad = math.radians(float(params["camera"]["hfov"]))
    aspect = float(params["camera"]["aspect"])
    vfov_rad = hfov_to_vfov(hfov_rad, aspect)

    # Load DEM sampler if configured. Absence of the key, or the file not
    # existing, both fall back silently to the flat-ground path.
    dem_sampler = None
    dem_step_m = 5.0
    dem_max_range_m = 10000.0
    dem_strict = True
    if "dem" in params:
        dem_cfg = params["dem"]
        dem_path = Path(dem_cfg.get("path", "island_dem.tif"))
        if dem_path.exists():
            dem_sampler = DEMSampler(
                dem_path,
                takeoff_lat=float(params["takeoff"]["lat"]),
                takeoff_lon=float(params["takeoff"]["lon"]),
            )
            dem_step_m = float(dem_cfg.get("step_m", 5.0))
            dem_max_range_m = float(dem_cfg.get("max_range_m", 10000.0))
            dem_strict = bool(dem_cfg.get("strict", True))
            print(f"DEM loaded: {dem_path}  step={dem_step_m}m  strict={dem_strict}")
        else:
            print(f"DEM '{dem_path}' not found — flat-ground fallback active")

    georef = GeoReferencer(
        dem_sampler=dem_sampler,
        step_m=dem_step_m,
        max_range_m=dem_max_range_m,
        strict=dem_strict,
    )

    map_image_path = Path("IslandMap_NorthUp.png")
    base_map = cv2.imread(str(map_image_path))
    if base_map is None:
        print(f"Failed to load map image: {map_image_path}")
        return

    annotated = base_map.copy()
    orig_y, orig_x, _ = find_reddest_kernel(base_map)
    ne_scale = 1.0
    for image_path, telemetry_data in pairs:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Failed to load image: {image_path}")
            continue

        telemetry = telemetry_data

        # Telemetry roll/pitch/yaw are radians; gimbal angles are in degrees.
        roll_rad = telemetry.get("roll", 0.0) + math.radians(telemetry.get("gimbal_roll", 0.0))
        pitch_rad = telemetry.get("pitch", 0.0) + math.radians(telemetry.get("gimbal_pitch", 0.0))
        yaw_rad = telemetry.get("yaw", 0.0) + math.radians(telemetry.get("gimbal_yaw", 0.0))

        pose = PlatformPose(
            alt=float(telemetry.get("d", telemetry.get("alt", 0.0))),
            roll=roll_rad,
            pitch=pitch_rad,
            yaw=yaw_rad,
        )

        camera = CameraModel(
            hfov=hfov_rad,
            vfov=vfov_rad,
            pitch_offset_deg=0.0,
        )

        frame_h, frame_w = frame.shape[:2]
        image_size = ImageSize(width=frame_w, height=frame_h)
        footprint = georef.compute(pose, camera, image_size)
        if footprint is None:
            print(f"DEM footprint rejected for {image_path.name} (ray exits coverage) — skipping")
            continue

        # Local origin is the takeoff position (red dot); +x is east (right), +y is north (up).
        # Use local north/east meters from telemetry to place footprint in map space.
        north_raw = float(telemetry.get("n", 0.0))
        east_raw = float(telemetry.get("e", 0.0))

        north_m = north_raw * ne_scale
        east_m = east_raw * ne_scale

        def meters_to_map_pixels(x_m: float, y_m: float) -> tuple[int, int]:
            px = orig_x + (east_m + x_m) * scale_x
            py = orig_y - (north_m + y_m) * scale_y
            return int(round(px)), int(round(py))

        pixel_corners = [meters_to_map_pixels(p[0], p[1]) for p in footprint.corners_local]
        xs = [p[0] for p in pixel_corners]
        ys = [p[1] for p in pixel_corners]
        if max(xs) < 0 or max(ys) < 0 or min(xs) > width_px or min(ys) > height_px:
            print(f"Footprint off-map for {image_path.name}; check units or origin.")

        polygon = np.array(pixel_corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [polygon], True, (0, 255, 255), 2)

        # Warp the camera frame into map space and overlay it.
        if len(pixel_corners) == 4:
            src = np.float32(
                [
                    [0, 0],
                    [frame_w - 1, 0],
                    [frame_w - 1, frame_h - 1],
                    [0, frame_h - 1],
                ]
            )
            dst = np.float32(pixel_corners)
            img_to_map = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(
                frame,
                img_to_map,
                (width_px, height_px),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_TRANSPARENT,
            )
            mask = (warped.sum(axis=2) > 0).astype(np.uint8) * 255
            mask = cv2.merge([mask, mask, mask])
            annotated = np.where(mask > 0, warped, annotated)

        if footprint.center_local is not None:
            center_px = meters_to_map_pixels(footprint.center_local[0], footprint.center_local[1])
            cv2.circle(annotated, center_px, radius=4, color=(255, 0, 0), thickness=-1)

        # Mark takeoff (origin) and current aircraft position.
        cv2.circle(annotated, (orig_x, orig_y), radius=4, color=(0, 0, 255), thickness=-1)
        aircraft_px = meters_to_map_pixels(0.0, 0.0)
        cv2.circle(annotated, aircraft_px, radius=4, color=(0, 255, 0), thickness=-1)

        # Draw heading line for yaw debugging (0 deg = north, positive clockwise).
        # Uses the platform yaw only — not the combined gimbal yaw — so the
        # arrow tracks the aircraft nose direction regardless of gimbal offset.
        platform_yaw_rad = telemetry.get("yaw", 0.0)
        heading_len_m = 20.0
        heading_e = heading_len_m * math.sin(platform_yaw_rad)
        heading_n = heading_len_m * math.cos(platform_yaw_rad)
        heading_px = meters_to_map_pixels(heading_e, heading_n)
        cv2.line(annotated, aircraft_px, heading_px, (0, 255, 0), 2)

        cv2.imshow("Georeferencing Loop", annotated)
        cv2.waitKey(0)

        

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
