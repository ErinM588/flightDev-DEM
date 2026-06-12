import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from affine import Affine
from camera_calculator import CameraCalculator


MeterPoint = Tuple[float, float]
LatLon = Tuple[float, float]


def find_reddest_kernel(img_bgr: np.ndarray) -> Tuple[int, int, float]:
    """
    Find the 3x3 region with the highest total 'redness' in a BGR image.
    Returns (best_y, best_x, best_score).
    """

    # Split channels (OpenCV uses BGR)
    B = img_bgr[:, :, 0].astype(np.float32)
    G = img_bgr[:, :, 1].astype(np.float32)
    R = img_bgr[:, :, 2].astype(np.float32)

    # Define redness metric
    redness = R - (B + G) / 2.0

    # Compute summed redness for every 3x3 window using convolution-like slicing
    h, w = redness.shape
    best_score = -np.inf
    best_pos = (0, 0)

    for y in range(h - 2):
        for x in range(w - 2):
            window_score = np.sum(redness[y:y+3, x:x+3])
            if window_score > best_score:
                best_score = window_score
                best_pos = (y, x)

    return best_pos[0], best_pos[1], best_score


@dataclass(frozen=True)
class PlatformPose:
    alt: float
    roll: float  # radians
    pitch: float  # radians
    yaw: float  # radians


@dataclass(frozen=True)
class CameraModel:
    hfov: float  # radians
    vfov: float  # radians
    pitch_offset_deg: float = 0.0


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int


@dataclass(frozen=True)
class Footprint:
    corners_raw: List[MeterPoint]
    corners_local: List[MeterPoint]
    homography: Optional[np.ndarray]
    center_local: Optional[MeterPoint]


def offsets_to_decimal_degrees(lat: float, lon: float, dx: float, dy: float) -> LatLon:
    """
    Convert local meter offsets into decimal degrees from an origin.
    """
    earth_radius_m = 6378137

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    dlat = dy / earth_radius_m
    dlon = dx / (earth_radius_m * math.cos(lat_rad))

    new_lat = math.degrees(lat_rad + dlat)
    new_lon = math.degrees(lon_rad + dlon)
    return new_lat, new_lon


def convert_corners_to_decimal_degrees(origin: LatLon, corners: List[MeterPoint]) -> Dict[str, LatLon]:
    """
    Convert footprint corners from local offsets to decimal degrees.
    """
    ddlat, ddlon = origin
    return {
        "top_left": offsets_to_decimal_degrees(ddlat, ddlon, corners[0][0], corners[0][1]),
        "top_right": offsets_to_decimal_degrees(ddlat, ddlon, corners[1][0], corners[1][1]),
        "bottom_right": offsets_to_decimal_degrees(ddlat, ddlon, corners[2][0], corners[2][1]),
        "bottom_left": offsets_to_decimal_degrees(ddlat, ddlon, corners[3][0], corners[3][1]),
    }


def get_homography_matrix(
    image_width: int,
    image_height: int,
    rotated_corners: List[MeterPoint],
) -> Optional[np.ndarray]:
    """
    Map any pixel in projected footprint pixels to ground coordinates
    using a homography matrix.
    """
    src = np.float32(
        [
            [0, 0],
            [image_width - 1, 0],
            [image_width - 1, image_height - 1],
            [0, image_height - 1],
        ]
    )

    dst = np.float32([[corner[0], corner[1]] for corner in rotated_corners])
    H, _ = cv2.findHomography(src, dst)
    return H


def transform_pixel_to_ground(x: float, y: float, H: np.ndarray) -> np.ndarray:
    """
    Transform a pixel coordinate to ground coordinates using the homography matrix.
    """
    pixel = np.array([x, y, 1], dtype=np.float32)
    ground_coord = H @ pixel
    ground_coord /= ground_coord[2]
    return ground_coord[:2]


def pixel_to_decimal_degrees(x: float, y: float, H: np.ndarray, origin: LatLon) -> LatLon:
    """
    Convert pixel coordinates to decimal degrees using the homography matrix.
    """
    ground_coord = transform_pixel_to_ground(x, y, H)
    return offsets_to_decimal_degrees(origin[0], origin[1], ground_coord[0], ground_coord[1])


class GeoReferencer:
    def __init__(
        self,
        dem_sampler=None,
        step_m: float = 5.0,
        max_range_m: float = 10000.0,
        strict: bool = True,
    ) -> None:
        self.camera_calculator = CameraCalculator()
        self.dem_sampler = dem_sampler
        self.step_m = step_m
        self.max_range_m = max_range_m
        self.strict = strict

    def _calculator_inputs(self, pose: PlatformPose, camera: CameraModel) -> Dict[str, float]:
        total_pitch_deg = math.degrees(pose.pitch) + camera.pitch_offset_deg

        # Convert camera total pitch from level=0, nadir=-90 to nadir=0, level=90
        pitch_conversion = -(90 + total_pitch_deg)

        if self.dem_sampler is not None:
            # DEM path: rays must land in local ENU (x=east, y=north) before
            # intersection so DEMSampler can look up terrain elevation. The
            # camera-calculator frame needs a 90° rotation to align with ENU,
            # so we bake the full yaw into rotateRays. This is mathematically
            # equivalent to the post-hoc 2D rotation used for flat-ground but
            # must be applied up-front for terrain-aware intersection.
            yaw = math.pi / 2 - pose.yaw
        else:
            # Flat-ground path: pass yaw=0 here; the 2D rotation in compute()
            # applies the ENU frame correction after the z=0 intersection.
            # Preserved exactly as-is to maintain byte-for-byte regression.
            yaw = 0.0

        return {
            "hfov": camera.hfov,
            "vfov": camera.vfov,
            "alt": pose.alt,
            "roll": pose.roll,
            "pitch": math.radians(pitch_conversion),
            "yaw": yaw,
        }

    def compute(
        self,
        pose: PlatformPose,
        camera: CameraModel,
        image: ImageSize,
    ) -> Optional[Footprint]:
        """
        Compute footprint corners and homography for a platform pose and camera model.
        Returns outputs in local ENU meters relative to the local origin.

        Returns None when dem_sampler is active, strict=True, and at least one
        corner ray fails to intersect the DEM (exits coverage or points up).
        Callers must handle None — typically by skipping the frame.
        """
        inputs = self._calculator_inputs(pose, camera)
        bbox = self.camera_calculator.getBoundingPolygon(
            *inputs.values(),
            dem_sampler=self.dem_sampler,
            strict=self.strict,
            step_m=self.step_m,
            max_range_m=self.max_range_m,
        )

        if bbox is None:
            # All-or-nothing: DEM strict mode rejected the polygon because at
            # least one corner ray could not be resolved against the DEM.
            return None

        corners_raw = [[p.x, p.y] for p in bbox]

        if self.dem_sampler is not None:
            # The yaw was baked into rotateRays, so corners already point to
            # local ENU positions (x = east, y = north). No 2D rotation needed.
            corners_local = [(p.x, p.y) for p in bbox]
        else:
            # Flat-ground path (original behaviour preserved exactly):
            # apply yaw-to-ENU 2D rotation after the z=0 intersection.
            yaw_deg = math.degrees(pose.yaw)
            corners_local = [
                Affine.rotation(90 - yaw_deg) * (p.x, p.y) for p in bbox
            ]

        homography = get_homography_matrix(image.width, image.height, corners_local)

        if homography is not None:
            center_local = transform_pixel_to_ground(
                image.width // 2,
                image.height // 2,
                homography,
            )
            center_local_tuple: Optional[MeterPoint] = (
                float(center_local[0]),
                float(center_local[1]),
            )
        else:
            center_local_tuple = None

        return Footprint(
            corners_raw=corners_raw,
            corners_local=corners_local,
            homography=homography,
            center_local=center_local_tuple,
        )


def footprint_corners_to_decimal_degrees(
    footprint: Footprint, origin: LatLon
) -> Dict[str, LatLon]:
    """
    Convert local footprint corners to decimal degrees using a lat/lon origin.
    """
    return convert_corners_to_decimal_degrees(origin, footprint.corners_local)


def footprint_center_to_decimal_degrees(
    footprint: Footprint, origin: LatLon
) -> Optional[LatLon]:
    """
    Convert the local image center to decimal degrees using a lat/lon origin.
    """
    if footprint.center_local is None:
        return None
    return offsets_to_decimal_degrees(
        origin[0], origin[1], footprint.center_local[0], footprint.center_local[1]
    )


def pretty_print_footprint(footprint: Footprint) -> None:
    """
    Print footprint results with consistent rounding.
    """
    def _format_point(point: Optional[MeterPoint]) -> str:
        if point is None:
            return "None"
        return f"x={round(point[0], 7)}, y={round(point[1], 7)}"

    corner_labels = ("top_left", "top_right", "bottom_right", "bottom_left")
    rounded_corners = [_format_point(p) for p in footprint.corners_local]
    rounded_center = _format_point(footprint.center_local)

    print("Footprint (local meters, 7dp):")
    for label, point in zip(corner_labels, rounded_corners):
        print(f"  {label}: {point}")
    print(f"  center: {rounded_center}")


if __name__ == "__main__":
    # Minimal demo of local-meter outputs with optional lat/lon conversion.
    pose = PlatformPose(
        alt=30.0,
        roll=math.radians(2.0),
        pitch=math.radians(-10.0),
        yaw=math.radians(0.0),
    )
    camera = CameraModel(
        hfov=math.radians(70.0),
        vfov=math.radians(50.0),
        pitch_offset_deg=0.0,
    )
    image = ImageSize(width=1920, height=1080)

    georef = GeoReferencer()
    footprint = georef.compute(pose, camera, image)

    # print("Corners (local meters):", footprint.corners_local)
    # print("Center (local meters):", footprint.center_local)

    origin = (38.8895, -77.0353)
    corners_dd = footprint_corners_to_decimal_degrees(footprint, origin)
    center_dd = footprint_center_to_decimal_degrees(footprint, origin)

    pretty_print_footprint(footprint)


    # Use real image and params
    import yaml
    yaml_path = "params.yaml"
    with open(yaml_path, "r") as f:
        params = yaml.safe_load(f)

    # param yaml content
    """
    local_origin:
    lat: 0.0
    lon: 0.0
    alt: 0.01

    takeoff:
    lat: -35.3633285
    lon: -149.1652232
    alt: 0.1

    image:
    width_px: 1431
    height_px: 1132
    width_m: 2130
    height_m: 1685
    gsd: 1.49

    camera:
    hfov: 60
    aspect: 1.333333

    stream:
    url: "rtsp://localhost:8554/stream"
    """

    img = cv2.imread("IslandMap_NorthUp.png")  # BGR image
    orig_y, orig_x, score = find_reddest_kernel(img)

    print(f"Reddest 3x3 kernel starts at (y={orig_y}, x={orig_x}) with score {score}")

    # Test position is 50 meters above origin, with 80 degree pitch down and looking SE
    test_pose = PlatformPose(
        alt=50.0,
        roll=math.radians(0.0),
        pitch=math.radians(-80.0),
        yaw=math.radians(135.0),
    )

    test_camera = CameraModel(
        hfov=math.radians(60.0),
        vfov=math.radians(45.0),
        pitch_offset_deg=0.0,
    )

    test_image = ImageSize(width=1431, height=1132)
    georef = GeoReferencer()
    footprint = georef.compute(test_pose, test_camera, test_image)
    pretty_print_footprint(footprint)

    print("Displaying footprint corners on image...")
    # Draw footprint on image using map scale from params.yaml
    if img is None:
        raise FileNotFoundError("IslandMap_NorthUp.png not found or failed to load.")

    width_px = int(params["image"]["width_px"])
    height_px = int(params["image"]["height_px"])
    width_m = float(params["image"]["width_m"])
    height_m = float(params["image"]["height_m"])

    scale_x = width_px / width_m
    scale_y = height_px / height_m

    # Local origin is the takeoff position (red dot); +x is east (right), +y is north (up).
    def meters_to_map_pixels(x_m: float, y_m: float) -> Tuple[int, int]:
        px = orig_x + x_m * scale_x
        py = orig_y - y_m * scale_y
        return int(round(px)), int(round(py))

    pixel_corners = [meters_to_map_pixels(p[0], p[1]) for p in footprint.corners_local]
    polygon = np.array(pixel_corners, dtype=np.int32).reshape((-1, 1, 2))

    annotated = img.copy()
    cv2.polylines(annotated, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)

    if footprint.center_local is not None:
        center_px = meters_to_map_pixels(footprint.center_local[0], footprint.center_local[1])
        cv2.circle(annotated, center_px, radius=4, color=(255, 0, 0), thickness=-1)

    cv2.circle(annotated, (orig_x, orig_y), radius=4, color=(0, 0, 255), thickness=-1)

    cv2.imshow("Georeferencing Demo", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    