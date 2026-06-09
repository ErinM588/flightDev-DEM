"""
***************************************************************************
    camera_calculator.py
    ---------------------
    Date                 : August 2019
    Copyright            : (C) 2019 by Luigi Pirelli
    Email                : luipir at gmail dot com
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

# __author__ = 'Luigi Pirelli'
# __date__ = 'August 2019'
# __copyright__ = '(C) 2019, Luigi Pirelli'

import math
import numpy as np

# pip install vector3d
from vector3d.vector import Vector


class CameraCalculator:
    """
    Porting of CameraCalculator.java

    This code is a 1to1 python porting of the java code:
        https://github.com/zelenmi6/thesis/blob/master/src/geometry/CameraCalculator.java
    referred in:
        https://stackoverflow.com/questions/38099915/calculating-coordinates-of-an-oblique-aerial-image
    The only part not ported are that explicetly abandoned or not used at all by the main
    call to getBoundingPolygon method.
    by: milan zelenka
    https://github.com/zelenmi6
    https://stackoverflow.com/users/6528363/milan-zelenka

    DEM support (added)
    -------------------
    All three public methods accept an optional ``dem_sampler``. When None
    (the default), every ray is intersected with the flat plane z=0 using
    the original closed-form solve. When a sampler is provided, each ray
    is marched against the DEM surface and the first downward crossing is
    returned. ``dem_sampler`` is duck-typed: any object exposing
    ``.sample(east_m, north_m) -> Optional[float]`` works; see ``dem.py``
    for the canonical implementation.

    Important: ray-marching assumes the rays passed into intersection are
    in local ENU meters with (ray.x, ray.y, ray.z) == (east, north, up).
    The flat-ground path commutes with yaw-rotation-about-vertical, so
    the existing pipeline can defer the yaw rotation until after the
    intersection -- the DEM path cannot. To use ray-marching correctly
    the caller must pass the true heading into ``rotateRays`` rather than
    rotating the resulting corners afterwards (see the GeoReferencer
    integration step).

    example:

        c=CameraCalculator()
        bbox=c.getBoundingPolygon(
            math.radians(62),
            math.radians(84),
            117.1,
            math.radians(0),
            math.radians(33.6),
            math.radians(39.1))
        for i, p in enumerate(bbox):
            print("point:", i, '-', p.x, p.y, p.z)
    """

    # Default ray-marching tuning. A 5 m step keeps ~20 samples per
    # kilometer of ray; max_range_m caps work for rays that never
    # converge (e.g. cameras pitched up).
    DEFAULT_STEP_M = 5.0
    DEFAULT_MAX_RANGE_M = 10000.0

    def __init__(self):
        pass

    def __del__(delf):
        pass

    @staticmethod
    def getBoundingPolygon(
        FOVh, FOVv, altitude, roll, pitch, heading,
        *,
        dem_sampler=None,
        strict: bool = True,
        step_m: float = DEFAULT_STEP_M,
        max_range_m: float = DEFAULT_MAX_RANGE_M,
    ):
        '''Get corners of the polygon captured by the camera on the ground. 
        The calculations are performed in the axes origin (0, 0, altitude)
        and the points are not yet translated to camera's X-Y coordinates.
        Parameters:
            FOVh (float): Horizontal field of view in radians
            FOVv (float): Vertical field of view in radians
            altitude (float): Altitude of the camera in meters
            heading (float): Heading of the camera (z axis) in radians
            roll (float): Roll of the camera (x axis) in radians
            pitch (float): Pitch of the camera (y axis) in radians
            dem_sampler: Optional object with
                ``.sample(east_m, north_m) -> Optional[float]``. When
                provided, each ray is marched against the DEM rather
                than intersected with the z=0 plane.
            strict: When True (default) and ``dem_sampler`` is provided,
                any corner whose DEM intersection fails causes the entire
                polygon to be returned as ``None`` rather than silently
                falling back to flat-ground for that corner. When False,
                failed corners fall back to the flat-ground intersection
                (the old behaviour). Has no effect when ``dem_sampler``
                is None.
            step_m: Ray-march step size in meters (DEM mode only).
            max_range_m: Maximum ray distance to march before giving up
                (DEM mode only).
        Returns:
            list[vector3d.vector.Vector] with 4 ground points, or None
            if strict=True, a dem_sampler was provided, and any corner
            ray failed to intersect the DEM.
        '''
        # import ipdb; ipdb.set_trace()
        ray11 = CameraCalculator.ray1(FOVh, FOVv)
        ray22 = CameraCalculator.ray2(FOVh, FOVv)
        ray33 = CameraCalculator.ray3(FOVh, FOVv)
        ray44 = CameraCalculator.ray4(FOVh, FOVv)

        rotatedVectors = CameraCalculator.rotateRays(
                ray11, ray22, ray33, ray44, roll, pitch, heading)

        origin = Vector(0, 0, altitude)
        intersections = CameraCalculator.getRayGroundIntersections(
            rotatedVectors, origin,
            dem_sampler=dem_sampler,
            strict=strict,
            step_m=step_m,
            max_range_m=max_range_m,
        )

        # All-or-nothing: if the DEM was in use and any corner failed,
        # return None so the caller knows the footprint is unreliable
        # rather than receiving a silently mixed DEM+flat polygon.
        if dem_sampler is not None and strict and any(p is None for p in intersections):
            return None

        return intersections


    # Ray-vectors defining the the camera's field of view. FOVh and FOVv are interchangeable
    # depending on the camera's orientation
    @staticmethod
    def ray1(FOVh, FOVv):
        '''
        tasto
        Parameters:
            FOVh (float): Horizontal field of view in radians
            FOVv (float): Vertical field of view in radians
        Returns:
            vector3d.vector.Vector: normalised vector
        '''
        pass
        ray = Vector(math.tan(FOVv/2), math.tan(FOVh/2), -1)
        return ray.normalize()

    @staticmethod
    def ray2(FOVh, FOVv):
        '''
        Parameters:
            FOVh (float): Horizontal field of view in radians
            FOVv (float): Vertical field of view in radians
        Returns:
            vector3d.vector.Vector: normalised vector
        '''
        ray = Vector(math.tan(FOVv/2), -math.tan(FOVh/2), -1)
        return ray.normalize()

    @staticmethod
    def ray3(FOVh, FOVv):
        '''
        Parameters:
            FOVh (float): Horizontal field of view in radians
            FOVv (float): Vertical field of view in radians
        Returns:
            vector3d.vector.Vector: normalised vector
        '''
        ray = Vector(-math.tan(FOVv/2), -math.tan(FOVh/2), -1)
        return ray.normalize()

    @staticmethod
    def ray4(FOVh, FOVv):
        '''
        Parameters:
            FOVh (float): Horizontal field of view in radians
            FOVv (float): Vertical field of view in radians
        Returns:
            vector3d.vector.Vector: normalised vector
        '''
        ray = Vector(-math.tan(FOVv/2), math.tan(FOVh/2), -1)
        return ray.normalize()

    @staticmethod
    def rotateRays(ray1, ray2, ray3, ray4, roll, pitch, yaw):
        """Rotates the four ray-vectors around all 3 axes
        Parameters:
            ray1 (vector3d.vector.Vector): First ray-vector
            ray2 (vector3d.vector.Vector): Second ray-vector
            ray3 (vector3d.vector.Vector): Third ray-vector
            ray4 (vector3d.vector.Vector): Fourth ray-vector
            roll float: Roll rotation
            pitch float: Pitch rotation
            yaw float: Yaw rotation
        Returns:
            Returns new rotated ray-vectors
        """
        sinAlpha = math.sin(yaw)
        sinBeta = math.sin(pitch)
        sinGamma = math.sin(roll)
        cosAlpha = math.cos(yaw)
        cosBeta = math.cos(pitch)
        cosGamma = math.cos(roll)
        m00 = cosAlpha * cosBeta
        m01 = cosAlpha * sinBeta * sinGamma - sinAlpha * cosGamma
        m02 = cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma
        m10 = sinAlpha * cosBeta
        m11 = sinAlpha * sinBeta * sinGamma + cosAlpha * cosGamma
        m12 = sinAlpha * sinBeta * cosGamma - cosAlpha * sinGamma
        m20 = -sinBeta
        m21 = cosBeta * sinGamma
        m22 = cosBeta * cosGamma

        # Matrix rotationMatrix = new Matrix(new double[][]{{m00, m01, m02}, {m10, m11, m12}, {m20, m21, m22}})
        rotationMatrix = np.array([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])

        # Matrix ray1Matrix = new Matrix(new double[][]{{ray1.x}, {ray1.y}, {ray1.z}})
        # Matrix ray2Matrix = new Matrix(new double[][]{{ray2.x}, {ray2.y}, {ray2.z}})
        # Matrix ray3Matrix = new Matrix(new double[][]{{ray3.x}, {ray3.y}, {ray3.z}})
        # Matrix ray4Matrix = new Matrix(new double[][]{{ray4.x}, {ray4.y}, {ray4.z}})
        ray1Matrix = np.array([[ray1.x], [ray1.y], [ray1.z]])
        ray2Matrix = np.array([[ray2.x], [ray2.y], [ray2.z]])
        ray3Matrix = np.array([[ray3.x], [ray3.y], [ray3.z]])
        ray4Matrix = np.array([[ray4.x], [ray4.y], [ray4.z]])

        res1 = rotationMatrix.dot(ray1Matrix)
        res2 = rotationMatrix.dot(ray2Matrix)
        res3 = rotationMatrix.dot(ray3Matrix)
        res4 = rotationMatrix.dot(ray4Matrix)

        rotatedRay1 = Vector(res1[0, 0], res1[1, 0], res1[2, 0])
        rotatedRay2 = Vector(res2[0, 0], res2[1, 0], res2[2, 0])
        rotatedRay3 = Vector(res3[0, 0], res3[1, 0], res3[2, 0])
        rotatedRay4 = Vector(res4[0, 0], res4[1, 0], res4[2, 0])
        rayArray = [rotatedRay1, rotatedRay2, rotatedRay3, rotatedRay4]

        return rayArray

    @staticmethod
    def getRayGroundIntersections(
        rays, origin,
        *,
        dem_sampler=None,
        strict: bool = True,
        step_m: float = DEFAULT_STEP_M,
        max_range_m: float = DEFAULT_MAX_RANGE_M,
    ):
        """
        Finds the intersections of the camera's ray-vectors 
        and the ground (flat plane z=0 by default, DEM surface when a sampler is provided).
        Parameters:
            rays (vector3d.vector.Vector[]): Array of 4 ray-vectors
            origin (vector3d.vector.Vector): Position of the camera. The computation were developed 
                                            assuming the camera was at the axes origin (0, 0, altitude) and the 
                                            results translated by the camera's real position afterwards.
            dem_sampler, strict, step_m, max_range_m: see ``getBoundingPolygon``.
        Returns:
            list[vector3d.vector.Vector | None]
        """
        # Vector3d [] intersections = new Vector3d[rays.length];
        # for (int i = 0; i < rays.length; i ++) {
        #     intersections[i] = CameraCalculator.findRayGroundIntersection(rays[i], origin);
        # }
        # return intersections

        # 1to1 translation without python syntax optimisation
        intersections = []
        for i in range(len(rays)):
            intersections.append(
                CameraCalculator.findRayGroundIntersection(
                    rays[i], origin,
                    dem_sampler=dem_sampler,
                    strict=strict,
                    step_m=step_m,
                    max_range_m=max_range_m,
                )
            )
        return intersections

    @staticmethod
    def findRayGroundIntersection(
        ray, origin,
        *,
        dem_sampler=None,
        strict: bool = True,
        step_m: float = DEFAULT_STEP_M,
        max_range_m: float = DEFAULT_MAX_RANGE_M,
    ):
        """
        Find a ray-vector's intersection with the ground.

        By default (``dem_sampler=None``) the ground is treated as the
        flat plane z=0, exactly matching the original behaviour and
        return type. When a DEM sampler is supplied, the ray is marched
        against the DEM surface in local ENU meters:

            (ray.x, ray.y, ray.z) == (east, north, up)

        and the first downward crossing is returned.

        Parameters:
            ray (vector3d.vector.Vector): Ray direction.
            origin (vector3d.vector.Vector): Camera position.
            dem_sampler: Optional sampler (see ``getBoundingPolygon``).
            strict: see ``getBoundingPolygon``.
            step_m: Ray-march step size in meters (DEM mode only).
            max_range_m: Stop marching after this far along the ray.

        Returns:
            vector3d.vector.Vector, or None if strict=True and the DEM
            intersection failed.
        """
        if dem_sampler is None:
            return CameraCalculator._intersect_flat_ground(ray, origin)

        hit = CameraCalculator._intersect_dem(
            ray, origin, dem_sampler,
            step_m=step_m, max_range_m=max_range_m,
        )
        if hit is not None:
            return hit

        # DEM march failed (ray exited coverage, pointed up, or exceeded
        # max_range_m). Caller decides the failure policy via strict.
        if strict:
            return None
        return CameraCalculator._intersect_flat_ground(ray, origin)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _intersect_flat_ground(ray, origin):
        """Closed-form intersection with the z=0 plane.

        This is the original implementation, lifted verbatim into a
        helper so the public method can pick between the flat-ground
        path and the DEM path without duplicating math.
        """
        # Parametric form of an equation
        # P = origin + vector * t
        x = Vector(origin.x, ray.x)
        y = Vector(origin.y, ray.y)
        z = Vector(origin.z, ray.z)

        # Equation of the horizontal plane (ground)
        # -z = 0

        # Calculate t by substituting z
        t = - (z.x / z.y)

        # Substitute t in the original parametric equations to get points of intersection
        return Vector(x.x + x.y * t, y.x + y.y * t, z.x + z.y * t)

    @staticmethod
    def _intersect_dem(ray, origin, dem_sampler, *, step_m, max_range_m):
        """Ray-march a downward-pointing ray against a DEM.

        Assumes (ray.x, ray.y, ray.z) == (east, north, up) in local
        meters. Returns the intersection ``Vector`` at the first
        downward crossing, or ``None`` for:
            - rays that don't head toward the ground (``ray.z >= 0``),
            - rays that exit the DEM coverage before crossing terrain,
            - rays that exceed ``max_range_m`` without crossing terrain.

        The first hit is refined with 8 bisection iterations, which
        gives a worst-case error of ``step_m / 2**8`` along the ray
        (~2 cm with the default 5 m step).
        """
        if ray.z >= 0:
            return None  # no downward intersection in finite range

        # Degenerate case: camera is already below the local terrain.
        # Return the camera position so the footprint corner is at least
        # defined; the caller can decide whether to flag/skip.
        base = dem_sampler.sample(origin.x, origin.y)
        if base is not None and origin.z < base:
            return Vector(origin.x, origin.y, origin.z)

        prev_t = 0.0
        t = step_m
        while t <= max_range_m:
            px = origin.x + ray.x * t
            py = origin.y + ray.y * t
            pz = origin.z + ray.z * t
            terrain_z = dem_sampler.sample(px, py)
            if terrain_z is None:
                return None  # exited DEM; caller may fall back to flat ground
            if pz <= terrain_z:
                # Crossed the surface between prev_t and t — bisect.
                lo, hi = prev_t, t
                for _ in range(8):
                    mid = 0.5 * (lo + hi)
                    mpz = origin.z + ray.z * mid
                    mt = dem_sampler.sample(
                        origin.x + ray.x * mid,
                        origin.y + ray.y * mid,
                    )
                    if mt is None:
                        # Skimmed the DEM edge mid-bisection; bail with
                        # the current bracket.
                        break
                    if mpz > mt:
                        lo = mid
                    else:
                        hi = mid
                t_hit = 0.5 * (lo + hi)
                return Vector(
                    origin.x + ray.x * t_hit,
                    origin.y + ray.y * t_hit,
                    origin.z + ray.z * t_hit,
                )
            prev_t = t
            t += step_m

        return None  # didn't cross terrain within max_range_m
