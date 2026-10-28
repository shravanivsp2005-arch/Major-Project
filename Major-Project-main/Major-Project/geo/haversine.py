"""
CivicPulse Geo-Spatial Intelligence & Haversine Distance Engine
==============================================================
Provides high-performance spatial distance calculations, proximity checks,
and temporal-spatial clustering for civic complaint deduplication.
"""

import math
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any, Union


EARTH_RADIUS_METERS = 6371000.0  # Mean radius of the Earth in meters
EARTH_RADIUS_KM = 6371.0


def haversine_distance_meters(
    lat1: float,
    lng1: float,
    lat2: float,
    lng2: float
) -> float:
    """
    Computes the great-circle distance between two GPS points in meters
    using the Haversine formula.
    """
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlamb = math.radians(lng2 - lng1)

    a = (math.sin(dphi / 2.0) ** 2) + math.cos(p1) * math.cos(p2) * (math.sin(dlamb / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return EARTH_RADIUS_METERS * c


def haversine_distance_km(
    lat1: float,
    lng1: float,
    lat2: float,
    lng2: float
) -> float:
    """
    Computes the distance between two GPS coordinates in kilometers.
    """
    return haversine_distance_meters(lat1, lng1, lat2, lng2) / 1000.0


def is_within_radius(
    lat1: float,
    lng1: float,
    lat2: float,
    lng2: float,
    radius_meters: float = 200.0
) -> bool:
    """
    Checks if point 2 is within `radius_meters` of point 1.
    """
    dist = haversine_distance_meters(lat1, lng1, lat2, lng2)
    return dist <= radius_meters


def compute_time_difference_hours(
    t1: Union[datetime, str],
    t2: Union[datetime, str]
) -> float:
    """
    Calculates the absolute difference in hours between two timestamps.
    """
    if isinstance(t1, str):
        try:
            t1 = datetime.fromisoformat(t1.replace("Z", "+00:00"))
        except Exception:
            t1 = datetime.now(timezone.utc)
    if isinstance(t2, str):
        try:
            t2 = datetime.fromisoformat(t2.replace("Z", "+00:00"))
        except Exception:
            t2 = datetime.now(timezone.utc)

    diff_seconds = abs((t1 - t2).total_seconds())
    return diff_seconds / 3600.0


def is_spatial_temporal_duplicate(
    lat1: float,
    lng1: float,
    time1: Union[datetime, str],
    lat2: float,
    lng2: float,
    time2: Union[datetime, str],
    max_distance_m: float = 200.0,
    max_time_hours: float = 72.0
) -> Tuple[bool, float, float]:
    """
    Evaluates whether two civic reports represent the same incident based on:
    1. Distance < max_distance_m (default 200m)
    2. Time delta < max_time_hours (default 72h)
    Returns: (is_duplicate, distance_meters, time_delta_hours)
    """
    dist = haversine_distance_meters(lat1, lng1, lat2, lng2)
    time_diff = compute_time_difference_hours(time1, time2)
    is_dup = (dist <= max_distance_m) and (time_diff <= max_time_hours)
    return is_dup, round(dist, 2), round(time_diff, 2)
