"""Captain walking-skeleton Phase 9a: local equirectangular projection between the
mission-sim's lat/lon frame and the encounter-sim's local metric (x, y in metres) frame
(design_captain_missions.md Sec 13.A.1's own handoff decision) -- accurate enough over an
encounter's bounded few-nm/few-minute window, NEVER used for the whole-voyage distance/ETA
math (that stays in great-circle nm throughout, Sec 13.A.1's own explicit convention).

Pure Python, no GPU/API key, safe to run locally. Pure pipeline/ module -- no app/
dependency (same decoupling convention as the rest of this domain); the encounter-sim's
own Mission/Vessel objects are constructed by a caller in app/ from these primitives.
"""
from __future__ import annotations
import math

_EARTH_RADIUS_M = 6371000.0  # mean Earth radius in metres


def to_local_frame(lat: float, lon: float, origin: tuple[float, float]) -> tuple[float, float]:
    """Projects (lat, lon) to local (x, y) metres -- x=east, y=north -- equirectangular,
    centred on `origin` (lat, lon). Accurate over a few-nm window; NOT for whole-voyage
    distances (Sec 13.A.1)."""
    origin_lat, origin_lon = origin
    lat_rad = math.radians(origin_lat)
    x = math.radians(lon - origin_lon) * math.cos(lat_rad) * _EARTH_RADIUS_M
    y = math.radians(lat - origin_lat) * _EARTH_RADIUS_M
    return x, y


def from_local_frame(x: float, y: float, origin: tuple[float, float]) -> tuple[float, float]:
    """Inverse of `to_local_frame()` -- local (x, y) metres back to (lat, lon), same
    equirectangular approximation centred on `origin`."""
    origin_lat, origin_lon = origin
    lat_rad = math.radians(origin_lat)
    lat = origin_lat + math.degrees(y / _EARTH_RADIUS_M)
    lon = origin_lon + math.degrees(x / (_EARTH_RADIUS_M * math.cos(lat_rad)))
    return lat, lon
