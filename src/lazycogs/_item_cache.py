"""In-memory answers to the per-chunk item searches, from one query at open time."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np


class ItemCache:
    """The items one DuckDB query found, searchable by chunk bbox and time step.

    :func:`~lazycogs.open` queries once with the array's bbox, datetime,
    filter, ids and sortby, and keeps the results here. Each chunk's search
    then matches footprints and times as DuckDB does, in the query's order,
    without a DuckDB query per chunk and time step. The cache holds no DuckDB
    client, so arrays that use it can be pickled.
    """

    def __init__(self, items: list[dict[str, Any]]) -> None:
        """Index ``items``, as returned by ``DuckdbClient.search``."""
        self._items = items
        spans = np.array([_span(item) for item in items], dtype="datetime64[us]")
        self._starts, self._ends = spans.reshape(-1, 2).T
        self._bounds = np.array(
            [_bounds(item.get("geometry")) for item in items],
            dtype=float,
        ).reshape(-1, 4)

    def search(
        self,
        href: str,  # noqa: ARG002 - same signature as DuckdbClient.search
        *,
        bbox: list[float] | None = None,
        datetime: str | None = None,
        **_: object,
    ) -> list[dict[str, Any]]:
        """Return the items whose footprint meets ``bbox`` and time meets ``datetime``.

        The query at open time already applied the href, filter, ids and
        sortby, so the other arguments are accepted and ignored.
        """
        keep = np.ones(len(self._items), dtype=bool)
        if datetime is not None:
            start, end = _interval(datetime)
            keep &= (self._ends >= start) & (self._starts <= end)
        if bbox is not None:
            west, south, east, north = bbox
            keep &= (
                (self._bounds[:, 0] <= east)
                & (self._bounds[:, 2] >= west)
                & (self._bounds[:, 1] <= north)
                & (self._bounds[:, 3] >= south)
            )
        return [
            self._items[i]
            for i in np.flatnonzero(keep)
            if bbox is None or _intersects(self._items[i]["geometry"], bbox)
        ]


def _span(item: dict[str, Any]) -> tuple[np.datetime64, np.datetime64]:
    """Return an item's start and end, as DuckDB coalesces them with ``datetime``."""
    properties = item.get("properties", {})
    when = properties.get("datetime")
    return (
        _timestamp(properties.get("start_datetime") or when),
        _timestamp(properties.get("end_datetime") or when),
    )


def _interval(value: str) -> tuple[np.datetime64, np.datetime64]:
    """Return a datetime filter's inclusive bounds, as rustac reads them.

    A bare date runs from its midnight to 23:59:59.
    """
    start, _, end = value.partition("/")
    end = end or start
    return (
        _timestamp(start if "T" in start else f"{start}T00:00:00Z"),
        _timestamp(end if "T" in end else f"{end}T23:59:59Z"),
    )


def _timestamp(value: str | None) -> np.datetime64:
    """Return a UTC timestamp, treating one with no zone as UTC, or NaT for ``None``."""
    if value is None:
        return np.datetime64("NaT", "us")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return np.datetime64(parsed, "us")


def _bounds(geometry: dict[str, Any] | None) -> tuple[float, float, float, float]:
    """Return a GeoJSON geometry's bounds, or NaNs, which match nothing, for none."""
    if not geometry:
        return (np.nan, np.nan, np.nan, np.nan)
    points = np.concatenate(
        [
            np.asarray(ring, dtype=float)[:, :2]
            for polygon in _polygons(geometry)
            for ring in polygon
        ],
    )
    return (*points.min(axis=0), *points.max(axis=0))


def _polygons(geometry: dict[str, Any]) -> list[list[list[list[float]]]]:
    """Return a GeoJSON Polygon or MultiPolygon as a list of polygons."""
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    return geometry["coordinates"]


def _intersects(geometry: dict[str, Any], bbox: list[float]) -> bool:
    """Return whether a GeoJSON Polygon or MultiPolygon intersects ``bbox``."""
    return any(_polygon_intersects(polygon, bbox) for polygon in _polygons(geometry))


def _polygon_intersects(polygon: list[list[list[float]]], bbox: list[float]) -> bool:
    """Return whether a polygon, an exterior ring then holes, intersects ``bbox``."""
    exterior, *holes = (np.asarray(ring, dtype=float)[:, :2] for ring in polygon)
    if any(_ring_touches_box(ring, bbox) for ring in (exterior, *holes)):
        return True
    # No edge reaches the box, so it's wholly inside or outside each ring, and
    # any corner says which.
    corner = (bbox[0], bbox[1])
    return _contains(exterior, corner) and not any(
        _contains(hole, corner) for hole in holes
    )


def _ring_touches_box(ring: np.ndarray, bbox: list[float]) -> bool:
    """Return whether any edge of a ring touches the closed box, by clipping it."""
    start, delta = ring[:-1], np.diff(ring, axis=0)
    low, high = np.array(bbox[:2]), np.array(bbox[2:])
    with np.errstate(divide="ignore", invalid="ignore"):
        to_low, to_high = (low - start) / delta, (high - start) / delta
    # An edge parallel to an axis is always or never within the box's span on it.
    parallel, within = delta == 0, (start >= low) & (start <= high)
    enter = np.where(
        parallel,
        np.where(within, -np.inf, np.inf),
        np.minimum(to_low, to_high),
    )
    leave = np.where(
        parallel,
        np.where(within, np.inf, -np.inf),
        np.maximum(to_low, to_high),
    )
    return bool(
        np.any(np.maximum(enter.max(axis=1), 0) <= np.minimum(leave.min(axis=1), 1)),
    )


def _contains(ring: np.ndarray, point: tuple[float, float]) -> bool:
    """Return whether ``point`` is inside a ring, by ray casting."""
    x, y = point
    (x0, y0), (x1, y1) = ring[:-1].T, ring[1:].T
    with np.errstate(divide="ignore", invalid="ignore"):
        crosses = ((y0 > y) != (y1 > y)) & (x < x0 + (y - y0) * (x1 - x0) / (y1 - y0))
    return bool(np.count_nonzero(crosses) % 2)
