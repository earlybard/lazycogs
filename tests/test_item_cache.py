"""Tests for the in-memory item cache that answers per-chunk searches."""

from __future__ import annotations

import itertools
import pickle
from unittest.mock import patch

import numpy as np
import pytest
import rustac
from pyproj import Transformer

import lazycogs
from lazycogs._item_cache import ItemCache

INCLUDE = ["id", "geometry", "datetime", "start_datetime", "end_datetime", "assets"]


def _item(item_id: str, geometry: dict, **times: str | None) -> dict:
    points = np.concatenate(
        [
            np.asarray(ring)
            for ring in (
                geometry["coordinates"]
                if geometry["type"] == "Polygon"
                else [r for p in geometry["coordinates"] for r in p]
            )
        ],
    )
    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": item_id,
        "geometry": geometry,
        "bbox": [float(v) for v in (*points.min(axis=0), *points.max(axis=0))],
        "properties": {"datetime": None, **times},
        "links": [],
        "assets": {"B04": {"href": f"file:///tmp/{item_id}.tif"}},
    }


def _polygon(*rings: list[tuple[float, float]]) -> dict:
    return {"type": "Polygon", "coordinates": [[*ring, ring[0]] for ring in rings]}


ITEMS = [
    _item(
        "square",
        _polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        datetime="2026-01-15T00:00:00Z",
    ),
    # Its bbox covers (3, 3)-(4, 4); its footprint doesn't.
    _item(
        "triangle",
        _polygon([(0, 0), (4, 0), (0, 4)]),
        datetime="2026-01-15T10:30:00Z",
    ),
    _item(
        "holed",
        _polygon(
            [(10, 10), (20, 10), (20, 20), (10, 20)],
            [(12, 12), (18, 12), (18, 18), (12, 18)],
        ),
        datetime="2026-01-15T23:59:59.500000Z",
    ),
    _item(
        "multi",
        {
            "type": "MultiPolygon",
            "coordinates": [
                [[[30, 0], [31, 0], [31, 1], [30, 1], [30, 0]]],
                [[[35, 0], [36, 0], [36, 1], [35, 1], [35, 0]]],
            ],
        },
        datetime="2026-01-16T00:00:00Z",
    ),
    _item(
        "span",
        _polygon([(0, 10), (2, 10), (2, 12), (0, 12)]),
        datetime="2026-01-12T00:00:00Z",
        start_datetime="2026-01-10T00:00:00Z",
        end_datetime="2026-01-14T12:00:00Z",
    ),
]

BBOXES = [
    None,
    [0.5, 0.5, 0.6, 0.6],  # inside the square and triangle
    [3, 3, 4, 4],  # in the triangle's bbox, outside its footprint
    [1, 1, 2, 2],  # touches the square's corner
    [14, 14, 16, 16],  # inside the hole
    [11, 11, 13, 13],  # across the hole's edge
    [9, 9, 21, 21],  # around the holed square
    [32, 0, 34, 1],  # between the multipolygon's parts
    [35.5, 0.5, 40, 2],  # over its second part
    [1.5, 11, 5, 11.5],  # over the span's edge
]

DATETIMES = [
    None,
    "2026-01-15",
    "2026-01-14/2026-01-15",
    "2026-01-15/2026-01-16",
    "2026-01-13",
    "2026-01-15T00:00:00Z",
    "2026-01-15T10:00:00Z/2026-01-15T10:59:59Z",
    "2026-01-01/2026-01-31",
]


@pytest.fixture(scope="module")
def parquet(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("cache") / "items.parquet")
    rustac.write_sync(path, ITEMS)
    return path


@pytest.fixture(scope="module")
def client() -> rustac.DuckdbClient:
    return rustac.DuckdbClient()


@pytest.mark.parametrize(
    ("bbox", "datetime"),
    list(itertools.product(BBOXES, DATETIMES)),
)
def test_item_cache_finds_what_duckdb_finds(parquet, client, bbox, datetime):
    """The cache finds the same items as DuckDB, in the same order."""
    cache = ItemCache(client.search(parquet, sortby="-id", include=INCLUDE))

    expected = client.search(
        parquet,
        bbox=bbox,
        datetime=datetime,
        sortby="-id",
        include=INCLUDE,
    )

    found = cache.search(parquet, bbox=bbox, datetime=datetime)
    assert [item["id"] for item in found] == [item["id"] for item in expected]


def test_open_reads_chunks_without_querying_duckdb(synthetic_cog, tmp_path):
    """Chunk reads find their items in the cache, and the array pickles."""
    west, south, east, north = Transformer.from_crs(
        "EPSG:32632",
        "EPSG:4326",
        always_xy=True,
    ).transform_bounds(500_000, 5_579_520, 520_480, 5_600_000)
    item = _item(
        "synthetic",
        _polygon([(west, south), (east, south), (east, north), (west, north)]),
        datetime="2023-01-15T10:00:00Z",
    )
    item["assets"]["B04"]["href"] = synthetic_cog.as_uri()
    parquet = str(tmp_path / "items.parquet")
    rustac.write_sync(parquet, [item])
    da = lazycogs.open(
        parquet,
        bbox=(500_000, 5_599_360, 500_640, 5_600_000),
        crs="EPSG:32632",
        resolution=10.0,
        chunks={"x": 32, "y": 32},
    )

    with patch("rustac.DuckdbClient.search", side_effect=AssertionError):
        values = da.values

    assert (values != 0).all()
    restored = pickle.loads(pickle.dumps(da))  # noqa: S301
    np.testing.assert_array_equal(restored.values, values)
