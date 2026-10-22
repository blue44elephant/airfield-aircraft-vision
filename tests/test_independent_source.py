import io
import json
import urllib.error

import pytest

from airfield_aircraft_vision.independent_source import (
    RangeBudget,
    RangeReader,
    frozen_json,
    json_request,
    nearest_rareplanes,
)


class Response(io.BytesIO):
    def __init__(
        self, data, *, status=206, start=0, end=None, total=None, etag='"fixed"'
    ):
        super().__init__(data)
        self.status = status
        self.headers = {
            "Content-Length": str(len(data)),
            "Content-Range": f"bytes {start}-{end if end is not None else len(data)-1}/{total or len(data)}",
            "ETag": etag,
        }


def fake_opener(content, calls):
    def opener(request, timeout):
        calls.append((request.get_header("Range"), timeout))
        first, last = map(
            int, request.get_header("Range").removeprefix("bytes=").split("-")
        )
        return Response(
            content[first : last + 1], start=first, end=last, total=len(content)
        )

    return opener


def test_bounded_range_seek_native_bytes_and_cache():
    content = bytes(range(100))
    calls = []
    budget = RangeBudget(max_bytes=100, max_requests=10)
    source = RangeReader(
        "https://example.test/data?secret=redacted",
        "https://example.test/data",
        100,
        '"fixed"',
        budget,
        opener=fake_opener(content, calls),
        block_size=16,
    )
    assert source.read(20) == content[:20]
    assert source.seek(-10, io.SEEK_CUR) == 10
    assert source.read(10) == content[10:20]
    assert len(calls) == 2
    assert source.seek(-5, io.SEEK_END) == 95
    assert source.read(50) == content[95:]
    assert source.read(1) == b""
    assert budget.used_bytes == 52
    assert "secret" not in json.dumps(budget.ranges)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", 200),
        ("Content-Length", "17"),
        ("Content-Range", "bytes 1-16/100"),
        ("ETag", '"changed"'),
    ],
)
def test_range_rejects_silent_full_download_or_changed_source(field, value):
    def opener(request, timeout):
        result = Response(b"x" * 16, start=0, end=15, total=100)
        if field == "status":
            result.status = value
        else:
            result.headers[field] = value
        return result

    source = RangeReader(
        "https://test/data?sig=secret",
        "https://test/data",
        100,
        '"fixed"',
        RangeBudget(),
        opener=opener,
        block_size=16,
    )
    with pytest.raises(ValueError):
        source.read(1)


def test_budget_checked_before_second_range_transfer():
    calls = []
    source = RangeReader(
        "https://test/data",
        "https://test/data",
        100,
        '"fixed"',
        RangeBudget(max_bytes=16),
        opener=fake_opener(bytes(100), calls),
        block_size=16,
    )
    source.read(16)
    with pytest.raises(ValueError, match="byte budget"):
        source.read(1)
    assert len(calls) == 1


def test_request_limit_and_unsigned_provenance():
    budget = RangeBudget(max_requests=1)
    budget.reserve(1)
    with pytest.raises(ValueError, match="request budget"):
        budget.reserve(1)
    with pytest.raises(ValueError, match="credentials"):
        RangeReader(
            "https://test/data?secret=1",
            "https://test/data?secret=1",
            100,
            '"fixed"',
            RangeBudget(),
        )


def test_range_never_exposes_signed_url_in_network_error():
    def fail(request, timeout):
        raise urllib.error.HTTPError(
            "https://test/data?sig=SECRET", 403, "SECRET", {}, None
        )

    source = RangeReader(
        "https://test/data?sig=SECRET",
        "https://test/data",
        100,
        '"fixed"',
        RangeBudget(),
        opener=fail,
    )
    with pytest.raises(RuntimeError, match="403") as caught:
        source.read(1)
    assert "SECRET" not in str(caught.value)


def test_metadata_cap_and_secret_redaction():
    with pytest.raises(ValueError, match="bounded metadata"):
        json_request("https://test", cap=5, opener=lambda *a, **kw: Response(b"123456"))

    def fail(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://test?secret=SECRET", 429, "SECRET", {}, None
        )

    with pytest.raises(RuntimeError) as caught:
        json_request("https://test", opener=fail)
    assert "SECRET" not in str(caught.value)


def test_frozen_evidence_refuses_overwrite(tmp_path):
    path = tmp_path / "reservation.json"
    frozen_json(path, {"a": 1})
    frozen_json(path, {"a": 1})
    with pytest.raises(ValueError, match="preserved file"):
        frozen_json(path, {"a": 2})


def test_all_rareplanes_geographic_independence(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "geometry": {
                            "coordinates": [[[-100, 30], [-99, 31], [-100, 30]]]
                        },
                        "properties": {"loc_id": 7, "cat_id": "known"},
                    }
                ]
            }
        )
    )
    assert nearest_rareplanes(path, -109, 44)["distance_m"] > 100_000
    with pytest.raises(ValueError, match="100km"):
        nearest_rareplanes(path, -100, 30)


def test_rasterio_opener_native_grid_and_independent_cursors():
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    from airfield_aircraft_vision.independent_source import make_range_opener

    pixels = np.arange(4 * 20 * 30, dtype=np.uint8).reshape(4, 20, 30)
    affine = from_origin(500000, 4000000, 0.6, 0.6)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=30,
            height=20,
            count=4,
            dtype="uint8",
            crs="EPSG:26912",
            transform=affine,
        ) as source:
            source.write(pixels)
        content = memory.read()
    calls = []
    budget = RangeBudget()
    opener = make_range_opener(
        "https://test/image?sig=SECRET",
        "https://test/image",
        len(content),
        '"fixed"',
        budget,
        transport=fake_opener(content, calls),
    )
    with opener("independent-naip.tif") as a, opener("independent-naip.tif") as b:
        assert a.read(10) == content[:10]
        assert b.tell() == 0
        assert b.read(10) == content[:10]
    with pytest.raises(FileNotFoundError):
        opener("test")
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open("independent-naip.tif", opener=opener) as source:
            window = Window(2, 3, 10, 8)
            actual = source.read((1, 2, 3), window=window)
            assert actual.shape == (3, 8, 10)
            assert np.array_equal(actual, pixels[:3, 3:11, 2:12])
            assert source.window_transform(
                window
            ) == affine * rasterio.Affine.translation(2, 3)
    assert len(calls) == 1


def test_native_window_selected_from_metadata_before_pixels_and_clamped_at_edge():
    from airfield_aircraft_vision.independent_source import native_window_grid

    item = {
        "id": "test",
        "properties": {
            "proj:transform": [0.00001, 0, -110, 0, -0.00001, 40],
            "proj:shape": [2000, 2000],
            "proj:epsg": 4326,
        },
    }
    result = native_window_grid(
        item, {"name": "airport", "lon": -109.981, "lat": 39.99}
    )
    assert result["window"] == [976, 488, 1024, 1024]
    assert result["origin_adjustment_pixels"][0] < 0
    assert result["resampling"] is False


def test_partial_retrieval_stops_before_network(tmp_path, monkeypatch):
    from airfield_aircraft_vision import independent_source as module

    output = tmp_path / "output"
    output.mkdir()
    (output / "cody.tif").write_bytes(b"preserved")
    monkeypatch.setattr(
        module, "json_request", lambda *a, **kw: pytest.fail("network access")
    )
    with pytest.raises(FileExistsError, match="partial independent"):
        module.prepare_independent(
            tmp_path / "annotations.json", output, tmp_path / "evidence"
        )
    assert (output / "cody.tif").read_bytes() == b"preserved"


def test_full_native_extraction_keeps_rgb_values_grid_and_internal_mask(
    tmp_path, monkeypatch
):
    import hashlib
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from airfield_aircraft_vision import independent_source as module

    pixels = np.arange(4 * 1100 * 1100, dtype=np.uint8).reshape(4, 1100, 1100)
    affine = from_origin(500000, 4000000, 0.6, 0.6)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=1100,
            height=1100,
            count=4,
            dtype="uint8",
            crs="EPSG:26912",
            transform=affine,
        ) as dataset:
            dataset.write(pixels)
            dataset.write_mask(np.full((1100, 1100), 255, dtype=np.uint8))
            dataset.colorinterp = (
                rasterio.enums.ColorInterp.red,
                rasterio.enums.ColorInterp.green,
                rasterio.enums.ColorInterp.blue,
                rasterio.enums.ColorInterp.undefined,
            )
        content = memory.read()
    calls = []
    budget = RangeBudget()
    true_opener = module.make_range_opener
    monkeypatch.setattr(
        module,
        "make_range_opener",
        lambda signed, unsigned, size, etag, budget: true_opener(
            signed, unsigned, size, etag, budget, transport=fake_opener(content, calls)
        ),
    )

    def head(*args, **kwargs):
        response = Response(b"", status=200)
        response.headers["Content-Length"] = str(len(content))
        return response

    monkeypatch.setattr(module.urllib.request, "urlopen", head)
    item = {
        "id": "known-item",
        "properties": {
            "proj:epsg": 26912,
            "proj:shape": [1100, 1100],
            "proj:transform": list(affine)[:6],
            "datetime": "2022-01-01T16:00:00Z",
        },
    }
    reservation = {
        "name": "known-airport",
        "source_uri": "https://test/native.tif",
        "native_window": [12, 23, 1024, 1024],
    }
    result = module.extract_native(
        item, reservation, "https://test/native.tif?sig=SECRET", tmp_path, budget
    )
    expected = pixels[:3, 23:1047, 12:1036]
    with rasterio.open(result["path"]) as dataset:
        assert np.array_equal(dataset.read(), expected)
        assert dataset.transform == affine * rasterio.Affine.translation(12, 23)
        assert not any(p.endswith(".msk") for p in dataset.files)
    assert result["source"]["timestamp_precision"] == "date"
    assert result["source"]["acquisition_timestamp"] is None
    assert result["source"]["native_gsd_m"] == 0.6
    assert result["pixels_sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
    assert "SECRET" not in json.dumps(result)


def test_extra_apron_uses_remaining_budget_and_never_retries_incomplete(
    tmp_path, monkeypatch
):
    from airfield_aircraft_vision import independent_source as module

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    initial = {"range_bytes_reserved": 10_747_904, "range_requests": 41}
    monkeypatch.setattr(module, "verify_cached_retrieval", lambda *a: initial)
    contract = {
        "raw_items": [{"id": module.LOCATIONS[1]["item"]}],
        "locations": [{"name": "cody", "source_uri": "https://test/cody.tif"}],
    }
    (evidence / "independent-source-reservation.json").write_text(json.dumps(contract))
    monkeypatch.setattr(module, "json_request", lambda *a, **kw: {"token": "SECRET"})

    def failed(item, reservation, signed_url, output, budget):
        assert reservation["native_window"] == [5658, 8394, 1024, 1024]
        assert budget.max_bytes == 5 * 1024**2
        assert budget.max_requests == 20
        budget.reserve(1024)
        raise ValueError("secret=SECRET must never appear in failure evidence")

    monkeypatch.setattr(module, "extract_native", failed)
    with pytest.raises(RuntimeError, match="no automatic retry"):
        module.prepare_extra_cody_ga(
            tmp_path / "annotations", tmp_path / "data", evidence
        )
    record = (evidence / "independent-source-ga-retrieval.json").read_text()
    assert "SECRET" not in record
    assert json.loads(record)["status"] == "incomplete"
    monkeypatch.setattr(
        module, "json_request", lambda *a, **kw: pytest.fail("automatic network retry")
    )
    with pytest.raises(ValueError, match="do not repeat automatically"):
        module.prepare_extra_cody_ga(
            tmp_path / "annotations", tmp_path / "data", evidence
        )
