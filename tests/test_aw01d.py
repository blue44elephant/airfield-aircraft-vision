from pathlib import Path
import json
from airfield_aircraft_vision.aw01d import infer_safe, main
from airfield_aircraft_vision.optical_contracts import RasterSource


def source():
    return RasterSource(
        scene_id="x",
        sensor_id="test",
        native_gsd_m=0.5,
        width=32,
        height=32,
        sha256="a" * 64,
    )


def test_missing_weights_returns_null_count_without_predicting(tmp_path):
    def failed(*a, **k):
        raise FileNotFoundError("weights")

    result = infer_safe(
        tmp_path / "missing.tif",
        source(),
        tmp_path / "model.pt",
        detector_factory=failed,
    )
    assert result.status == "failed"
    assert result.observed_count is None
    assert result.scene_total is None
    assert result.analysis_failure_reason.startswith("model_unavailable")
    assert result.preprocessing["source_verified"] is False


def test_wrong_checkpoint_hash_fails_before_loading(tmp_path):
    p = tmp_path / "weights"
    p.write_bytes(b"wrong")

    def forbidden(*a, **k):
        raise AssertionError("should not load")

    result = infer_safe(
        tmp_path / "image",
        source(),
        p,
        detector_factory=forbidden,
        expected_sha256="0" * 64,
    )
    assert result.analysis_failure_reason == "checkpoint_sha256_mismatch"
    assert result.observed_count is None


def test_invalid_source_cli_writes_failure_not_zero_or_secret(tmp_path):
    src = tmp_path / "source.json"
    src.write_text('{"private":"SENSITIVE"}')
    output = tmp_path / "failure.json"
    status = main(
        [
            "infer",
            "--image",
            str(tmp_path / "image"),
            "--source",
            str(src),
            "--checkpoint",
            str(tmp_path / "missing"),
            "--output",
            str(output),
        ]
    )
    assert status == 2
    value = json.loads(output.read_text())
    assert value["observed_count"] is None
    assert "SENSITIVE" not in output.read_text()


def test_infer_safe_invalid_request_returns_null_result_not_exception(tmp_path):
    class Detector:
        manifest = {"selected_threshold": 0.7}

    result = infer_safe(
        tmp_path / "image",
        source(),
        tmp_path / "checkpoint",
        detector_factory=lambda *a, **k: Detector(),
        max_seconds=float("nan"),
    )
    assert result.status == "failed" and result.observed_count is None
    assert result.analysis_failure_reason.startswith("invalid_request")


def test_infer_cli_cannot_overwrite_original_checkpoint_or_input(tmp_path):
    import pytest

    for target in ("image", "source", "checkpoint", "mask", "aux"):
        paths = {key: tmp_path / key for key in ("image", "source", "checkpoint")}
        paths["mask"] = Path(str(paths["image"]) + ".msk")
        paths["aux"] = Path(str(paths["image"]) + ".aux.xml")
        for path in paths.values():
            path.write_bytes(b"preserved")
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "infer",
                    "--image",
                    str(paths["image"]),
                    "--source",
                    str(paths["source"]),
                    "--checkpoint",
                    str(paths["checkpoint"]),
                    "--output",
                    str(paths[target]),
                ]
            )
        assert exc.value.code == 2
        assert all(path.read_bytes() == b"preserved" for path in paths.values())


def test_infer_cli_requires_distinct_json_and_geojson_paths(tmp_path):
    import pytest

    output = tmp_path / "output"
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "infer",
                "--image",
                str(tmp_path / "image"),
                "--source",
                str(tmp_path / "source"),
                "--checkpoint",
                str(tmp_path / "checkpoint"),
                "--output",
                str(output),
                "--geojson",
                str(output),
            ]
        )
    assert exc.value.code == 2 and not output.exists()


def test_missing_torch_cli_still_writes_null_failure_result(tmp_path, monkeypatch):
    import sys

    metadata = tmp_path / "source.json"
    metadata.write_text(source().model_dump_json())
    output = tmp_path / "failure.json"
    monkeypatch.setitem(sys.modules, "torch", None)
    status = main(
        [
            "infer",
            "--image",
            str(tmp_path / "image"),
            "--source",
            str(metadata),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--output",
            str(output),
        ]
    )
    assert status == 2
    result = json.loads(output.read_text())
    assert result["status"] == "failed" and result["observed_count"] is None
