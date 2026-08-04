from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from airfield_aircraft_vision.capability import CapabilityTier
from airfield_aircraft_vision.schemas import (
    Detection,
    DetectionStatus,
    GeoJSONGeometry,
    ImageQualityAssessment,
    InferenceResult,
    OrientedBox,
)


def detection(
    detection_id: str,
    predicted_class: str,
    probabilities: dict[str, float],
) -> Detection:
    return Detection(
        detection_id=detection_id,
        status=DetectionStatus.ACCEPTED,
        oriented_box=OrientedBox(
            center_x=100,
            center_y=200,
            width=30,
            height=20,
            angle_degrees_clockwise_from_north=45,
        ),
        predicted_class=predicted_class,
        class_probabilities=probabilities,
        confidence=max(probabilities.values()),
    )


def valid_payload() -> dict:
    known = detection("d-1", "combat_jet", {"combat_jet": 0.8, "unknown_aircraft": 0.2})
    unknown = detection(
        "d-2", "unknown_aircraft", {"combat_jet": 0.1, "unknown_aircraft": 0.9}
    )
    rejected = Detection(
        detection_id="d-3",
        status=DetectionStatus.REJECTED,
        rejection_reason="duplicate_after_oriented_nms",
    )
    return {
        "aoi": GeoJSONGeometry(
            type="Polygon",
            coordinates=[[[77.0, 28.0], [77.1, 28.0], [77.1, 28.1], [77.0, 28.0]]],
        ),
        "source_scene_id": "scene-1",
        "sensor_id": "sensor-1",
        "acquisition_timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "native_resolution_m": 0.5,
        "selected_capability_tier": CapabilityTier.A,
        "image_quality": ImageQualityAssessment(status="acceptable"),
        "detections": [known, unknown, rejected],
        "total_detection_count": 2,
        "total_candidate_detections": 3,
        "known_classified_detections": 1,
        "unknown_detections": 1,
        "rejected_detections": 1,
        "per_class_totals": {"combat_jet": 1, "unknown_aircraft": 1},
        "unavailable_features": ["exact_aircraft_model_identification"],
        "unavailable_feature_reasons": {
            "exact_aircraft_model_identification": "Not an initial validated output"
        },
    }


def test_valid_result_reconciles_known_unknown_rejected_and_accepted() -> None:
    result = InferenceResult.model_validate(valid_payload())
    assert result.total_detection_count == 2
    assert result.known_classified_detections + result.unknown_detections == 2
    assert result.total_detection_count + result.rejected_detections == 3


def test_inconsistent_per_class_total_is_rejected() -> None:
    payload = valid_payload()
    payload["known_classified_detections"] = 2
    with pytest.raises(ValidationError, match="known per-class totals"):
        InferenceResult.model_validate(payload)


def test_inconsistent_detection_records_are_rejected() -> None:
    payload = valid_payload()
    payload["detections"] = payload["detections"][:1]
    with pytest.raises(ValidationError, match="detection records"):
        InferenceResult.model_validate(payload)


def test_unavailable_feature_requires_reason() -> None:
    payload = valid_payload()
    payload["unavailable_feature_reasons"] = {}
    with pytest.raises(ValidationError, match="lack reasons"):
        InferenceResult.model_validate(payload)


def test_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1"):
        detection("d-bad", "combat_jet", {"combat_jet": 0.4, "unknown_aircraft": 0.4})


def test_acquisition_timestamp_must_be_timezone_aware() -> None:
    payload = valid_payload()
    payload["acquisition_timestamp"] = datetime(2026, 1, 1)
    with pytest.raises(ValidationError):
        InferenceResult.model_validate(payload)


def test_annotation_schema_is_machine_readable_and_supports_redaction() -> None:
    schema_path = Path(__file__).parents[1] / "schemas" / "aircraft_annotation.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["$schema"].endswith("2020-12/schema")
    assert "coordinates_redacted" in schema["required"]
    assert "aoi" not in schema["required"]
    assert "geographic_geometry" not in schema["required"]
    assert "unknown_aircraft" in schema["properties"]["class_label"]["enum"]

