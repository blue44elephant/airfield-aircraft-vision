"""No-ML regressions for the single fixed AW-01D correction pilot."""

from types import SimpleNamespace
import pytest
from airfield_aircraft_vision.robustness_correction import (
    CONFIGURATION,
    FIXED_THRESHOLD,
    validate_pilot_manifest,
    union_training_lineage,
)


def test_configuration_is_one_fixed_small_validation_only_pilot():
    assert CONFIGURATION == {
        "architecture": "fasterrcnn_resnet50_fpn",
        "epochs": 1,
        "lr": 0.00005,
        "threads": 4,
        "max_seconds": 90,
        "seed": 91,
    }
    assert FIXED_THRESHOLD == 0.7


def test_pilot_rejects_test_labels_before_training():
    with pytest.raises(ValueError, match="forbids test"):
        validate_pilot_manifest(
            SimpleNamespace(samples=[SimpleNamespace(split="test")])
        )


def test_lineage_union_preserves_parent_and_new_training_exposure():
    parent = {
        "training_airfields": ["old"],
        "training_captures": ["old-capture"],
        "training_source_checksums": ["old-image"],
        "validation_airfields": ["old-val"],
    }
    child = {
        "training_airfields": ["new"],
        "training_captures": ["new-capture"],
        "training_source_checksums": ["new-image"],
        "selected_threshold": 0.4,
        "selected_validation_metrics": {"f1": 0.5},
    }
    merged = union_training_lineage(parent, child, "a" * 64)
    assert merged["training_airfields"] == ["new", "old"]
    assert merged["training_captures"] == ["new-capture", "old-capture"]
    assert merged["training_source_checksums"] == ["new-image", "old-image"]
    assert merged["parent_checkpoint_sha256"] == "a" * 64
    assert merged["selected_threshold"] == 0.7
    assert merged["training_helper_validation_selection"]["threshold"] == 0.4
    assert parent["training_airfields"] == ["old"]
    assert child["training_airfields"] == ["new"]


def test_missing_parent_lineage_is_not_silently_dropped():
    with pytest.raises(ValueError, match="auditable training lineage"):
        union_training_lineage({}, {}, "a" * 64)
