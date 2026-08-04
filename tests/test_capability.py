from __future__ import annotations

import pytest

from airfield_aircraft_vision.capability import (
    CapabilityTier,
    ImageQuality,
    SceneMetadata,
    SensorDomain,
    route_scene,
)


@pytest.mark.parametrize(
    ("gsd", "expected"),
    [
        (0.3, CapabilityTier.A),
        (1.0, CapabilityTier.A),
        (1.0001, CapabilityTier.B),
        (5.0, CapabilityTier.B),
        (5.0001, CapabilityTier.C),
        (10.0, CapabilityTier.C),
    ],
)
def test_optical_boundaries(gsd: float, expected: CapabilityTier) -> None:
    decision = route_scene(
        SceneMetadata(
            sensor_domain=SensorDomain.OPTICAL,
            native_gsd_m=gsd,
            image_quality=ImageQuality.ACCEPTABLE,
        )
    )
    assert decision.tier == expected
    assert decision.governing_resolution_m == gsd


@pytest.mark.parametrize("gsd", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_missing_or_invalid_resolution_blocks_routing(gsd: float | None) -> None:
    decision = route_scene(SceneMetadata(sensor_domain="optical", native_gsd_m=gsd))
    assert decision.tier == CapabilityTier.UNAVAILABLE
    assert decision.allowed_tasks == ("metadata_review",)
    assert "guessed" in decision.reason.lower()


def test_ten_meter_upscaled_to_one_meter_stays_tier_c() -> None:
    decision = route_scene(
        SceneMetadata(
            sensor_domain="optical",
            native_gsd_m=10.0,
            artificially_upscaled_gsd_m=1.0,
            image_quality="acceptable",
        )
    )
    assert decision.tier == CapabilityTier.C
    assert "individual_aircraft_morphology_classification" in decision.forbidden_tasks
    assert any("upscaling" in warning.lower() for warning in decision.warnings)


def test_super_resolution_never_raises_tier() -> None:
    decision = route_scene(
        SceneMetadata(
            sensor_domain="optical",
            native_gsd_m=10.0,
            super_resolved_gsd_m=0.5,
            image_quality="acceptable",
        )
    )
    assert decision.tier == CapabilityTier.C
    assert any("super-resolution" in warning.lower() for warning in decision.warnings)


def test_effective_resolution_can_only_make_routing_more_conservative() -> None:
    decision = route_scene(
        SceneMetadata(
            sensor_domain="optical",
            native_gsd_m=0.5,
            effective_spatial_resolution_m=2.0,
            image_quality="acceptable",
        )
    )
    assert decision.tier == CapabilityTier.B
    assert decision.governing_resolution_m == 2.0


def test_legitimate_pan_sharpened_geometry_uses_physical_pan_metadata() -> None:
    decision = route_scene(
        SceneMetadata(
            sensor_domain="optical",
            native_gsd_m=2.0,
            panchromatic_gsd_m=0.4,
            pan_sharpened_gsd_m=0.5,
            effective_spatial_resolution_m=0.7,
            image_quality="acceptable",
        )
    )
    assert decision.tier == CapabilityTier.A
    assert decision.governing_resolution_m == 0.7
    assert any("multispectral detail" in warning.lower() for warning in decision.warnings)


def test_pan_sharpened_claim_without_source_pan_metadata_does_not_upgrade() -> None:
    decision = route_scene(
        SceneMetadata(
            sensor_domain="optical",
            native_gsd_m=2.0,
            pan_sharpened_gsd_m=0.5,
            image_quality="acceptable",
        )
    )
    assert decision.tier == CapabilityTier.B
    assert decision.governing_resolution_m == 2.0
    assert any("not trusted" in warning.lower() for warning in decision.warnings)


def test_sar_routes_to_separate_future_pipeline() -> None:
    decision = route_scene(
        SceneMetadata(sensor_domain="sar", native_gsd_m=0.5, image_quality="acceptable")
    )
    assert decision.tier == CapabilityTier.SAR_FUTURE
    assert decision.allowed_tasks == ("route_to_future_sar_pipeline",)
    assert "individual_aircraft_detection" in decision.forbidden_tasks
    assert "optical" in decision.reason.lower()


def test_unknown_sensor_domain_is_unavailable() -> None:
    decision = route_scene(SceneMetadata(sensor_domain="hyperspace", native_gsd_m=0.5))
    assert decision.tier == CapabilityTier.UNAVAILABLE
    assert any("unsupported" in warning.lower() for warning in decision.warnings)


@pytest.mark.parametrize(
    ("gsd", "expected"),
    [(0.5, CapabilityTier.B), (2.0, CapabilityTier.C), (10.0, CapabilityTier.C)],
)
def test_poor_quality_conservatively_reduces_capability(
    gsd: float, expected: CapabilityTier
) -> None:
    decision = route_scene(
        SceneMetadata(sensor_domain="optical", native_gsd_m=gsd, image_quality="poor")
    )
    assert decision.tier == expected
    assert any("poor image quality" in warning.lower() for warning in decision.warnings)


def test_unusable_quality_blocks_inference() -> None:
    decision = route_scene(
        SceneMetadata(sensor_domain="optical", native_gsd_m=0.5, image_quality="unusable")
    )
    assert decision.tier == CapabilityTier.UNAVAILABLE
    assert decision.allowed_tasks == ("select_alternate_scene",)


def test_unreliable_resolution_metadata_is_not_guessed() -> None:
    decision = route_scene(
        SceneMetadata(
            sensor_domain="optical",
            native_gsd_m=0.5,
            resolution_metadata_reliable=False,
        )
    )
    assert decision.tier == CapabilityTier.UNAVAILABLE
    assert any("unreliable" in warning.lower() for warning in decision.warnings)

