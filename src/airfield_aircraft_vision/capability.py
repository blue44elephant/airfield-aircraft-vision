"""Conservative, explainable capability routing from scene metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite


class SensorDomain(StrEnum):
    OPTICAL = "optical"
    SAR = "sar"
    UNKNOWN = "unknown"


class ImageQuality(StrEnum):
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"
    UNKNOWN = "unknown"


class CapabilityTier(StrEnum):
    A = "tier_a"
    B = "tier_b"
    C = "tier_c"
    SAR_FUTURE = "sar_future"
    UNAVAILABLE = "unavailable"


TIER_A_ALLOWED = (
    "individual_aircraft_detection",
    "oriented_bounding_boxes",
    "approximate_aircraft_orientation",
    "broad_aircraft_morphology_classification",
    "per_class_counts",
    "confidence_and_uncertainty",
    "aircraft_family_candidates_if_empirically_validated",
)
TIER_B_ALLOWED = (
    "aircraft_like_object_detection",
    "approximate_individual_counts_if_separated",
    "broad_size_categories",
    "fixed_wing_vs_rotary_wing_if_empirically_validated",
    "large_vs_small_aircraft",
    "occupied_parking_regions",
    "uncertainty_aware_count_ranges",
    "unknown_class",
)
TIER_C_ALLOWED = (
    "apron_activity",
    "parking_stand_occupancy",
    "dense_aircraft_storage_detection",
    "large_aircraft_presence",
    "temporal_change",
    "approximate_count_ranges_if_validated",
    "confidence_and_image_quality_reporting",
)

TIER_A_FORBIDDEN = (
    "exact_aircraft_model_identification_without_validation",
)
TIER_B_FORBIDDEN = (
    "exact_aircraft_model_identification",
    "fighter_bomber_trainer_or_family_labels_without_resolution_specific_validation",
)
TIER_C_FORBIDDEN = (
    "per_aircraft_boxes_for_unresolved_objects",
    "individual_aircraft_morphology_classification",
    "fighter_bomber_transport_or_helicopter_labels",
    "exact_aircraft_model_identification",
)
ALL_OPTICAL_TASKS_FORBIDDEN = tuple(
    dict.fromkeys(TIER_A_ALLOWED + TIER_B_ALLOWED + TIER_C_ALLOWED)
)


@dataclass(frozen=True)
class SceneMetadata:
    sensor_domain: SensorDomain | str
    native_gsd_m: float | None = None
    effective_spatial_resolution_m: float | None = None
    panchromatic_gsd_m: float | None = None
    pan_sharpened_gsd_m: float | None = None
    artificially_upscaled_gsd_m: float | None = None
    super_resolved_gsd_m: float | None = None
    resolution_metadata_reliable: bool = True
    image_quality: ImageQuality | str = ImageQuality.UNKNOWN


@dataclass(frozen=True)
class CapabilityDecision:
    tier: CapabilityTier
    allowed_tasks: tuple[str, ...]
    forbidden_tasks: tuple[str, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""
    governing_resolution_m: float | None = None


def _enum_value(value: SensorDomain | ImageQuality | str) -> str:
    return value.value if isinstance(value, StrEnum) else str(value).lower()


def _valid_resolution(value: float | None) -> bool:
    return value is not None and isfinite(value) and value > 0


def _resolution_for_routing(scene: SceneMetadata, warnings: list[str]) -> float | None:
    native = scene.native_gsd_m
    effective = scene.effective_spatial_resolution_m
    pan = scene.pan_sharpened_gsd_m
    panchromatic = scene.panchromatic_gsd_m

    if pan is not None or panchromatic is not None:
        if not (_valid_resolution(pan) and _valid_resolution(panchromatic)):
            warnings.append(
                "Pan-sharpened resolution is not trusted without valid output and source panchromatic GSD metadata."
            )
        else:
            warnings.append(
                "Pan-sharpening can carry genuine panchromatic geometry but does not create multispectral detail; use only validated pan-sharpened models."
            )
            physical = max(float(pan), float(panchromatic))
            if _valid_resolution(effective):
                physical = max(physical, float(effective))
            return physical

    if not _valid_resolution(native):
        return None
    physical = float(native)
    if _valid_resolution(effective):
        physical = max(physical, float(effective))
        if float(effective) > float(native):
            warnings.append(
                "Effective spatial resolution is poorer than pixel GSD and governs routing."
            )
    return physical


def _tier_for_gsd(gsd_m: float) -> CapabilityTier:
    if gsd_m <= 1.0:
        return CapabilityTier.A
    if gsd_m <= 5.0:
        return CapabilityTier.B
    return CapabilityTier.C


def _degrade_for_poor_quality(tier: CapabilityTier) -> CapabilityTier:
    return {
        CapabilityTier.A: CapabilityTier.B,
        CapabilityTier.B: CapabilityTier.C,
        CapabilityTier.C: CapabilityTier.C,
    }[tier]


def _tasks(tier: CapabilityTier) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return {
        CapabilityTier.A: (TIER_A_ALLOWED, TIER_A_FORBIDDEN),
        CapabilityTier.B: (TIER_B_ALLOWED, TIER_B_FORBIDDEN),
        CapabilityTier.C: (TIER_C_ALLOWED, TIER_C_FORBIDDEN),
    }[tier]


def route_scene(scene: SceneMetadata) -> CapabilityDecision:
    """Route a scene without inferring missing physical metadata.

    Quality `poor` conservatively lowers A to B and B to C. Tier C remains C,
    with warnings, because its outputs are already aggregate and uncertainty-aware.
    `unusable` blocks analysis. Unknown quality does not silently lower a tier, but
    makes the decision provisional via an explicit warning.
    """

    sensor = _enum_value(scene.sensor_domain)
    quality = _enum_value(scene.image_quality)
    warnings: list[str] = []

    if scene.artificially_upscaled_gsd_m is not None:
        warnings.append(
            "Artificial upscaling changes pixel spacing, not information content, and is ignored for capability routing."
        )
    if scene.super_resolved_gsd_m is not None:
        warnings.append(
            "Super-resolution does not raise the capability tier; its output resolution is ignored for routing."
        )

    if sensor == SensorDomain.SAR.value:
        return CapabilityDecision(
            tier=CapabilityTier.SAR_FUTURE,
            allowed_tasks=("route_to_future_sar_pipeline",),
            forbidden_tasks=ALL_OPTICAL_TASKS_FORBIDDEN,
            warnings=tuple(warnings + ["SAR is a separate sensor domain; no SAR detector is implemented in VISION-0."]),
            reason="SAR imagery must not be sent to an optical aircraft model.",
        )
    if sensor != SensorDomain.OPTICAL.value:
        return CapabilityDecision(
            tier=CapabilityTier.UNAVAILABLE,
            allowed_tasks=("metadata_review",),
            forbidden_tasks=ALL_OPTICAL_TASKS_FORBIDDEN,
            warnings=tuple(warnings + ["Sensor domain is unknown or unsupported."]),
            reason="A reliable optical or SAR sensor-domain declaration is required.",
        )
    if quality == ImageQuality.UNUSABLE.value:
        return CapabilityDecision(
            tier=CapabilityTier.UNAVAILABLE,
            allowed_tasks=("select_alternate_scene",),
            forbidden_tasks=ALL_OPTICAL_TASKS_FORBIDDEN,
            warnings=tuple(warnings + ["Image quality is unusable for automated analysis."]),
            reason="Quality gating rejected the scene before model routing.",
        )
    if not scene.resolution_metadata_reliable:
        return CapabilityDecision(
            tier=CapabilityTier.UNAVAILABLE,
            allowed_tasks=("metadata_review",),
            forbidden_tasks=ALL_OPTICAL_TASKS_FORBIDDEN,
            warnings=tuple(warnings + ["Resolution metadata is explicitly marked unreliable; no GSD is guessed."]),
            reason="Reliable physical-resolution metadata is required for capability selection.",
        )

    governing = _resolution_for_routing(scene, warnings)
    if governing is None:
        return CapabilityDecision(
            tier=CapabilityTier.UNAVAILABLE,
            allowed_tasks=("metadata_review",),
            forbidden_tasks=ALL_OPTICAL_TASKS_FORBIDDEN,
            warnings=tuple(warnings + ["Native or defensible physical-resolution metadata is missing or invalid."]),
            reason="The system does not replace missing GSD with a guessed value.",
        )

    tier = _tier_for_gsd(governing)
    reason = f"Optical physical resolution of {governing:g} m supports {tier.value}."
    if quality == ImageQuality.POOR.value:
        degraded = _degrade_for_poor_quality(tier)
        warnings.append(
            f"Poor image quality lowers usable capability from {tier.value} to {degraded.value}."
        )
        tier = degraded
        reason += f" Poor quality applies a conservative gate to {tier.value}."
    elif quality == ImageQuality.UNKNOWN.value:
        warnings.append(
            "Image quality has not been assessed; routing is provisional until cloud, haze, shadow, and obstruction checks complete."
        )
    elif quality != ImageQuality.ACCEPTABLE.value:
        warnings.append("Unrecognized image-quality value; treating quality as unknown.")

    allowed, forbidden = _tasks(tier)
    return CapabilityDecision(
        tier=tier,
        allowed_tasks=allowed,
        forbidden_tasks=forbidden,
        warnings=tuple(warnings),
        reason=reason,
        governing_resolution_m=governing,
    )

