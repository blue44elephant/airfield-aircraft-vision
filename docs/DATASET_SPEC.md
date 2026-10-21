# Dataset Specification

> AW-01 status: a bounded experimental high-resolution optical detector, preparation/training/evaluation CLI and regression tests now implement part of this VISION-0 design. See [AW-01 validation](AW01_VALIDATION.md) and [workflows](AW01_WORKFLOWS.md). Other future components below remain unimplemented.

## Goals

The dataset supports native high-resolution oriented aircraft annotations, controlled multi-resolution derivatives, and real low-resolution calibration/evaluation without airfield, scene, crop, geographic, or background leakage. It is a manifest-driven specification; VISION-0 downloads no imagery.

## Logical layout

```text
dataset_root/
  dataset_manifest.json
  taxonomy.json
  sources/
    <provider>/<scene_id>/
      scene_manifest.json
      imagery/                 # licensed original assets or external references
      quality/                 # masks/reports: cloud, shadow, haze, obstruction
  annotations/
    native_high_resolution/
      <scene_id>.jsonl         # aircraft_annotation.schema.json records
    negatives/
      <scene_id>.json
  derivatives/
    degraded/
      <degradation_profile_id>/<source_scene_id>/
        derivative_manifest.json
        imagery/
        annotations/           # transformed boxes linked to native IDs
  validation/
    real_low_resolution/
      <sensor>/<scene_id>/
        scene_manifest.json
        imagery_or_reference/
        annotations_or_regions/
  splits/
    split_manifest_v<N>.json
    train_airfields.txt
    validation_airfields.txt
    test_airfields.txt
    geographic_holdouts.txt
  audits/
    provenance_report.json
    duplicate_groups.json
    leakage_report.json
    licence_report.json
```

Large imagery may remain outside Git/object storage; manifests contain checksums, durable source references, licence/use restrictions, and local logical IDs. Paths are never the only provenance.

## Scene and image metadata

Every source scene has a stable `scene_id`, provider collection/item IDs, sensor/platform/domain, UTC acquisition time and precision, footprint/AOI, CRS/geotransform, dimensions, native GSD per band, effective spatial resolution when known, panchromatic/pan-sharpen metadata, band names/wavelengths, radiometric depth/scaling, processing level, nodata, cloud metadata, off-nadir/view angles when available, sun angles, licence, checksum, and lineage.

Each derived crop/image has `image_id`, parent `scene_id`, exact pixel window, transform, crop checksum, airfield group, geographic group, duplicate/capture group, split, and coordinate-redaction state. Unknown values remain null with provenance notes.

## Native annotations

Records validate against `schemas/aircraft_annotation.schema.json`. Individually resolved aircraft use pixel-space oriented boxes (center, width, height, clockwise angle from image-up), one broad class, optional family candidate, weak-label confidence where applicable, truncation fraction, occlusion, shadow severity, quality, annotation source, review state, notes, and stable annotation ID.

Broad classes are:

- `combat_jet`
- `bomber_or_large_combat_aircraft`
- `transport_or_tanker`
- `special_mission_aircraft`
- `trainer_or_light_aircraft`
- `helicopter`
- `uav_like`
- `unknown_aircraft`

Exact type is not a required target. `unknown_aircraft` is retained in training/evaluation. Family candidates are separate, nullable, and never silently promoted to ground truth. Public exports can remove/geographically generalize AOI and aircraft geometry while retaining image-space boxes and `coordinates_redacted: true`.

## Grouping and split units

The atomic split unit is an `airfield_group`, covering the airfield and adjacent operational surfaces across all providers, dates, crops, processing levels, and derivatives. Nearby airfields or visually/geographically coupled sites can be merged into a larger `geographic_group`. A `capture_group` joins bursts, mosaics, tiles, reprocessed versions, near-contemporaneous scenes, and perceptual near-duplicates.

Split assignment occurs before cropping or degradation and obeys the strictest group:

- no airfield group appears in more than one split;
- no geographic group crosses a holdout boundary;
- no capture/near-duplicate group crosses splits;
- all crops, tiles, augmented samples, and degraded derivatives inherit the source scene split;
- overlapping mosaics and repeated vendor products are duplicate-audited;
- temporal scenes of a test airfield remain test-only.

Primary evaluation uses airfield-level holdouts. At least one test slice is geographically isolated from training regions. Additional cross-sensor and cross-resolution holdouts are recorded as immutable named slices.

## Preventing identity and background leakage

Airfield identity, provider catalog IDs, coordinates, filenames, and split-only metadata are excluded from model features. The classifier must not infer type from “this base usually hosts fighters.” Training crops are checked for signage, map overlays, watermarks, fixed infrastructure, distinctive background, and acquisition/vendor artifacts correlated with labels. Evaluation reports stratify by unseen airfield, geography, sensor, resolution, season, and background.

The split audit checks geographic distance, footprint overlap, scene/crop lineage, checksums and perceptual hashes, acquisition proximity, airfield IDs/aliases, provider reprocessing IDs, and label distribution. A failed leakage audit blocks evaluation claims.

## Synthetic degradation derivatives

Every derivative links to native source scene/image/annotation IDs and an immutable `degradation_profile_id` containing random seed and ordered parameters for:

- sensor PSF/MTF blur;
- pixel-area integration/aggregation before sampling;
- resampling kernel and grid phase;
- atmospheric haze and contrast;
- signal-dependent and read noise;
- compression/quantization;
- band response/mixing and missing bands;
- shadows/illumination changes;
- georegistration offsets/warps.

Simple image resizing is never accepted as adequate sensor simulation. Profiles target named sensor/resolution conditions and are calibrated against real imagery. Degraded children cannot enter a different split from their native parents.

## Real low-resolution validation

Real scenes from recurring operational sensors form separate, sensor-specific calibration and validation slices. Labels match physically observable tasks: individuals only when clearly resolved, otherwise occupancy regions, dense storage, large-aircraft presence, activity/change, count intervals, and unknown/indeterminate. High-resolution contemporaneous reference may support adjudication but its detail must not leak into labels claimed visible in the low-resolution input.

Synthetic and real low-resolution results are reported separately. Synthetic success cannot substitute for real-sensor validation.

## Negatives and difficult examples

Negatives include empty airfields/aprons and confusers such as vehicles, shelters, buildings, markings, containers, ships, clouds, shadows, snow, glare, agricultural/industrial patterns, and compression artifacts. Hard examples include crowded storage, touching aircraft, mixed types, tiny edge objects, truncation, occlusion, strong aircraft shadows, camouflage, low contrast, unusual orientations, haze/cloud, seasonal surfaces, and imperfect registration.

Negative and difficult scenes carry the same provenance, grouping, split, quality, and licence requirements. Hard-negative mining never moves test examples into training.

## Quality flags and review

Scene/image flags separately encode cloud, cloud shadow, aircraft shadow, haze, obstruction, saturation, low contrast, blur, georegistration concern, and usable fraction. Object flags encode truncation, occlusion, shadow severity, ambiguity, and weak-label confidence. Unknown is distinct from none.

Recommended review states are unreviewed, single review, double review, adjudicated, and rejected. Test labels require double review/adjudication under a frozen guideline. Reviewer disagreement and revisions remain auditable.

## Dataset release gates

A dataset version is usable only with schema validation, licence audit, checksum/provenance coverage, taxonomy/version lock, duplicate and geographic leakage audit, split immutability, class/quality/sensor summaries, and documented known limitations. Coordinates may be redacted without invalidating image-space training records.

