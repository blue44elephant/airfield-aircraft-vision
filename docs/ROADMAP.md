# Gated Roadmap

Every milestone is blocked until its inputs are versioned and its acceptance criteria pass. “Must not claim” statements remain in force after the milestone unless a later named evaluation gate explicitly supplies evidence.

## VISION-0 — architecture and schemas

- **Inputs:** scientific constraints, initial tier policy, integration intent.
- **Outputs:** architecture/capability/dataset/provider/EA contracts, annotation schema, output schema, router, tests, repository skeleton.
- **Acceptance:** all requested contracts exist; boundary/upscale/pan/SAR/quality/count tests pass; Sentinel and EA remain unchanged; no heavyweight ML dependencies.
- **Blockers:** unresolved contract defects or scientific-policy ambiguity.
- **Must not claim:** working imagery search, inference, training, SAR support, or any accuracy.

## DATA-1 — provider and dataset inventory

- **Inputs:** provider interface, AOI use cases, licence/security requirements.
- **Outputs:** source catalogue, access/licence matrix, sample metadata audit, download/coverage/recency/cost estimates, dataset candidates.
- **Acceptance:** at least one feasible source per intended optical tier or a documented gap; native/effective GSD and licence provenance verified; no bulk download.
- **Blockers:** access, licence, tasking cost, coverage, metadata reliability.
- **Must not claim:** recurring availability or analytical suitability from marketing pixel size alone.

## DATA-2 — high-resolution annotation dataset

- **Inputs:** approved sources/licences, frozen taxonomy/guideline, split policy.
- **Outputs:** native imagery references, oriented annotations, negatives/hard cases, immutable airfield/geographic splits, audits.
- **Acceptance:** schema/provenance/licence/duplicate/leakage checks pass; test labels double-reviewed/adjudicated; unknowns retained.
- **Blockers:** insufficient class/airfield diversity, sensitive coordinates, annotation ambiguity.
- **Must not claim:** exact-model truth, low-resolution transfer, or generalization before held-out evaluation.

## DATA-3 — degradation pipeline

- **Inputs:** native DATA-2 samples, target sensor studies, real image statistics.
- **Outputs:** versioned PSF/aggregation/resampling/haze/noise/contrast/compression/band/shadow/georegistration simulator and lineage manifests.
- **Acceptance:** deterministic profiles, split inheritance, unit/property tests, and measured similarity diagnostics against real targets; resize-only baselines labeled inadequate.
- **Blockers:** unknown sensor MTF/band response or inadequate real calibration examples.
- **Must not claim:** synthetic equivalence to a real sensor or operational performance.

## DATA-4 — real low-resolution calibration set

- **Inputs:** licensed recurring-sensor scenes, task-appropriate labels, optional high-resolution adjudication references.
- **Outputs:** sensor-specific Tier B/C calibration/evaluation slices, occupancy/activity/count-range truth and quality strata.
- **Acceptance:** geographically held-out, leakage-audited real scenes with observable-only labels and documented uncertainty.
- **Blockers:** temporal reference mismatch, cloud, sparse positives, licence/geolocation limits.
- **Must not claim:** specific aircraft identity not visible in input pixels.

## MODEL-1 — high-resolution aircraft detector

- **Inputs:** DATA-2, oriented-box metric protocol, frozen holdouts.
- **Outputs:** Tier A oriented detector, manifest, calibrated detection/unknown thresholds.
- **Acceptance:** predefined cross-airfield metrics and error slices reported on untouched holdouts; provenance and reproducibility pass.
- **Blockers:** label quality, domain imbalance, compute/model licensing.
- **Must not claim:** morphology/exact type, medium/low-resolution, SAR, or production readiness.

## MODEL-2 — broad morphology classifier

- **Inputs:** validated MODEL-1 crops/features, DATA-2 broad labels, unknown/open-set samples.
- **Outputs:** Tier A broad-class probabilities, unknown handling, calibration.
- **Acceptance:** cross-airfield per-class/confusion/calibration results meet predefined gates; family candidates separately evaluated if attempted.
- **Blockers:** class ambiguity/imbalance, airfield-identity shortcuts, insufficient pixel support.
- **Must not claim:** exact aircraft model identification or classes without robust held-out evidence.

## MODEL-3 — medium-resolution detector

- **Inputs:** DATA-3 target profiles, real Tier B DATA-4 scenes, Tier B taxonomy/tasks.
- **Outputs:** separate aircraft-like/size/parking-region model or heads and count ranges.
- **Acceptance:** real sensor- and GSD-specific holdout gates pass; separated-object and crowded-scene limits documented.
- **Blockers:** synthetic-to-real gap, unresolved objects, inadequate real calibration.
- **Must not claim:** fighter/bomber/trainer/family distinctions without separate empirical proof.

## MODEL-4 — low-resolution occupancy and activity model

- **Inputs:** real Tier C DATA-4 scenes, registered temporal pairs, region/occupancy truth.
- **Outputs:** separate apron occupancy/activity/dense-storage/large-presence pipeline with uncertainty.
- **Acceptance:** cross-airfield real-sensor evaluation passes predefined occupancy/change gates; outputs remain aggregate.
- **Blockers:** registration, cloud/season change, weak temporal truth.
- **Must not claim:** one box/count per unresolved fighter or individual aircraft class.

## EVAL-1 — cross-airfield holdout evaluation

- **Inputs:** frozen models, untouched airfield/geographic holdouts, model/dataset manifests.
- **Outputs:** detection/classification/count/occupancy metrics, calibration, confidence intervals, failure taxonomy.
- **Acceptance:** zero leakage findings; predefined gates evaluated without test-driven retraining; known/unknown/rejected accounting reported.
- **Blockers:** leakage, small slices, changed labels/models after freeze.
- **Must not claim:** cross-sensor/resolution or operational generalization.

## EVAL-2 — cross-sensor and cross-resolution evaluation

- **Inputs:** EVAL-1 models, DATA-3/4 slices across named sensors/GSD/quality bands.
- **Outputs:** capability matrix, sensor/resolution calibration, downgrade/abstention thresholds.
- **Acceptance:** separate real versus synthetic results, adequate slice sizes, uncertainty and failure limits published; tier policy updated only through versioned evidence.
- **Blockers:** missing sensors, metadata uncertainty, domain shift.
- **Must not claim:** support outside evaluated sensors, bands, GSD, quality, or task definitions.

## EA-1 — offline integration

- **Inputs:** stable v1 API/result contract, local approved scenes, validated routing/model packages.
- **Outputs:** independently running local service and EA client workflow for search-like local catalogues, analysis, GeoJSON display, warnings.
- **Acceptance:** end-to-end contract tests; oriented geometry/totals/unavailable reasons render correctly; no model code or credentials embedded in EA.
- **Blockers:** API/auth/job design, CRS errors, model packaging.
- **Must not claim:** provider-backed or production operations.

## EA-2 — provider-backed AOI workflow

- **Inputs:** approved provider adapters, EA-1, security/licence/deployment review.
- **Outputs:** live metadata search, candidate display, newest-usable selection, eligible download and analysis jobs.
- **Acceptance:** provider sandbox/end-to-end tests, rate/error handling, cost/size confirmation, provenance, monitoring, and user-facing capability explanations.
- **Blockers:** credentials, terms, rate limits, tasking/cost, service reliability.
- **Must not claim:** universal coverage, guaranteed newest imagery, or unsupported analytical detail.

## SAR-1 — separate SAR feasibility study

- **Inputs:** approved SAR sources, SAR expertise, independent labels and evaluation plan.
- **Outputs:** scattering/resolution study, SAR-specific preprocessing/detector proposal, go/no-go evidence.
- **Acceptance:** optical pipeline remains inaccessible to SAR; feasibility is measured on SAR-specific held-out data.
- **Blockers:** speckle/domain complexity, labels, incidence/polarization diversity, licensing.
- **Must not claim:** SAR aircraft detection from optical results or pixel-spacing equivalence to optical GSD.

## Recommended next gate

After VISION-0 approval, proceed only to DATA-1. It is the lowest-risk way to validate source access, licence, metadata quality, physical resolution, recurrence, and cost before acquiring data or selecting models.

