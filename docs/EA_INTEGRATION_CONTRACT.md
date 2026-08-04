# Epic Archer Integration Contract

## Boundary

Airfield Aircraft Vision is an independently deployable service. EA is a client; it does not import model packages, access provider credentials, or reproduce capability rules. VISION-0 does not modify EA.

Proposed versioned HTTP boundary:

- `POST /v1/scene-searches` — AOI, UTC date range, optional provider/filter preferences; returns search ID;
- `GET /v1/scene-searches/{id}/candidates` — normalized candidate metadata and exclusion/ranking reasons;
- `POST /v1/analyses` — selected scene ID or `selection_policy: newest_usable`, AOI, optional compatible baseline scene;
- `GET /v1/analyses/{id}` — job state and final inference result;
- `GET /v1/analyses/{id}/detections.geojson` — GeoJSON FeatureCollection when individual geometry is supported.

Long operations use job resources, idempotency keys, cancellation, and explicit error objects. Auth and deployment are deferred; no provider credential crosses this API.

## User flow

1. EA sends a validated Polygon/MultiPolygon AOI and date range.
2. The service searches configured provider adapters by metadata.
3. EA displays candidates with provider, acquisition time, sensor domain, native/effective resolution, cloud/quality, licence summary, preview availability, eligibility, and size estimate.
4. EA asks the service to select the newest usable scene or submits the user's candidate.
5. EA shows the selected acquisition time, defensible physical resolution, quality, and provisional capability before analysis.
6. EA requests analysis; the service revalidates metadata and routes the correct sensor/tier pipeline.
7. The result contains a GeoJSON-compatible AOI and detection geometries where supported, plus image-space oriented boxes.
8. EA draws oriented polygons with stable class colors; `unknown_aircraft` and uncertainty use distinct styling. Tier C occupancy/activity regions are not presented as individual aircraft boxes.
9. EA displays total accepted, known, unknown, rejected, per-class totals where allowed, count range, confidence, and quality.
10. EA displays `unavailable_features`, their reasons, and warnings—for example, “individual fighter classification unavailable at 10 m; activity/occupancy only.”

## Result invariants

- Contract version is explicit.
- Acquisition timestamps are timezone-aware UTC at the transport boundary.
- Native and effective resolution remain distinct; display/upscaled spacing is never substituted.
- Known + unknown equals total accepted; accepted + rejected equals total candidates.
- Every unavailable feature has a reason.
- Unknown objects and rejected candidates are visible in totals/audit details.
- Sensor, provider, scene, quality, model/preprocessing versions (when implemented), and tier decision are traceable.

## GeoJSON

The detections endpoint returns WGS84 Features with `detection_id`, `status`, permitted class/probabilities, confidence, orientation, uncertainty flags, scene ID, acquisition time, sensor, native/effective GSD, and capability tier. The service performs projection; EA does not infer geolocation from display pixels. Coordinate-redacted dataset exports are separate from operational authorized results.

## Failure behavior

Structured errors distinguish invalid AOI, no provider coverage, no usable scene, licence/auth restriction, metadata uncertainty, unsupported sensor, no validated model, and processing failure. A missing capability is normally a successful analysis with explicit unavailable features—not a fabricated prediction.

