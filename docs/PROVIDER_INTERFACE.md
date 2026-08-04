# Imagery Provider Interface

## Contract

Provider adapters implement `ImageryProvider` from `src/airfield_aircraft_vision/providers/base.py`. The interface is asynchronous and metadata-first:

1. `search(request)` yields normalized candidate metadata without pixel downloads.
2. `get_metadata(scene_id)` refreshes one candidate before selection/download.
3. `download(scene_id, destination)` is called only after licence, eligibility, size, and authentication checks.

A search request carries GeoJSON AOI, UTC start/end, desired sensor domains, and optional cloud limit. A scene record carries provider/scene/sensor IDs, optical-versus-SAR domain, acquisition time, footprint, native/effective resolution, bands, cloud cover, licence, preview, download eligibility, auth requirement, archive/recurring coverage, archive/tasking/user-supplied mode, and estimated size. Raw provider fields remain namespaced for auditability.

## Adapter rules

- Adapters normalize units and timestamps but retain source values and provenance.
- Missing metadata stays `None`; adapters do not guess GSD, licence, or cloud cover.
- Credentials are injected in adapter configuration and never appear in normalized records, logs, or repository files.
- Search must not imply download rights. `download_eligible`, licence, authentication, and estimated size are evaluated explicitly.
- Preview URLs are optional and may have different licence/quality than analytical assets.
- Tasking offers and existing archive scenes remain distinguishable.
- User-supplied imagery is an adapter with the same validation requirements, not a bypass.
- SAR adapters emit `sensor_domain = "sar"` and cannot select an optical inference route.

## Planned adapters

Sentinel-2, Bhoonidhi/Resourcesat where permitted, CBERS, PRISMA, ESA-approved optical sources, open SAR archives, and user-supplied imagery can be added independently. Provider names in this plan do not assert API access, licence, recurring availability, or download eligibility; those are DATA-1 research outputs.

## Selection boundary

Adapters discover and normalize. Core scene selection applies AOI coverage, licence, cloud/quality, recency, physical resolution, download size, and capability policy. Vision modules receive a provider-independent local asset plus normalized immutable provenance.

