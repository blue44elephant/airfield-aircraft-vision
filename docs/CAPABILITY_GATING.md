# Capability Gating Policy

## Principle

Displayed sharpness is not physical information content. More output pixels cannot recover spatial frequencies the sensor never measured. Routing therefore follows trustworthy source physics and quality, never UI zoom level, interpolation, or visual plausibility.

The initial boundaries are policy defaults, not performance claims. A task is enabled only when both this policy and resolution-/sensor-specific empirical validation permit it.

## Resolution terms

| Term | Meaning | Effect on routing |
|---|---|---|
| Sensor-provided native GSD | Nominal ground distance between source pixel samples, by band/product | Baseline candidate; must be trustworthy |
| Effective spatial resolution | Actual separability after optics, MTF/PSF, atmosphere, motion, and processing | If poorer than GSD, the poorer value governs |
| Panchromatic GSD | Physical sampling of a genuine broad-band panchromatic detector | May support geometry at that scale, with validation |
| Pan-sharpened resolution | Pixel spacing of fused panchromatic and multispectral data | Can inherit genuine pan geometry only when source pan metadata is known; does not create pan-scale spectral detail |
| Artificially upscaled resolution | Smaller output pixels made by interpolation | Recorded as a transform; never raises a tier |
| Super-resolved output | Reconstructed or generated fine grid | Never raises a tier by itself; cannot be treated as new measured evidence |
| Unknown/unreliable metadata | Missing, invalid, conflicting, or untrusted physical resolution | Automatic optical routing unavailable; require metadata review |

For ordinary optical data, the governing value is no better than `max(native GSD, known effective resolution)`. For a legitimate pan-sharpened product it is no better than `max(source panchromatic GSD, pan-product pixel spacing, known effective resolution)`. A pan product without source panchromatic metadata does not receive a tier upgrade.

## Initial optical tiers

### Tier A — physical resolution <= 1.0 m/pixel

Allowed, subject to model validation and scene quality:

- individual aircraft detection and oriented bounding boxes;
- approximate major-axis orientation/heading with ambiguity;
- broad morphology labels: `combat_jet`, `bomber_or_large_combat_aircraft`, `transport_or_tanker`, `special_mission_aircraft`, `trainer_or_light_aircraft`, `helicopter`, `uav_like`, `unknown_aircraft`;
- per-class counts, confidence, and uncertainty;
- family candidates only as explicitly optional, calibrated candidates when separately validated.

Forbidden: promising exact aircraft-model identification or forcing uncertain objects into known classes.

### Tier B — physical resolution > 1.0 and <= 5.0 m/pixel

Allowed:

- aircraft-like object detection;
- approximate individual counts only where clearly separated;
- broad size and large/small categories;
- fixed-wing/rotary-wing only after Tier-B validation;
- occupied parking regions, uncertainty-aware ranges, and `unknown`.

Forbidden unless resolution-specific validation changes the policy: fighter, bomber, trainer, exact type, or family labels. A source that happens to cover a known fighter base does not make a blob a fighter.

### Tier C — physical resolution > 5.0 m/pixel

Allowed:

- apron activity and parking-stand occupancy;
- dense storage, large-aircraft presence, and temporal change;
- supportable count ranges, confidence, and image-quality reporting.

Forbidden: one box per unresolved aircraft and individual fighter/bomber/transport/helicopter or exact-model classification. Sentinel-2 at 10 m remains Tier C even if resampled to 1 m.

## Quality and metadata gates

- `acceptable`: retain the physical-resolution tier;
- `poor`: conservatively reduce A to B or B to C; C stays C with aggregate-only warnings;
- `unusable`: stop and select another scene;
- `unknown`: issue a provisional warning and require quality assessment before operational output;
- unreliable or absent resolution: return `unavailable`, never insert a guessed GSD.

Cloud, haze, shadows, saturation, obstruction, contrast, georegistration, motion blur, band availability, and usable AOI fraction inform quality. Model manifests can impose stricter constraints than this base policy.

## SAR gate

SAR is not an optical-resolution tier. Pixel spacing is not equivalent to optical GSD, and speckle/scattering signatures require separate preprocessing, training, and validation. Every SAR scene routes to `sar_future`; all optical models and tasks are forbidden until a separate SAR pipeline is approved.

## Non-elevation rule

No downstream operation—bicubic interpolation, pansharpening without validated physical pan provenance, denoising, deconvolution, generative enhancement, or super-resolution—may elevate a scene beyond its defensible measured information. Such outputs may assist visualization or a separately validated model, but their use, version, and uncertainty must be disclosed.

