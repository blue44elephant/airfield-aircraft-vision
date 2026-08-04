# Repository Assessment

Assessment date: 2026-08-02.

## Source workspace

The starting working directory was `sentinel-ship-motion`, a mature, partially implemented Python repository—not an empty aircraft project. It contains a `src/sentinel_ship_motion` package, tests, configuration, documentation, research manifests/reports, data placeholders, scripts, experiments, a virtual environment/cache, and ship-candidate review UI assets. Its Git worktree already had numerous modified and untracked user files when inspected.

Reusable ideas include src-layout packaging, Pydantic-style result boundaries, sensor profiles, quality/provenance thinking, dataset research contracts, and an adapter-based EA boundary. No code was copied because the task requires independence and the ship repository has a different sensor/task domain.

## Independence decision

The new project was created as sibling `airfield-aircraft-vision`. No Sentinel project, dataset, model, or EA file was modified. This project starts at VISION-0 with only lightweight contracts, policy, tests, and planning artifacts.

## Missing operational components

- provider-specific adapters, access research, credentials, and licence confirmation;
- preview/AOI quality assessment and scene-ranking implementation;
- raster/geospatial I/O and GSD verification;
- preprocessing profiles and degradation simulator;
- acquired/licensed dataset and annotation tooling;
- optical Tier A/B/C models and calibrated thresholds;
- evaluation datasets/results and performance numbers;
- job/API server, persistent storage, observability, and deployment;
- EA implementation and SAR detector.

## Dependency risks

The runtime dependency is currently only Pydantic. Future rasterio/GDAL/PROJ installation and binary compatibility are the main geospatial packaging risks; ML frameworks and GPU/CUDA stacks would add major weight and should wait for model selection. Provider SDK churn, authentication, licensing, rate limits, catalogue inconsistency, and incomplete effective-resolution metadata are external risks. Mermaid rendering is documentation-only.

