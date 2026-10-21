# AW-01 integration recommendations for Blue Elephant

No Blue Elephant code or runtime state was changed.

1. Keep aircraft inference in this independent package/process. Consume the versioned experimental result by file/job
   boundary; do not import models into the ship-detector pipeline.
2. Require optical Tier A imagery and source physical-resolution evidence. Existing Sentinel-2 RGB or other coarse
   basemaps must not be interpreted as individually resolved aircraft simply because the display is enlarged.
3. Retain `RasterSource`, exact source/checkpoint hashes, acquisition precision, native affine and pixel boxes.
   Render source-georeferenced polygons; do not derive geographic coordinates from a resized screenshot.
4. Label results **experimental aircraft detections** and show confidence threshold, model provenance, quality,
   uncertainty and rejected candidates. An empty result is not proof of an empty airfield.
5. Treat generic `unknown_aircraft` as an aircraft-detection taxonomy bucket. Its conditional class value of 1.0 is
   not certainty; the separate uncalibrated detector score is the relevant confidence signal.
6. Do not interpret the axis-aligned box angle as heading. Heading, aircraft type, UAV model classification and SAR
   aircraft detection are unavailable.
7. Introduce an aircraft layer only after larger unseen-airfield and empty-apron evaluation, confidence calibration,
   scene-edge/crowded-aircraft review, cancellation/resource budgeting and license review.
8. Keep optical aircraft counts distinct from SAR vessel counts/occupancy. Preserve all existing radar calibration,
   detection, tracking and storage formats.
