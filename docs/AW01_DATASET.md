# AW-01 RarePlanes suitability and conditions

## Verified primary sources

- [IQT dataset and attribution](https://www.iqt.org/library/the-rareplanes-dataset)
- [IQT user guide](https://www.iqt.org/library/rareplanes-public-user-guide)
- [AWS registry contract](https://raw.githubusercontent.com/awslabs/open-data-registry/main/datasets/rareplanes.yaml)
- [Bucket LICENSE.txt](https://rareplanes-public.s3.us-west-2.amazonaws.com/LICENSE.txt)
- [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- [Original paper](https://arxiv.org/abs/2006.02963)

The real subset contains WorldView-3 overhead imagery and aircraft annotations. It is suitable for a first generic
high-resolution optical aircraft detector, including different aircraft sizes and orientations. It is not SAR validation.
The pan-sharpened RGB product is useful for aircraft geometry; RGB spacing is not multispectral resolving power.

The AWS bucket is anonymous/readable without account credentials. Initial access was verified with bounded listings;
small metadata, selected individual PNG tiles and matching XML sidecars were retrieved. No bulk tarballs or original
experimental model checkpoints are needed.

## Conditions

The verified bucket license identifies CC BY-SA 4.0, with attribution to J. Shermeyer, T. Hossler, A. Van Etten,
D. Hogan, R. Lewis and D. Kim, In-Q-Tel / CosmiQ Works and AI.Reverie, RarePlanes Dataset, June 2020.
Retain the supplied notice and license. When sharing adapted dataset material, provide credit, source/license links,
an indication of changes and the required ShareAlike terms. Free access does not remove these conditions.

Source imagery/labels and generated overlays remain outside the repository's proprietary code license.
Evaluation directories contain an attribution/derivative-license notice. No original imagery is overwritten.
Checkpoint redistribution and commercial deployment require review of the dataset, COCO initialization and artifact
terms; the code license alone does not establish rights in datasets or trained weights.

TorchVision code is [BSD licensed](https://github.com/pytorch/vision/blob/v0.25.0/LICENSE).
The initialization is the official COCO Faster R-CNN MobileNetV3-Large FPN artifact, checked against its published
hash-bearing filename. Its COCO benchmark numbers are not AW-01 performance results.
[Model documentation](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html).

## Preparation contract

- Native 512 x 512 RGB tiles, corresponding georeferencing sidecars, source CSV and aircraft-only COCO annotations.
- All original downloaded bytes retained. SHA-256 image, sidecar, CSV and annotation references recorded.
- CSV collection time is published to minute precision. Original string retained; no seconds are fabricated.
  Its timezone is treated as UTC for the experiment, an explicit assumption rather than newly verified sub-minute sensing time.
- Original affine transform/CRS retained, including EPSG:4326 grids. X/Y metre spacing is calculated locally by transforming
  three points; raster pixels are not reprojected. Source panchromatic resolution, output spacing and unknown effective
  resolution remain separate.
- Unknown local cloud/quality is retained. Experimental inference requires explicit permission for unknown quality.
- Stable grouping unions location IDs sharing a capture identifier or country/airfield name.
  Train/validation groups are selected from official training locations; test groups must be absent from that full set.
  All three split groups, capture IDs and exact image hashes must be disjoint.
- Split selection is seeded and independent of annotation counts or model results. An existing frozen manifest cannot
  be silently replaced with a different subset. Preparation does not use source identity/location as model features.

## Limits

The published default split also contains observation holdouts at locations seen during training; it is insufficient
on its own for this milestone's unseen-airfield claim. AW-01 uses additional group checks.
The small sample is a plumbing/baseline experiment, not a global benchmark. Native tiles are positive-only, so
empty-airfield false-alarm rate cannot be measured. A geographic-distance/footprint overlap audit beyond known
location/capture/name groups remains future work.

Aircraft-only instances annotations lack per-object visible-fraction metadata. Edge/truncated annotations are retained;
no unsupported visibility fraction is guessed. Axis-aligned boxes enclose the provided annotations.
Synthetic imagery, attribute classes, SAR and UAV model classification are not used.
