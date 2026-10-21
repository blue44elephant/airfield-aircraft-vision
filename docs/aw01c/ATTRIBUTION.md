# AW-01C data and model attribution

RarePlanes Dataset, June 2020. J. Shermeyer, T. Hossler, A. Van Etten, D. Hogan, R. Lewis and D. Kim. In-Q-Tel / CosmiQ Works and AI.Reverie.

- [Owner's dataset and usage information](https://www.iqt.org/library/the-rareplanes-dataset)
- [Official preparation and annotation guide](https://www.iqt.org/library/rareplanes-public-user-guide)
- [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

The experiment uses the public real WorldView-3 portion. No synthetic imagery is used.
Native crops and detection overlays are marked derivatives; those imagery derivatives retain CC BY-SA 4.0. Original image pixels, sidecars, timestamps and annotations remain unchanged. Provenance JSON identifies the source, parent checksum, native window and changes.

The new background reviews are Codex visual inspections of exact source windows, not independent human surveys. Negative crops preserve source sampling and use lossless GeoTIFF storage. They are not new satellite acquisitions.

TorchVision 0.25 supplies the detector architecture and public COCO pretrained initializer:
[official ResNet-50 FPN documentation](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html),
[TorchVision BSD license](https://github.com/pytorch/vision/blob/v0.25.0/LICENSE).
Pretrained-data conditions are separate from the software license; no unrestricted redistribution permission for trained weights is asserted here. Downloaded/trained weights remain local in ignored directories.

These notices preserve the existing project's software license and original authorship. No endorsement by the dataset or model authors is claimed.
