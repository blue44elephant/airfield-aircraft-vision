"""Explicit local workflows; only prepare/fetch-weights access the network."""

import argparse, json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(
        description="AW-01 experimental high-resolution optical aircraft detection and counting"
    )
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser(
        "prepare", help="bounded RarePlanes sample; never bulk imagery"
    )
    prep.add_argument("--root", type=Path, default=Path("data/rareplanes"))
    for name, default in (
        ("seed", 17),
        ("train-groups", 3),
        ("validation-groups", 2),
        ("test-groups", 3),
        ("tiles-per-group", 3),
    ):
        prep.add_argument("--" + name, type=int, default=default)
    weights = sub.add_parser(
        "fetch-weights", help="explicit ~75 MiB official COCO initial weights"
    )
    weights.add_argument(
        "--output",
        type=Path,
        default=Path("data/weights/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth"),
    )
    tr = sub.add_parser(
        "train", help="freeze backbone, fine-tune single-aircraft head/RPN"
    )
    tr.add_argument("--manifest", type=Path, required=True)
    tr.add_argument("--init-weights", type=Path, required=True)
    tr.add_argument("--output", type=Path, required=True)
    tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--seed", type=int, default=17)
    tr.add_argument("--lr", type=float, default=0.005)
    ev = sub.add_parser(
        "evaluate",
        help="holdout metrics, machine-readable results and native-size overlays",
    )
    ev.add_argument("--manifest", type=Path, required=True)
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--outdir", type=Path, required=True)
    ev.add_argument("--split", choices=["validation", "test"], default="test")
    ev.add_argument("--confidence", type=float, default=0.5)
    inf = sub.add_parser(
        "infer", help="infer on a local image and explicit source metadata JSON"
    )
    inf.add_argument("--image", type=Path, required=True)
    inf.add_argument("--source", type=Path, required=True)
    inf.add_argument("--checkpoint", type=Path, required=True)
    inf.add_argument("--output", type=Path, required=True)
    inf.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="defaults to checkpoint validation threshold, otherwise 0.5",
    )
    inf.add_argument("--tile-size", type=int, default=512)
    inf.add_argument("--overlap", type=int, default=128)
    inf.add_argument("--allow-unknown-quality", action="store_true")
    for cmd in (tr, ev, inf):
        cmd.add_argument("--device", default="cpu")
        cmd.add_argument("--threads", type=int, default=4)
    args = vars(p.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "prepare":
            from .dataset import prepare

            root = args.pop("root")
            report = prepare(root, **args)
            print(
                json.dumps(
                    {
                        "manifest": str(root / "manifest.json"),
                        "samples": len(report.samples),
                        "preparation": report.preparation,
                    },
                    indent=2,
                )
            )
        elif command == "fetch-weights":
            from .dataset import BoundedDownload
            from .torch_backend import WEIGHTS_URL
            from .preprocessing import sha256_file

            path = BoundedDownload().fetch(
                WEIGHTS_URL, args["output"], max_file=90 * 1024 * 1024
            )
            checksum = sha256_file(path)
            if not checksum.startswith("fb6a3cc7"):
                raise ValueError("official model artifact checksum mismatch")
            print(
                json.dumps(
                    {"path": str(path), "sha256": checksum, "source": WEIGHTS_URL}
                )
            )
        elif command == "train":
            from .workflows import train

            args["manifest_path"] = args.pop("manifest")
            args["initial_weights"] = args.pop("init_weights")
            print(json.dumps(train(**args), indent=2))
        elif command == "evaluate":
            from .workflows import evaluate

            args["manifest_path"] = args.pop("manifest")
            print(json.dumps(evaluate(**args)["metrics"], indent=2))
        elif command == "infer":
            import torch
            from .optical_contracts import RasterSource
            from .torch_backend import TorchDetector
            from .inference.optical_high_resolution import infer_scene

            torch.set_num_threads(args["threads"])
            output = args["output"]
            if output.exists():
                raise ValueError("refusing to overwrite inference output")
            source = RasterSource.model_validate_json(
                args["source"].read_text(encoding="utf-8")
            )
            detector = TorchDetector(args["checkpoint"], args["device"])
            confidence = (
                args["confidence"]
                if args["confidence"] is not None
                else detector.manifest.get("selected_threshold", 0.5)
            )
            result = infer_scene(
                args["image"],
                source,
                detector,
                tile_size=args["tile_size"],
                overlap=args["overlap"],
                confidence=confidence,
                allow_unknown_quality=args["allow_unknown_quality"],
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "output": str(output),
                        "total_count": result.total_count,
                        "experimental": True,
                    }
                )
            )
        return 0
    except (ValueError, OSError, KeyError, ImportError, RuntimeError) as exc:
        p.exit(2, f"AW-01: {type(exc).__name__}: {exc}\n")
