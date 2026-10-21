"""Reproduce AW-01B stages without implicit test-set tuning."""

import argparse, json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(
        description="AW-01B bounded aircraft diagnosis and improvement"
    )
    sub = p.add_subparsers(dest="command", required=True)
    diag = sub.add_parser("diagnose")
    diag.add_argument(
        "--manifest", type=Path, default=Path("data/rareplanes/manifest.json")
    )
    diag.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/aw01-3epoch.pt")
    )
    diag.add_argument("--outdir", type=Path, required=True)
    diag.add_argument("--steps", type=int, default=80)
    prep = sub.add_parser("prepare")
    prep.add_argument(
        "--original-manifest", type=Path, default=Path("data/rareplanes/manifest.json")
    )
    prep.add_argument("--root", type=Path, default=Path("data/rareplanes-aw01b"))
    val = sub.add_parser("broaden-validation")
    val.add_argument(
        "--manifest", type=Path, default=Path("data/rareplanes-aw01b/manifest.json")
    )
    tr = sub.add_parser("train")
    tr.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/rareplanes-aw01b/manifest-balanced.json"),
    )
    tr.add_argument(
        "--init-weights",
        type=Path,
        default=Path("data/weights/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth"),
    )
    tr.add_argument("--outdir", type=Path, required=True)
    tr.add_argument("--profile", choices=["frozen_sgd", "fpn_adamw"], required=True)
    tr.add_argument("--epochs", type=int, default=6)
    cmp = sub.add_parser("compare")
    cmp.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/rareplanes-aw01b/manifest-balanced.json"),
    )
    cmp.add_argument(
        "--original-manifest", type=Path, default=Path("data/rareplanes/manifest.json")
    )
    cmp.add_argument(
        "--original-checkpoint", type=Path, default=Path("artifacts/aw01-3epoch.pt")
    )
    cmp.add_argument("--candidate-dirs", type=Path, nargs="+", required=True)
    cmp.add_argument("--outdir", type=Path, required=True)
    for cmd in (diag, tr, cmp):
        cmd.add_argument("--threads", type=int, default=4)
    args = vars(p.parse_args(argv))
    cmd = args.pop("command")
    try:
        if cmd == "diagnose":
            from .diagnostics import diagnose

            args["manifest_path"] = args.pop("manifest")
            diagnose(**args)
        elif cmd == "prepare":
            from .expanded_dataset import prepare_expanded

            original = args.pop("original_manifest")
            root = args.pop("root")
            m = prepare_expanded(original, root)
            print(
                json.dumps(
                    {"manifest": str(root / "manifest.json"), "samples": len(m.samples)}
                )
            )
        elif cmd == "broaden-validation":
            from .expanded_dataset import broaden_validation

            m = broaden_validation(args["manifest"])
            print(
                json.dumps(
                    {
                        "manifest": str(
                            args["manifest"].with_name("manifest-balanced.json")
                        ),
                        "samples": len(m.samples),
                    }
                )
            )
        elif cmd == "train":
            from .improved_training import train_candidate

            profile = args.pop("profile")
            args["manifest_path"] = args.pop("manifest")
            args["initial_weights"] = args.pop("init_weights")
            args.update(
                name="expanded_" + profile,
                lr=0.005 if profile == "frozen_sgd" else 0.0003,
                optimizer_name="sgd" if profile == "frozen_sgd" else "adamw",
                backbone="frozen" if profile == "frozen_sgd" else "fpn",
            )
            train_candidate(**args)
        elif cmd == "compare":
            from .model_comparison import compare

            args["manifest_path"] = args.pop("manifest")
            compare(**args)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as exc:
        p.exit(2, f"AW-01B: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
