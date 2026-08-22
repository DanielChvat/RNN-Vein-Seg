"""Pipeline entry point.

    python -m veinseg <stage> [options]

Stages compose left to right; ``prepare`` and ``all`` are shorthands for common
runs. Replaces ``prepare_data.sh`` / ``prepare_data.ps1``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from . import config

# Order matters: this is the dependency chain.
PREPARE_STAGES = ("preprocess", "train-detector", "filter", "augment")
ALL_STAGES = PREPARE_STAGES + ("train", "predict", "fit")


def _banner(name: str) -> None:
    print(f"\n{'=' * 64}\n  {name}\n{'=' * 64}", flush=True)


# ----------------------------------------------------------------------
# Stage implementations
# ----------------------------------------------------------------------


def stage_preprocess(args) -> None:
    from . import preprocess

    preprocess.run(args.raw_dir, args.processed_dir)


def stage_train_detector(args) -> None:
    from . import empty_detector

    empty_detector.train(args.processed_dir, args.detector_checkpoint, args.pipeline.detector)


def stage_filter(args) -> None:
    from . import empty_detector

    empty_detector.filter_frames(
        args.processed_dir, args.filtered_dir, args.detector_checkpoint, args.pipeline.detector
    )


def stage_augment(args) -> None:
    from . import augment

    augment.run(args.filtered_dir, args.augmented_dir, args.pipeline.augment)


def stage_train(args) -> None:
    from . import train

    train.run(args.augmented_dir, args.checkpoint_dir, args.pipeline.seg)


def stage_predict(args) -> None:
    from . import inference

    inference.predict_masks(
        args.filtered_dir, args.npz_dir, args.seg_checkpoint, args.upsample, args.pipeline.seg
    )


def stage_visualize(args) -> None:
    from . import visualize

    visualize.run(args.filtered_dir, args.vis_dir, args.seg_checkpoint, args.pipeline.seg)


def stage_benchmark(args) -> None:
    from . import inference

    inference.benchmark(args.filtered_dir, args.seg_checkpoint, cfg=args.pipeline.seg)


def stage_fit(args) -> None:
    from . import vessel_fitting

    vessel_fitting.run(args.npz_dir, args.report, args.degree)


def stage_analyze(args) -> None:
    from . import analysis

    analysis.intensity_correlation(
        args.processed_dir, show=args.show, figures_dir=args.figures_dir
    )


def stage_viewer(args) -> None:
    from . import analysis

    analysis.detector_viewer(args.processed_dir, args.detector_checkpoint)


STAGES = {
    "preprocess": stage_preprocess,
    "train-detector": stage_train_detector,
    "filter": stage_filter,
    "augment": stage_augment,
    "train": stage_train,
    "predict": stage_predict,
    "visualize": stage_visualize,
    "benchmark": stage_benchmark,
    "fit": stage_fit,
    "analyze": stage_analyze,
    "viewer": stage_viewer,
}


def run_stages(names: tuple[str, ...], args: argparse.Namespace) -> None:
    for name in names:
        _banner(name)
        start = time.perf_counter()
        STAGES[name](args)
        print(f"[{name}] finished in {time.perf_counter() - start:.1f}s")

    if args.clean:
        # The old shell script always deleted processed_data/ on the way out,
        # which quietly broke `analyze` and `viewer` -- both read it. Cleanup is
        # now opt-in.
        _banner("clean")
        for path in (Path(args.processed_dir), config.ROOT / "__pycache__"):
            if path.exists():
                shutil.rmtree(path)
                print(f"removed {path}")


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m veinseg",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "stages:\n"
            "  preprocess       data/raw/       -> data/processed/\n"
            "  train-detector   data/processed/ -> checkpoints/empty_detector.pth\n"
            "  filter           data/processed/ -> data/filtered/\n"
            "  augment          data/filtered/  -> filtered_data_augmented/\n"
            "  train            filtered_data_augmented/ -> checkpoints/\n"
            "  predict          data/filtered/  -> npz_outputs/\n"
            "  visualize        data/filtered/  -> vis_outputs/\n"
            "  benchmark        per-frame inference latency\n"
            "  fit              npz_outputs/    -> radii_report.txt\n"
            "  analyze          intensity/t-SNE study of empty masks -> figures/\n"
            "  viewer           interactive empty-detector browser\n"
            "\ngroups:\n"
            f"  prepare          {' -> '.join(PREPARE_STAGES)}\n"
            f"  all              {' -> '.join(ALL_STAGES)}\n"
        ),
    )

    parser.add_argument(
        "stages",
        nargs="+",
        metavar="STAGE",
        help="one or more stages, or the group names 'prepare' / 'all'",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help=f"YAML config (default: {config.DEFAULT_CONFIG_FILE.relative_to(config.ROOT)} "
        "if present; optional)",
    )

    # Path defaults are deliberately None so that "not passed" is
    # distinguishable from "passed the default value" -- otherwise a YAML
    # override could never win over an argparse default. Resolved in main().
    paths = parser.add_argument_group("paths", "override the YAML config")
    for flag in (
        "--raw-dir",
        "--processed-dir",
        "--filtered-dir",
        "--augmented-dir",
        "--checkpoint-dir",
        "--seg-checkpoint",
        "--detector-checkpoint",
        "--npz-dir",
        "--vis-dir",
        "--figures-dir",
        "--report",
    ):
        paths.add_argument(flag, type=Path, default=None)

    opts = parser.add_argument_group("options")
    opts.add_argument("--epochs", type=int, help="segmentation training epochs")
    opts.add_argument("--base-channels", type=int, help="must match between train and inference")
    opts.add_argument("--val-groups", nargs="+", help="sequence roots held out for validation")
    opts.add_argument("--num-augments", type=int, help="augmented copies per sequence")
    opts.add_argument(
        "--no-copy-originals",
        action="store_true",
        help="do not copy un-augmented sequences into the training root "
        "(reproduces the old behaviour: no clean holdout)",
    )
    opts.add_argument("--seed", type=int, help="augmentation seed")
    opts.add_argument("--upsample", type=int, help="logit upsampling before argmax")
    opts.add_argument("--degree", type=int, help="polynomial degree for vessel fitting")
    opts.add_argument(
        "--show",
        action="store_true",
        help="open interactive plot windows in 'analyze' (blocks; needs a display)",
    )
    opts.add_argument("--clean", action="store_true", help="delete the processed dir when done")

    return parser


#: argparse dest -> config constant, for the CLI > YAML > default cascade.
_PATH_ARGS = {
    "raw_dir": "RAW_DIR",
    "processed_dir": "PROCESSED_DIR",
    "filtered_dir": "FILTERED_DIR",
    "augmented_dir": "AUGMENTED_DIR",
    "checkpoint_dir": "CHECKPOINT_DIR",
    "seg_checkpoint": "SEG_CHECKPOINT",
    "detector_checkpoint": "DETECTOR_CHECKPOINT",
    "npz_dir": "NPZ_OUTPUT_DIR",
    "vis_dir": "VIS_OUTPUT_DIR",
    "figures_dir": "FIGURES_DIR",
    "report": "RADII_REPORT",
}


def resolve(args: argparse.Namespace, data: dict) -> argparse.Namespace:
    """Fill every unset option from the YAML config, then the built-in default.

    The whole CLI > YAML > default cascade happens here, once. It used to be
    split between this and a per-stage helper, so an override only took effect
    in the stages that remembered to call it.
    """
    config.apply_geometry(data)
    args.pipeline = config.pipeline_from_yaml(data)

    seg, augment = args.pipeline.seg, args.pipeline.augment
    if args.epochs is not None:
        seg.epochs = args.epochs
    if args.base_channels is not None:
        seg.base_channels = args.base_channels
    if args.val_groups:
        seg.val_groups = tuple(args.val_groups)
    if args.num_augments is not None:
        augment.num_augments = args.num_augments
    if args.seed is not None:
        augment.seed = args.seed
    if args.no_copy_originals:
        augment.copy_originals = False

    # Which paths the user actually typed. Captured before the fill below, while
    # "unset" is still distinguishable.
    explicit = {dest for dest in _PATH_ARGS if getattr(args, dest) is not None}

    overrides = config.resolve_paths(data)
    for dest, attr in _PATH_ARGS.items():
        if getattr(args, dest) is None:
            setattr(args, dest, overrides.get(attr, getattr(config, attr)))

    # Both checkpoints live inside the checkpoint dir, so they track it. The
    # explicit-CLI check is what keeps `--checkpoint-dir X` working: the shipped
    # config.yaml pins inference.checkpoint, and without this that YAML value
    # would beat the flag and send predict looking in the default directory
    # while train wrote to X.
    for dest, attr in (("seg_checkpoint", "SEG_CHECKPOINT"),
                       ("detector_checkpoint", "DETECTOR_CHECKPOINT")):
        if dest in explicit:
            continue  # named outright -- nothing outranks that
        if "checkpoint_dir" in explicit or attr not in overrides:
            setattr(args, dest, args.checkpoint_dir / getattr(config, attr).name)

    if args.upsample is None:
        args.upsample = config.get_key(data, "inference", "upsample", 4)
    if args.degree is None:
        args.degree = config.get_key(data, "fit", "degree", 4)

    return args


def expand(names: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    for name in names:
        if name == "prepare":
            out.extend(PREPARE_STAGES)
        elif name == "all":
            out.extend(ALL_STAGES)
        elif name in STAGES:
            out.append(name)
        else:
            raise SystemExit(
                f"unknown stage {name!r}. Choose from: "
                f"{', '.join(sorted(STAGES))}, prepare, all"
            )
    return tuple(dict.fromkeys(out))  # de-duplicate, preserve order


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stages = expand(args.stages)

    # A named config must exist; the default one is allowed to be absent.
    named = args.config is not None
    args = resolve(args, config.load_yaml(args.config or config.DEFAULT_CONFIG_FILE, required=named))

    print(f"Pipeline: {' -> '.join(stages)}")
    run_stages(stages, args)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
