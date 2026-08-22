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


def _seg_config(args: argparse.Namespace) -> config.SegConfig:
    cfg = config.SegConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.base_channels is not None:
        cfg.base_channels = args.base_channels
    if args.val_groups:
        cfg.val_groups = tuple(args.val_groups)
    return cfg


def stage_preprocess(args) -> None:
    from . import preprocess

    preprocess.run(args.raw_dir, args.processed_dir)


def stage_train_detector(args) -> None:
    from . import empty_detector

    empty_detector.train(args.processed_dir, args.detector_checkpoint)


def stage_filter(args) -> None:
    from . import empty_detector

    empty_detector.filter_frames(args.processed_dir, args.filtered_dir, args.detector_checkpoint)


def stage_augment(args) -> None:
    from . import augment

    cfg = config.AugmentConfig(copy_originals=not args.no_copy_originals)
    if args.num_augments is not None:
        cfg.num_augments = args.num_augments
    if args.seed is not None:
        cfg.seed = args.seed
    augment.run(args.filtered_dir, args.augmented_dir, cfg)


def stage_train(args) -> None:
    from . import train

    train.run(args.augmented_dir, args.checkpoint_dir, _seg_config(args))


def stage_predict(args) -> None:
    from . import inference

    inference.predict_masks(
        args.filtered_dir, args.npz_dir, args.seg_checkpoint, args.upsample, _seg_config(args)
    )


def stage_visualize(args) -> None:
    from . import visualize

    visualize.run(args.filtered_dir, args.vis_dir, args.seg_checkpoint, _seg_config(args))


def stage_benchmark(args) -> None:
    from . import inference

    inference.benchmark(args.filtered_dir, args.seg_checkpoint, cfg=_seg_config(args))


def stage_fit(args) -> None:
    from . import vessel_fitting

    vessel_fitting.run(args.npz_dir, args.report, args.degree)


def stage_analyze(args) -> None:
    from . import analysis

    analysis.intensity_correlation(args.processed_dir)


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
            "  preprocess       raw_data/     -> processed_data/\n"
            "  train-detector   processed_data/ -> empty_detector.pth\n"
            "  filter           processed_data/ -> filtered_data/\n"
            "  augment          filtered_data/  -> filtered_data_augmented/\n"
            "  train            filtered_data_augmented/ -> checkpoints/\n"
            "  predict          filtered_data/  -> npz_outputs/\n"
            "  visualize        filtered_data/  -> vis_outputs/\n"
            "  benchmark        per-frame inference latency\n"
            "  fit              npz_outputs/    -> radii_report.txt\n"
            "  analyze          intensity/t-SNE study of empty masks\n"
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

    paths = parser.add_argument_group("paths")
    paths.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    paths.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR)
    paths.add_argument("--filtered-dir", type=Path, default=config.FILTERED_DIR)
    paths.add_argument("--augmented-dir", type=Path, default=config.AUGMENTED_DIR)
    paths.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIR)
    paths.add_argument("--seg-checkpoint", type=Path, default=config.SEG_CHECKPOINT)
    paths.add_argument("--detector-checkpoint", type=Path, default=config.DETECTOR_CHECKPOINT)
    paths.add_argument("--npz-dir", type=Path, default=config.NPZ_OUTPUT_DIR)
    paths.add_argument("--vis-dir", type=Path, default=config.VIS_OUTPUT_DIR)
    paths.add_argument("--report", type=Path, default=config.RADII_REPORT)

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
    opts.add_argument("--upsample", type=int, default=4, help="logit upsampling before argmax")
    opts.add_argument("--degree", type=int, default=4, help="polynomial degree for vessel fitting")
    opts.add_argument("--clean", action="store_true", help="delete processed_data/ when done")

    return parser


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

    print(f"Pipeline: {' -> '.join(stages)}")
    run_stages(stages, args)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
