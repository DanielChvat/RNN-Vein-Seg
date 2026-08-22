"""Stage 6: run the trained segmenter over sequences -- predicted masks and timing."""

from __future__ import annotations

import os
import re
import statistics
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .datasets import SequenceDataset
from .models import RNN, load_checkpoint

SLICE_RE = re.compile(r"_slice_(\d+)")


def slice_index(frame_name: str, fallback: int) -> int:
    """Recover the acquisition slice index from a frame filename.

    The old inference script took the *minimum* slice number in a sequence and
    incremented it once per emitted frame. Because the empty-frame filter
    removes frames, that counter drifts out of step with the real acquisition
    index as soon as a single frame is dropped, so the saved ``_slice_N``
    labels did not identify the slice they came from. The index is parsed from
    the source filename instead.
    """
    match = SLICE_RE.search(frame_name)
    return int(match.group(1)) if match else fallback


def load_segmenter(
    checkpoint: os.PathLike | str = config.SEG_CHECKPOINT,
    cfg: config.SegConfig | None = None,
    device: torch.device | None = None,
) -> RNN:
    """Build the segmenter and load weights, with checkpointing disabled.

    Gradient checkpointing only buys activation memory during backward, so it
    is dead weight at inference time.
    """
    cfg = cfg or config.SegConfig()
    device = device or config.device()

    model = RNN(
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        num_classes=cfg.num_classes,
        dropout_p=cfg.dropout_p,
        use_checkpoint=False,
    )
    load_checkpoint(model, checkpoint, device=device)
    return model.to(device).eval()


@torch.no_grad()
def iter_sequence_logits(
    model: RNN, dataset: SequenceDataset, device: torch.device, skip_augmented: bool = True
) -> Iterator[tuple[str, int, str, torch.Tensor, torch.Tensor]]:
    """Yield ``(seq_name, t, frame_name, image, logits)`` frame by frame.

    Shared by mask export, visualisation and benchmarking so the recurrent
    state handling lives in exactly one place.
    """
    for idx in range(len(dataset)):
        sample = dataset[idx]
        seq_name = sample["seq_name"]
        if skip_augmented and "_AUG_" in seq_name:
            continue

        images = sample["images"].to(device)
        model.reset_state()

        for t in range(images.shape[0]):
            image = images[t].unsqueeze(0)
            yield seq_name, t, sample["frame_names"][t], image, model(image, t_idx=t)

        model.reset_state()


def predict_masks(
    data_dir: os.PathLike | str = config.FILTERED_DIR,
    out_dir: os.PathLike | str = config.NPZ_OUTPUT_DIR,
    checkpoint: os.PathLike | str = config.SEG_CHECKPOINT,
    upsample: int = 4,
    cfg: config.SegConfig | None = None,
) -> None:
    """Write ``<seq>_slice_<N>.npz`` predicted masks under key ``pred``.

    ``upsample`` interpolates the logits before the argmax, which smooths class
    boundaries at the cost of an ``upsample**2`` blow-up in memory and file
    size. Set it to 1 to keep predictions at training resolution.
    """
    device = config.device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = SequenceDataset(data_dir)
    model = load_segmenter(checkpoint, cfg, device)

    written = 0
    current = None
    for seq_name, t, frame_name, _img, logits in iter_sequence_logits(model, dataset, device):
        if seq_name != current:
            print(f"Sequence: {seq_name}")
            current = seq_name

        if upsample > 1:
            logits = F.interpolate(
                logits, scale_factor=upsample, mode="bilinear", align_corners=False
            )

        pred = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
        idx = slice_index(frame_name, fallback=t)
        np.savez_compressed(out_dir / f"{seq_name}_slice_{idx}.npz", pred=pred)
        written += 1

    print(f"\nWrote {written} predicted masks to {out_dir}")


@torch.no_grad()
def benchmark(
    data_dir: os.PathLike | str = config.FILTERED_DIR,
    checkpoint: os.PathLike | str = config.SEG_CHECKPOINT,
    warmup: int = 10,
    cfg: config.SegConfig | None = None,
) -> None:
    """Report per-frame inference latency.

    Timing brackets the forward call itself and synchronises on both sides;
    the old version started the clock before the call but only synchronised
    after it, so every measurement absorbed whatever work was still queued from
    the previous frame. It also called ``torch.cuda.synchronize()``
    unconditionally, which raises on a CPU-only machine, and had no warm-up, so
    the first frames charged CUDA context and cuDNN autotune to the average.
    """
    device = config.device()
    dataset = SequenceDataset(data_dir)
    model = load_segmenter(checkpoint, cfg, device)
    on_cuda = device.type == "cuda"

    def sync() -> None:
        if on_cuda:
            torch.cuda.synchronize()

    times: list[float] = []
    seen = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if "_AUG_" in sample["seq_name"]:
            continue

        images = sample["images"].to(device)
        model.reset_state()

        for t in range(images.shape[0]):
            image = images[t].unsqueeze(0)
            sync()
            start = time.perf_counter()
            model(image, t_idx=t)
            sync()
            elapsed = time.perf_counter() - start

            seen += 1
            if seen > warmup:
                times.append(elapsed)

        model.reset_state()

    if not times:
        raise RuntimeError(
            f"No frames left to benchmark under {data_dir} after {warmup} warm-up frames"
        )

    times.sort()
    mean = statistics.fmean(times)
    print("\n======== INFERENCE SPEED ========")
    print(f"Device:        {device}")
    print(f"Frames:        {len(times)}")
    print(f"Mean:          {mean * 1000:.3f} ms  ({1 / mean:.2f} FPS)")
    print(f"Median:        {statistics.median(times) * 1000:.3f} ms")
    print(f"p95:           {times[int(0.95 * (len(times) - 1))] * 1000:.3f} ms")
    print("=================================")
