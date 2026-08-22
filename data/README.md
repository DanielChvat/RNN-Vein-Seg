# data/

Everything here except this file is gitignored — it is regenerated from
`raw/` by `python -m veinseg prepare`.

```
raw/<SEQ>/imgs/     hand-provided source frames
raw/<SEQ>/masks/    hand-provided labels, matched to imgs by filename stem
processed/<SEQ>/    CASE_<SEQ>_slice_NNNN.npz   keys: image float32, label uint8
filtered/<SEQ>/     the frames the empty-detector kept
```

Frames are resized to 256x128 (1024x512 downscaled by 4), windowed to
[-125, 275] and min-max normalised per slice.

Two data roots are **not** here, because they are checked into git and moving
them would rewrite thousands of tracked paths:

- `../filtered_data_augmented/` — what `train` reads (2760 tracked files)
- `../npz_outputs/` — what `predict` writes and `fit` reads (971 tracked files)

Sequence names are discovered from disk, never hardcoded. See
`veinseg/config.py:discover_sequences`.
