# EfficientAD on MVTec AD — Kaggle GPU T4 (single-GPU)

A self-contained Jupyter notebook (`efficientad_mvtec_demo.ipynb`) that trains
[**EfficientAD**](https://arxiv.org/abs/2303.14535) — a fast teacher–student
anomaly detector — on the [**MVTec AD**](https://www.mvtec.com/company/research/datasets/mvtec-ad)
dataset using the [**Anomalib**](https://github.com/openvinotoolkit/anomalib)
library on a **single Kaggle T4 GPU**.

> **Branch layout**
> - `master` — TPU v5e-8 variant (8 cores, `accelerator="tpu"`)
> - `gpu` — **this branch**, single-GPU variant (1 × T4, `accelerator="gpu"`, `devices=1`)
>
> Switch with `git checkout master` ↔ `git checkout gpu`.

## Why only one GPU?

Kaggle's *GPU T4 ×2* runtime gives you two T4s, but this branch deliberately uses
just one. Reasons:

- EfficientAD's published protocol uses `BATCH_SIZE=1`. Running DDP across two
  GPUs would give an effective batch of 2 — small accuracy drift vs. paper numbers.
- Single-GPU avoids the `ddp_notebook` strategy and its associated multi-process
  quirks (datamodule reload, MemoryBankMixin sync, etc.).
- The setup is dramatically simpler. The second T4 stays idle — that's fine.

If you actually want both GPUs, see the multi-GPU notes at the bottom.

## Quick start

1. Open the notebook on Kaggle: *File → Import Notebook → Upload `efficientad_mvtec_demo.ipynb`*.
2. *Settings → Accelerator → **GPU T4 ×2*** (single T4 also works).
3. Run all cells. After the install cell finishes, **Runtime → Restart session**, then re-run from the install cell.
4. Default category is `bottle`; change `CATEGORY` in the configuration cell to swap.

Wall time: roughly 15–25 min for one category on a single T4.

## Notebook structure

| Cell | Purpose |
|------|---------|
| 1 | GPU sanity check; pins `CUDA_VISIBLE_DEVICES=0` so only the first T4 is used |
| 2 | Install `anomalib==2.4.1`; uninstall the legacy `pytorch-lightning` package which conflicts with anomalib's `lightning.pytorch.LightningModule` |
| 3 | Imports + seeding + `RESULTS_DIR = /kaggle/working/results` |
| 4 | Markdown reference for the 15 MVTec categories |
| 5 | Configuration: `CATEGORY`, `MODEL_SIZE`, `IMAGE_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS` |
| 6 | `MVTecAD` datamodule — auto-downloads MVTec (~5 GB) on first run |
| 7 | `EfficientAd(model_size=...)` |
| 8 | `Engine(accelerator="gpu", devices=1)` → `engine.fit(...)` |
| 9 | `engine.test(...)` — image/pixel AUROC, F1Max |
| 10 | Markdown |
| 11 | `engine.predict(...)` + 4-column matplotlib grid (image / GT mask / heatmap / binary) |
| 12 | Manual single-image inference using `cuda` |
| 13 | Markdown |
| 14 | Optional loop: train + evaluate every category, write `all_categories_results.csv` |
| 15 | Custom-data instructions (Folder datamodule structure) |

## Configuration knobs

```python
CATEGORY     = "bottle"   # one of the 15 MVTec AD categories
MODEL_SIZE   = "small"    # "small" or "medium"
IMAGE_SIZE   = 256
BATCH_SIZE   = 1          # paper protocol — keep at 1 for accuracy parity
NUM_EPOCHS   = 250        # anomalib default
```

## Outputs

Everything lands under `/kaggle/working/results/`:

- `predictions_<category>.png` — 4-column visualization grid (cell 11)
- `single_inference_<category>.png` — single-image demo (cell 12)
- `all_categories_results.csv` — per-category metrics (cell 14, when `RUN_ALL_CATEGORIES=True`)
- `lightning_logs/` — Lightning's per-run logs and the trained checkpoint

## Known caveats

- **MemoryBankMixin tail.** EfficientAD computes per-channel quantiles on CPU at the end of training. This appears as a long pause before the "✓ Training complete" line — it's working, not hung.
- **`pytorch-lightning` vs `lightning`.** Kaggle ships the legacy standalone `pytorch-lightning` package, whose `LightningModule` class is a different Python object than `lightning.pytorch.LightningModule`. Anomalib 2.x uses the latter. The install cell uninstalls the legacy package; **always restart the runtime after the install cell** so the kernel drops cached imports.
- **First-time MVTec download is ~5 GB** to `/kaggle/working/mvtec`. To avoid re-downloading on every session, attach the public Kaggle MVTec dataset and point `DATASET_ROOT` at `/kaggle/input/...` (it's read-only, so anomalib will create its index files inside `/kaggle/working/` and pull image data from the input mount).

## Want both GPUs?

Replace cell 1 (don't pin `CUDA_VISIBLE_DEVICES`) and cells 8 + 14:

```python
# cell 8 — multi-GPU
engine = Engine(
    accelerator="gpu",
    devices=2,
    strategy="ddp_notebook",   # required in notebooks; plain "ddp" hangs
    max_epochs=NUM_EPOCHS,
    default_root_dir=str(RESULTS_DIR),
    num_sanity_val_steps=0, limit_val_batches=0, check_val_every_n_epoch=None,
)
```

Be aware that `BATCH_SIZE=1 × 2 GPUs = effective batch 2`, which deviates
slightly from the paper's protocol.

## License

Notebook code is MIT. Anomalib is Apache-2.0 (Intel OpenVINO project). MVTec AD has its own
[research-only license](https://www.mvtec.com/company/research/datasets/mvtec-ad/license-mvtec-ad)
— review before commercial use.
