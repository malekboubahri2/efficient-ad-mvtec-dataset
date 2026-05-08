# EfficientAD on MVTec AD — CPU branch

A self-contained Jupyter notebook (`efficientad_mvtec_demo.ipynb`) that trains
[**EfficientAD**](https://arxiv.org/abs/2303.14535) — a fast teacher–student
anomaly detector — on the [**MVTec AD**](https://www.mvtec.com/company/research/datasets/mvtec-ad)
dataset using the [**Anomalib**](https://github.com/openvinotoolkit/anomalib)
library, **on CPU only**.

> **⚠️ This is slow.** Use this branch for smoke-testing the install / data
> pipeline, for laptops without a GPU, or for CI. For real training use the
> `gpu`, `tpu`, or `master` (adaptive) branches. Default `NUM_EPOCHS` here is
> **50** instead of 250 — bump back up if you actually want paper-protocol
> parity and have the patience.

> **Branch layout**
> - `master` — adaptive (auto-detects TPU/GPU/MPS/CPU)
> - `tpu` — TPU-only (`accelerator="tpu"`, `devices=1`)
> - `gpu` — single CUDA GPU (`accelerator="gpu"`, `devices=1`)
> - `cpu` — **this branch**, CPU-only fallback
>
> Switch with `git checkout <branch>`.

The notebook is editor-agnostic — runs in Kaggle, Colab, Jupyter Lab, Jupyter
Notebook, or VSCode. Output paths resolve relative to the kernel's working
directory.

## Quick start

1. Open `efficientad_mvtec_demo.ipynb` in your notebook editor.
2. No accelerator needed (this branch ignores any GPU/TPU that's available).
3. Run all cells. After the install cell finishes, **restart the kernel**, then re-run from the install cell.
4. Default category is `bottle`; change `CATEGORY` in the configuration cell to swap.

Approximate wall time for one category at `NUM_EPOCHS=50`:

| Hardware | Wall time |
|---|---|
| Modern desktop CPU (8 threads) | 1–2 hours |
| Laptop CPU (4 threads) | 3–5 hours |
| Cloud notebook CPU | 4–6 hours |

At `NUM_EPOCHS=250` (paper protocol) multiply by ~5×.

## Notebook structure

| Cell | Purpose |
|------|---------|
| 1 | CPU sanity check — prints torch version + thread count |
| 2 | Install `anomalib==2.4.1`; uninstall the legacy `pytorch-lightning` package |
| 3 | Imports + seeding + `RESULTS_DIR = ./results` (editor-agnostic) |
| 4 | Markdown reference for the 15 MVTec categories |
| 5 | Configuration: `CATEGORY`, `MODEL_SIZE`, `IMAGE_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS = 50`, `DATASET_ROOT = ./mvtec` |
| 6 | `MVTecAD` datamodule — auto-downloads MVTec (~5 GB) on first run |
| 7 | `EfficientAd(model_size=...)` |
| 8 | `Engine(accelerator="cpu", devices=1)` → `engine.fit(...)` |
| 9 | `engine.test(...)` — image/pixel AUROC, F1Max |
| 10 | Markdown |
| 11 | `engine.predict(...)` + 4-column matplotlib grid (image / GT mask / heatmap / binary) |
| 12 | Manual single-image inference on CPU |
| 13 | Markdown |
| 14 | Optional loop: train + evaluate every category, write `all_categories_results.csv` |
| 15 | Custom-data instructions (Folder datamodule structure) |

## Configuration knobs

```python
CATEGORY     = "bottle"   # one of the 15 MVTec AD categories
MODEL_SIZE   = "small"    # "small" or "medium" — leave "small" on CPU
IMAGE_SIZE   = 256
BATCH_SIZE   = 1
NUM_EPOCHS   = 50         # paper protocol is 250; bump if you have time
```

## Outputs

Everything lands under `./results/` (resolved relative to the kernel's working dir):

- `predictions_<category>.png` — 4-column visualization grid (cell 11)
- `single_inference_<category>.png` — single-image demo (cell 12)
- `all_categories_results.csv` — per-category metrics (cell 14)
- `lightning_logs/` — Lightning's per-run logs and the trained checkpoint

## Known caveats

- **CPU is fundamentally slow** for EfficientAD. There's no clever fix here — torch CPU just isn't fast on the convolutions.
- **MemoryBankMixin tail.** EfficientAD computes per-channel quantiles on CPU at the end of training; on this branch the whole run is on CPU so this isn't a separate "tail" — it's all the same regime.
- **`pytorch-lightning` vs `lightning`.** Some hosts ship the legacy standalone `pytorch-lightning` whose `LightningModule` is a different Python class than `lightning.pytorch.LightningModule`. Anomalib 2.x uses the latter. The install cell uninstalls the legacy package — **always restart the kernel after the install cell**.
- **Lower `NUM_EPOCHS` means lower accuracy** vs. the published numbers. The default of 50 is a tradeoff for runtime; raise it if you want closer parity with the paper.

## License

Notebook code is MIT. Anomalib is Apache-2.0 (Intel OpenVINO project). MVTec AD has its own
[research-only license](https://www.mvtec.com/company/research/datasets/mvtec-ad/license-mvtec-ad)
— review before commercial use.
