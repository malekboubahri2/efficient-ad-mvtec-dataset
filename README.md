# EfficientAD on MVTec AD — TPU branch (single chip)

A self-contained Jupyter notebook (`efficientad_mvtec_demo.ipynb`) that trains
[**EfficientAD**](https://arxiv.org/abs/2303.14535) — a fast teacher–student
anomaly detector — on the [**MVTec AD**](https://www.mvtec.com/company/research/datasets/mvtec-ad)
dataset using the [**Anomalib**](https://github.com/openvinotoolkit/anomalib)
library on **one TPU chip**.

> **Branch layout**
> - `master` — adaptive (auto-detects TPU/GPU/MPS/CPU)
> - `tpu` — **this branch**, TPU-only (`accelerator="tpu"`, `devices=1`)
> - `gpu` — single CUDA GPU (`accelerator="gpu"`, `devices=1`)
> - `cpu` — CPU-only fallback
>
> Switch with `git checkout <branch>`.

The notebook is editor-agnostic — runs in Kaggle, Colab, Jupyter Lab, Jupyter Notebook,
or VSCode. Output paths are resolved relative to the kernel's working directory.

## Why only one of the 8 chips?

Kaggle exposes TPU v5e-8 as an **8-worker pod slice**: each chip is fronted by
its own host process, and initializing all 8 requires 8 host processes — which
can't be done from a single Jupyter kernel. Setting `devices=8` triggers
Lightning's XLA launcher to fail with:

```
RuntimeError: Bad StatusOr access: UNKNOWN: TPU initialization failed:
Invalid --…_slice_builder_worker_addresses specified.
Expected 8 worker addresses, got 1.
```

With `devices=1`, Lightning runs inline on chip 0 (no `xmp.spawn`, no worker
discovery) and training works reliably. EfficientAD's `BATCH_SIZE=1` paper
protocol benefits little from 8-way replication.

To use all 8 chips you'd have to convert the notebook to a script and launch
it with `python -m torch_xla.launch script.py` — out of scope for this demo.

## Quick start

1. Open `efficientad_mvtec_demo.ipynb` in your notebook editor of choice (Kaggle, Colab, Jupyter, VSCode).
2. On Kaggle: *Settings → Accelerator → **TPU VM v5e-8***. Other editors: ensure `torch_xla` is installed and a TPU is reachable.
3. Run all cells. After the install cell finishes, **restart the kernel**, then re-run from the install cell.
4. Default category is `bottle`; change `CATEGORY` in the configuration cell to swap.

Wall time: roughly 10–15 min for one category on a single v5e chip (the first
epoch is slow because XLA traces and compiles the graph; subsequent epochs are fast).

## Notebook structure

| Cell | Purpose |
|------|---------|
| 1 | TPU sanity check — imports `torch_xla` and prints the XLA device |
| 2 | Install `anomalib==2.4.1`; uninstall the legacy `pytorch-lightning` package which conflicts with anomalib's `lightning.pytorch.LightningModule` |
| 3 | Imports + seeding + `RESULTS_DIR = ./results` (editor-agnostic) |
| 4 | Markdown reference for the 15 MVTec categories |
| 5 | Configuration: `CATEGORY`, `MODEL_SIZE`, `IMAGE_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS`, `NUM_CORES = 1`, `DATASET_ROOT = ./mvtec` |
| 6 | `MVTecAD` datamodule — auto-downloads MVTec (~5 GB) on first run |
| 7 | `EfficientAd(model_size=...)` |
| 8 | `Engine(accelerator="tpu", devices=1)` → `engine.fit(...)` |
| 9 | `engine.test(...)` — image/pixel AUROC, F1Max |
| 10 | Markdown |
| 11 | `engine.predict(...)` + 4-column matplotlib grid (image / GT mask / heatmap / binary) |
| 12 | Manual single-image inference using `xm.xla_device()` and `xm.mark_step()` |
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
NUM_CORES    = 1          # do NOT raise this; see "Why only one of the 8 chips?"
```

## Outputs

Everything lands under `./results/` (resolved relative to the kernel's working dir):

- `predictions_<category>.png` — 4-column visualization grid (cell 11)
- `single_inference_<category>.png` — single-image demo (cell 12)
- `all_categories_results.csv` — per-category metrics (cell 14, when `RUN_ALL_CATEGORIES=True`)
- `lightning_logs/` — Lightning's per-run logs and the trained checkpoint

## Known caveats

- **First fit is slow.** XLA traces and compiles the EfficientAD graph before the first training step (~1–3 min). Don't kill the cell.
- **MemoryBankMixin tail.** EfficientAD computes per-channel quantiles on CPU at the end of training. The pause before "✓ Training complete" is the model working, not hung.
- **`pytorch-lightning` vs `lightning`.** Kaggle ships the legacy standalone `pytorch-lightning` whose `LightningModule` class is a different Python object than `lightning.pytorch.LightningModule`. Anomalib 2.x uses the latter. The install cell uninstalls the legacy package; **always restart the kernel after the install cell** so old imports are dropped.
- **Don't upgrade `torch` on Kaggle TPU.** It's tied to a specific `torch_xla` build. The install cell only installs anomalib; it does not pass `-U torch`.
- **`devices=8` does not work in a notebook.** See *Why only one of the 8 chips?* above.

## License

Notebook code is MIT. Anomalib is Apache-2.0 (Intel OpenVINO project). MVTec AD has its own
[research-only license](https://www.mvtec.com/company/research/datasets/mvtec-ad/license-mvtec-ad)
— review before commercial use.
