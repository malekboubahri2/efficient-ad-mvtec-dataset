# EfficientAD on MVTec AD — Kaggle TPU v5e-8

A self-contained Jupyter notebook (`efficientad_mvtec_demo.ipynb`) that trains
[**EfficientAD**](https://arxiv.org/abs/2303.14535) — a fast teacher–student
anomaly detector — on the [**MVTec AD**](https://www.mvtec.com/company/research/datasets/mvtec-ad)
dataset using the [**Anomalib**](https://github.com/openvinotoolkit/anomalib)
library, optimized for the **Kaggle TPU VM v5e-8** runtime (8 cores).

## Quick start

1. Open the notebook on Kaggle: *File → Import Notebook → Upload `efficientad_mvtec_demo.ipynb`*.
2. *Settings → Accelerator → **TPU VM v5e-8***.
3. Run all cells. After the install cell finishes, **Runtime → Restart session**, then re-run from the install cell.
4. Default category is `bottle`; change `CATEGORY` in the configuration cell to swap.

Wall time: roughly 5–10 min for one category on TPU v5e-8 (the first epoch is slow because XLA traces and compiles the graph; subsequent epochs are fast).

## Notebook structure

| Cell | Purpose |
|------|---------|
| 1 | TPU sanity check — imports `torch_xla` and prints the XLA device |
| 2 | Install `anomalib==2.4.1`; uninstall the legacy `pytorch-lightning` package which conflicts with anomalib's `lightning.pytorch.LightningModule` |
| 3 | Imports + seeding + `RESULTS_DIR = /kaggle/working/results` |
| 4 | Markdown reference for the 15 MVTec categories |
| 5 | Configuration: `CATEGORY`, `MODEL_SIZE`, `IMAGE_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS`, `NUM_CORES = 8` |
| 6 | `MVTecAD` datamodule — auto-downloads MVTec (~5 GB) on first run |
| 7 | `EfficientAd(model_size=...)` |
| 8 | `Engine(accelerator="tpu", devices=8)` → `engine.fit(...)` |
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
IMAGE_SIZE   = 256        # 192 reduces compile time slightly
BATCH_SIZE   = 1          # paper protocol — keep at 1 for accuracy parity
NUM_EPOCHS   = 250        # anomalib default
NUM_CORES    = 8          # TPU v5e-8
```

## Outputs

Everything lands under `/kaggle/working/results/`:

- `predictions_<category>.png` — 4-column visualization grid (cell 11)
- `single_inference_<category>.png` — single-image demo (cell 12)
- `all_categories_results.csv` — per-category metrics (cell 14, when `RUN_ALL_CATEGORIES=True`)
- `lightning_logs/` — Lightning's per-run logs and the trained checkpoint

## Known caveats

- **First fit is slow.** XLA traces and compiles the EfficientAD graph before the first training step (~1–3 min). Don't kill the cell.
- **MemoryBankMixin tail.** EfficientAD computes per-channel quantiles on CPU at the end of training. This appears as a long pause before the "✓ Training complete" line — it's working, not hung.
- **`pytorch-lightning` vs `lightning`.** Kaggle ships the legacy standalone `pytorch-lightning` package, whose `LightningModule` class is a different Python object than `lightning.pytorch.LightningModule`. Anomalib 2.x uses the latter. The install cell uninstalls the legacy package; **always restart the runtime after the install cell** so the kernel drops cached imports.
- **Don't upgrade `torch` on Kaggle TPU.** It's tied to a specific `torch_xla` build. The install cell only installs anomalib; it does not pass `-U torch`.
- **Batch size of 1.** Per the EfficientAD paper. Bumping it up would help TPU throughput but degrades accuracy parity with published numbers.

## Switching to GPU

If you want to run on a GPU runtime instead, change cell 1, 5, 8 and 12:
- Cell 1: `import torch; torch.cuda.is_available()` instead of importing `torch_xla`
- Cell 5: drop `NUM_CORES`
- Cell 8: `accelerator="gpu", devices=1`
- Cell 12: `device = "cuda"` and remove the `xm.mark_step()` call

## License

Notebook code is MIT. Anomalib is Apache-2.0 (Intel OpenVINO project). MVTec AD has its own
[research-only license](https://www.mvtec.com/company/research/datasets/mvtec-ad/license-mvtec-ad)
— review before commercial use.
