# EfficientAD on MVTec AD — Adaptive (auto-detect TPU / CUDA / MPS / CPU)

A self-contained Jupyter notebook (`efficientad_mvtec_demo.ipynb`) that trains
[**EfficientAD**](https://arxiv.org/abs/2303.14535) — a fast teacher–student
anomaly detector — on the [**MVTec AD**](https://www.mvtec.com/company/research/datasets/mvtec-ad)
dataset using the [**Anomalib**](https://github.com/openvinotoolkit/anomalib)
library. Picks the best available accelerator at runtime.

> **Branch layout**
> - `master` — **this branch**, adaptive (auto-detects TPU/CUDA/MPS/CPU)
> - `tpu` — TPU-only (`accelerator="tpu"`, `devices=1`)
> - `gpu` — single CUDA GPU (`accelerator="gpu"`, `devices=1`)
> - `cpu` — CPU-only fallback
>
> Use `master` if you want a single notebook that "just works" anywhere. Use a
> specialized branch if you want a stripped-down notebook with no detection logic.

The notebook is editor-agnostic — runs in Kaggle, Colab, Jupyter Lab, Jupyter
Notebook, or VSCode. Output paths resolve relative to the kernel's working
directory.

## Accelerator priority

Cell 1 walks this list and stops at the first match:

| Priority | Detected when… | Used as |
|---|---|---|
| 1. **TPU** | `torch_xla` imports and `xm.xla_device()` succeeds | `accelerator="tpu",  devices=1` |
| 2. **CUDA GPU** | `torch.cuda.is_available()` | `accelerator="gpu",  devices=1` (pinned to GPU 0) |
| 3. **Apple MPS** | `torch.backends.mps.is_available()` (Apple Silicon) | `accelerator="mps",  devices=1` |
| 4. **CPU** | fallback | `accelerator="cpu",  devices=1` |

`devices=1` everywhere by design:
- TPU pods on Kaggle expose 8 chips behind 8 separate worker hosts; only one is
  reachable from a notebook kernel (see the `tpu` branch README for details).
- Multi-GPU DDP requires `strategy="ddp_notebook"` and would shift the effective
  batch off the EfficientAD `BATCH_SIZE=1` paper protocol.

## Quick start

1. Open `efficientad_mvtec_demo.ipynb` in your notebook editor of choice.
2. (If on Kaggle / Colab) pick the accelerator runtime — TPU, GPU, or none, the notebook handles it.
3. Run all cells. After the install cell finishes, **restart the kernel**, then re-run from the install cell.
4. Default category is `bottle`; change `CATEGORY` in the configuration cell to swap.

Approximate wall time for one category:

| Backend | Wall time |
|---|---|
| TPU v5e (1 chip) | ~10–15 min |
| CUDA T4 / equivalent | ~15–25 min |
| Apple M-series MPS | varies; ~30–60 min |
| CPU | hours; only useful for smoke-testing |

## Notebook structure

| Cell | Purpose |
|------|---------|
| 1 | Detect accelerator → `ACCELERATOR`, `NUM_DEVICES` |
| 2 | Install `anomalib==2.4.1`; uninstall the legacy `pytorch-lightning` package |
| 3 | Imports + seeding + `RESULTS_DIR = ./results` (editor-agnostic) |
| 4 | Markdown reference for the 15 MVTec categories |
| 5 | Configuration: `CATEGORY`, `MODEL_SIZE`, `IMAGE_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS`, `DATASET_ROOT = ./mvtec` |
| 6 | `MVTecAD` datamodule — auto-downloads MVTec (~5 GB) on first run |
| 7 | `EfficientAd(model_size=...)` |
| 8 | `Engine(accelerator=ACCELERATOR, devices=NUM_DEVICES)` → `engine.fit(...)` |
| 9 | `engine.test(...)` — image/pixel AUROC, F1Max |
| 10 | Markdown |
| 11 | `engine.predict(...)` + 4-column matplotlib grid (image / GT mask / heatmap / binary) |
| 12 | Manual single-image inference, device chosen by `_torch_device(ACCELERATOR)` |
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

`ACCELERATOR` and `NUM_DEVICES` are detected, not configured. To force a
specific accelerator, override them in cell 1 immediately after the detection
block.

## Outputs

Everything lands under `./results/` (resolved relative to the kernel's working dir):

- `predictions_<category>.png` — 4-column visualization grid (cell 11)
- `single_inference_<category>.png` — single-image demo (cell 12)
- `all_categories_results.csv` — per-category metrics (cell 14)
- `lightning_logs/` — Lightning's per-run logs and the trained checkpoint

## Known caveats

- **MemoryBankMixin tail.** EfficientAD computes per-channel quantiles on CPU at the end of training — the pause before "✓ Training complete" is the model working, not hung.
- **`pytorch-lightning` vs `lightning`.** Some hosts (Kaggle, sometimes Colab) ship the legacy standalone `pytorch-lightning` whose `LightningModule` is a different Python class than `lightning.pytorch.LightningModule`. Anomalib 2.x uses the latter. The install cell uninstalls the legacy package — **always restart the kernel after the install cell**.
- **TPU first.** If `torch_xla` is importable and a TPU is reachable, the notebook uses it even if a CUDA GPU is also present. Override in cell 1 if you'd rather use the GPU.
- **TPU on Kaggle = 1 chip only.** The `tpu` branch README explains why `devices=8` doesn't work in a notebook.

## License

Notebook code is MIT. Anomalib is Apache-2.0 (Intel OpenVINO project). MVTec AD has its own
[research-only license](https://www.mvtec.com/company/research/datasets/mvtec-ad/license-mvtec-ad)
— review before commercial use.
