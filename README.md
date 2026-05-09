# EfficientAD on MVTec AD — GPU branch (single CUDA GPU: T4 / P100 / …)

A self-contained Jupyter notebook (`efficientad_mvtec_demo.ipynb`) that trains
[**EfficientAD**](https://arxiv.org/abs/2303.14535) — a fast teacher–student
anomaly detector — on the [**MVTec AD**](https://www.mvtec.com/company/research/datasets/mvtec-ad)
dataset using the [**Anomalib**](https://github.com/openvinotoolkit/anomalib)
library on **a single CUDA GPU**. Tested on Kaggle's *GPU T4 ×2* and *GPU P100*
runtimes; works on any other CUDA device too (Colab GPU, local RTX, etc.).

> **Branch layout**
> - `master` — adaptive (auto-detects TPU/GPU/MPS/CPU)
> - `tpu` — TPU-only (`accelerator="tpu"`, `devices=1`)
> - `gpu` — **this branch**, single CUDA GPU (`accelerator="gpu"`, `devices=1`)
> - `cpu` — CPU-only fallback
>
> Switch with `git checkout <branch>`.

The notebook is editor-agnostic — runs in Kaggle, Colab, Jupyter Lab, Jupyter
Notebook, or VSCode. Output paths resolve relative to the kernel's working
directory.

## Supported GPUs

| Runtime | Visible GPUs | Per-category wall time | Notes |
|---|---|---|---|
| **Kaggle GPU T4 ×2** | 2 × T4 (16 GB each, Turing) | ~15–25 min | Cell 1 pins to `cuda:0`; second T4 stays idle |
| **Kaggle GPU P100** | 1 × P100 (16 GB, Pascal) | ~12–20 min | Slightly faster on FP32; pin is a no-op |
| **Colab GPU runtime** | 1 × T4 / L4 / A100 | varies | Whatever Colab assigns; pin is a no-op |
| **Local CUDA GPU** | host-dependent | host-dependent | Any compute capability ≥ 6.0 |

The notebook is hardware-generic — it just calls `Engine(accelerator="gpu", devices=1)`
and Lightning routes to whatever CUDA device is visible. Cell 1 prints the device
name, compute capability and memory so you know what you actually got.

## Why only one GPU?

If you're on Kaggle's *GPU T4 ×2* runtime you get two T4s, but this branch
deliberately uses just one:

- EfficientAD's published protocol uses `BATCH_SIZE=1`. Running DDP across two
  GPUs would give an effective batch of 2 — small accuracy drift vs. paper numbers.
- Single-GPU avoids the `ddp_notebook` strategy and its associated multi-process
  quirks (datamodule reload, MemoryBankMixin sync, etc.).
- Setup is dramatically simpler. The second T4 stays idle — that's fine.

On the P100 runtime (or any other single-GPU host) this is moot — there's only
one GPU anyway, and the `CUDA_VISIBLE_DEVICES=0` pin is a harmless no-op.

If you actually want both T4s, see the multi-GPU snippet at the bottom.

## Quick start

1. Open `efficientad_mvtec_demo.ipynb` in your notebook editor of choice.
2. Pick a GPU runtime:
   - **Kaggle**: *Settings → Accelerator → **GPU T4 ×2*** *or* **GPU P100**
   - **Colab**: *Runtime → Change runtime type → GPU*
   - **Local**: any CUDA-enabled PyTorch install
3. Run all cells. After the install cell finishes, **restart the kernel**, then re-run from the install cell.
4. Default category is `bottle`; change `CATEGORY` in the configuration cell to swap.

## Notebook structure

| Cell | Purpose |
|------|---------|
| 1 | GPU sanity check — pins `CUDA_VISIBLE_DEVICES=0`; prints device name, compute capability and memory |
| 2 | Install `anomalib==2.4.1`; uninstall the legacy `pytorch-lightning` package which conflicts with anomalib's `lightning.pytorch.LightningModule` |
| 3 | Imports + seeding + `RESULTS_DIR = ./results` (editor-agnostic) |
| 4 | Markdown reference for the 15 MVTec categories |
| 5 | Configuration: `CATEGORY`, `MODEL_SIZE`, `IMAGE_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS`, `DATASET_ROOT = ./mvtec` |
| 6 | `MVTecAD` datamodule — auto-downloads MVTec (~5 GB) on first run |
| 7 | `EfficientAd(model_size=...)` |
| 8 | `Engine(accelerator="gpu", devices=1)` → `engine.fit(...)` |
| 9 | `engine.test(...)` — image/pixel AUROC, F1Max |
| 10 | Markdown |
| 11 | `engine.predict(...)` + 4-column matplotlib grid (image / GT mask / heatmap / binary) |
| 12 | Manual single-image inference using `cuda` |
| 13 | Markdown |
| 14 | Optional loop: train + evaluate every category, write `all_categories_results.csv` |
| 15 | Markdown — Pi export overview |
| 16 | `engine.export(ExportType.ONNX)` → `results/exports/onnx/model.onnx` |
| 17 | Dynamic INT8 quantization → `results/exports/onnx/model_int8.onnx` |
| 18 | Markdown — Pi setup + run instructions |
| 19 | Custom-data instructions (Folder datamodule structure) |

## Configuration knobs

```python
CATEGORY     = "bottle"   # one of the 15 MVTec AD categories
MODEL_SIZE   = "small"    # "small" or "medium"
IMAGE_SIZE   = 256
BATCH_SIZE   = 1          # paper protocol — keep at 1 for accuracy parity
NUM_EPOCHS   = 250        # anomalib default
```

## Outputs

Everything lands under `./results/` (resolved relative to the kernel's working dir):

- `predictions_<category>.png` — 4-column visualization grid (cell 11)
- `single_inference_<category>.png` — single-image demo (cell 12)
- `all_categories_results.csv` — per-category metrics (cell 14, when `RUN_ALL_CATEGORIES=True`)
- `lightning_logs/` — Lightning's per-run logs and the trained checkpoint
- `exports/onnx/model.onnx` and `exports/onnx/model_int8.onnx` — produced by the export cells (see next section)

## Deploying to a Raspberry Pi

Cells 15–18 export the trained model to ONNX and run dynamic INT8 quantization
for ARM. The companion script [`pi_inference.py`](pi_inference.py) loads either
artifact and runs inference with **ONNX Runtime** on the Pi.

### Why ONNX (and not OpenVINO / TFLite / TorchScript)

- **Portable**: same `.onnx` file runs on Pi, x86, Jetson, etc.
- **Easy install on Pi**: `pip install onnxruntime` resolves to the right ARM wheel
  on Raspberry Pi OS 64-bit. No apt repos, no special builds.
- **Dynamic INT8 in one line** via `onnxruntime.quantization.quantize_dynamic` —
  weights → INT8, activation scales computed at runtime, no calibration set
  needed. For anomaly detection the accuracy hit vs. FP32 is typically negligible.

### Workflow

| Step | Where | What |
|---|---|---|
| 1 | Notebook (cell 16) | `engine.export(ExportType.ONNX)` → `results/exports/onnx/model.onnx` (~32 MB) |
| 2 | Notebook (cell 17) | `quantize_dynamic` → `results/exports/onnx/model_int8.onnx` (~8 MB) |
| 3 | `scp` | Copy the `.onnx` and `pi_inference.py` to the Pi |
| 4 | Pi shell | `pip install onnxruntime numpy pillow matplotlib` (one-time) |
| 5 | Pi shell | `python pi_inference.py model_int8.onnx my_image.jpg` |

### Expected per-image latency (Pi 4 / Pi 5, `MODEL_SIZE="small"`, 256×256)

| Model | Latency | RAM footprint |
|---|---|---|
| `model.onnx` (FP32)        | ~150–250 ms | ~70 MB |
| `model_int8.onnx` (INT8)   | ~50–100 ms  | ~25 MB |

`pi_inference.py` accepts:

```bash
python pi_inference.py MODEL IMAGE [--threshold 0.5] [--image-size 256] [--no-viz] [--warmup 1]
```

It prints latency, anomaly score and a NORMAL/ANOMALOUS verdict, and (unless
`--no-viz`) writes `<image>_heatmap.png` next to the input. Threshold defaults
to 0.5; tune it to match your precision/recall target on a held-out validation
set.

## Known caveats

- **MemoryBankMixin tail.** EfficientAD computes per-channel quantiles on CPU at the end of training. The pause before "✓ Training complete" is the model working, not hung.
- **`pytorch-lightning` vs `lightning`.** Some hosts (Kaggle, sometimes Colab) ship the legacy standalone `pytorch-lightning` whose `LightningModule` is a different Python class than `lightning.pytorch.LightningModule`. Anomalib 2.x uses the latter. The install cell uninstalls the legacy package; **always restart the kernel after the install cell**.
- **First-time MVTec download is ~5 GB.** On Kaggle, attach the public MVTec dataset and point `DATASET_ROOT` at `/kaggle/input/...` (read-only, but anomalib creates index files in the working dir).
- **P100 lacks Tensor Cores.** EfficientAD here trains in FP32 anyway, so this isn't a slowdown — but if you experiment with `precision="16-mixed"` you'll see the gains on T4 / Ampere but not on P100.

## Want both T4s?

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
slightly from the paper's protocol. (Not applicable on the P100 runtime —
there's only one GPU there.)

## License

Notebook code is MIT. Anomalib is Apache-2.0 (Intel OpenVINO project). MVTec AD has its own
[research-only license](https://www.mvtec.com/company/research/datasets/mvtec-ad/license-mvtec-ad)
— review before commercial use.
