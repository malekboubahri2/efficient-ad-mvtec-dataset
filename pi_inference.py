#!/usr/bin/env python3
"""EfficientAD inference on a Raspberry Pi using ONNX Runtime.

Usage:
    python pi_inference.py path/to/model.onnx path/to/image.jpg
    python pi_inference.py model_int8.onnx photo.jpg --threshold 0.5 --no-viz

What it does:
    1. Loads an ONNX model exported from the notebook (anomalib's Engine.export)
    2. Resizes + ImageNet-normalizes the input image
    3. Runs the ONNX session on CPU
    4. Prints anomaly_score + a NORMAL/ANOMALOUS verdict
    5. Optionally writes <image>_heatmap.png next to the input image

Install once on the Pi:
    pip install onnxruntime numpy pillow matplotlib
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# Same preprocessing the model was trained with (anomalib defaults to ImageNet stats)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE    = 256   # must match the IMAGE_SIZE used in the notebook


def preprocess(img_path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Load → resize → normalize → CHW with batch dim. Returns float32 (1, 3, H, W)."""
    img = Image.open(img_path).convert("RGB").resize(
        (image_size, image_size), Image.BILINEAR
    )
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr.astype(np.float32)


def make_session(model_path: Path) -> ort.InferenceSession:
    """Build a CPU-only InferenceSession with sensible thread settings for the Pi."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4   # Pi 4 / Pi 5 have 4 cores
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def parse_outputs(output_names: list[str], outputs: list[np.ndarray]) -> tuple[float, np.ndarray | None]:
    """Pull (anomaly_score, anomaly_map) out of the ONNX outputs by name with safe fallbacks."""
    by_name = dict(zip(output_names, outputs))
    score_arr = (by_name.get("pred_score")
                 or by_name.get("anomaly_score")
                 or outputs[-1])
    map_arr   = (by_name.get("anomaly_map")
                 or (outputs[0] if len(outputs) > 1 else None))
    return float(np.asarray(score_arr).squeeze()), (None if map_arr is None else np.asarray(map_arr).squeeze())


def maybe_save_heatmap(image_path: Path, anomaly_map: np.ndarray, score: float) -> None:
    """Write a side-by-side input/heatmap PNG. Silently skips if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless: no DISPLAY needed on the Pi
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping heatmap; "
              "pip install matplotlib to enable)")
        return

    out = image_path.with_name(image_path.stem + "_heatmap.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(Image.open(image_path))
    axes[0].set_title("Input"); axes[0].axis("off")
    im = axes[1].imshow(anomaly_map, cmap="inferno")
    axes[1].set_title(f"Anomaly map (score={score:.3f})"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved : {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", type=Path, help="Path to .onnx model")
    ap.add_argument("image", type=Path, help="Path to input image (jpg/png/etc)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Score threshold for the NORMAL/ANOMALOUS verdict (default: 0.5)")
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                    help=f"Resize input to this square (default: {IMAGE_SIZE})")
    ap.add_argument("--no-viz", action="store_true", help="Skip the heatmap PNG")
    ap.add_argument("--warmup", type=int, default=1,
                    help="Warmup runs before timing the real one (default: 1)")
    args = ap.parse_args()

    if not args.model.exists():
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        return 2
    if not args.image.exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 2

    print(f"Model  : {args.model}  ({args.model.stat().st_size / 1e6:.1f} MB)")
    print(f"Image  : {args.image}")

    session = make_session(args.model)
    in_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]
    x = preprocess(args.image, args.image_size)
    print(f"Input  : {in_name}  shape={x.shape}  dtype={x.dtype}")
    print(f"Outputs: {out_names}")

    # Warmup (first run pays graph-loading + memory-allocation cost)
    for _ in range(max(0, args.warmup)):
        session.run(out_names, {in_name: x})

    # Timed run
    t0 = time.perf_counter()
    outputs = session.run(out_names, {in_name: x})
    dt_ms = (time.perf_counter() - t0) * 1000

    score, amap = parse_outputs(out_names, outputs)
    label = "ANOMALOUS" if score > args.threshold else "NORMAL"

    print(f"\nLatency      : {dt_ms:6.1f} ms")
    print(f"Anomaly score: {score:.4f}")
    print(f"Threshold τ  : {args.threshold:.4f}")
    print(f"Decision     : {label}")

    if not args.no_viz and amap is not None:
        maybe_save_heatmap(args.image, amap, score)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
