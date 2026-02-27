#!/usr/bin/env python3
"""
Minimal YOLOv8 training entrypoint for local CPU runs on macOS.

Usage (from project root):
  source scanner/.venv/bin/activate
  python scanner/scripts/train.py

This script is intentionally CPU-only and uses small settings for quick
iterations on a MacBook Air.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Tuple


def resolve_data_yaml() -> Path:
    """
    Resolve the Roboflow data.yaml path relative to this file, so the script
    works regardless of the current working directory.
    """
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    data_yaml = project_root / "traindata" / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Could not find data.yaml at: {data_yaml}")
    return data_yaml


def train_with_batch_fallback(
    model,
    data_yaml: Path,
    batch_candidates: Iterable[int],
) -> Tuple[object, int]:
    """
    Try training with progressively smaller batches if we hit CPU OOM.

    Returns (results, batch_size_used) on success, or raises the last error.
    """
    from ultralytics.utils import LOGGER  # imported here to keep module import light

    base_kwargs = {
        "data": str(data_yaml),
        "epochs": 15,  # To train longer later, increase this (e.g. 50, 100, ...).
        "imgsz": 640,
        "workers": 2,
        "device": "cpu",  # Force CPU-only, suitable for MacBook Air.
        "project": "runs",
        "name": "train",
        "exist_ok": True,
    }

    last_error: Exception | None = None
    for batch in batch_candidates:
        try:
            LOGGER.info(f"Starting training with batch={batch} on CPU...")
            results = model.train(batch=batch, **base_kwargs)
            LOGGER.info(f"Training completed successfully with batch={batch}.")
            return results, batch
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message or "cuda out of memory" in message or "oom" in message:
                LOGGER.warning(
                    f"Training failed with batch={batch} due to OOM; "
                    "trying a smaller batch size if available."
                )
                last_error = exc
                continue
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("Training failed and no batch sizes were attempted.")


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(
            "Ultralytics is not installed in the current environment.\n"
            "Activate the venv and install it with:\n"
            "  source scanner/.venv/bin/activate\n"
            "  pip install ultralytics",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        data_yaml = resolve_data_yaml()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    # Model choice:
    # - This uses the YOLOv8 nano checkpoint for fast CPU training: "yolov8n.pt".
    # - To try a larger model (more accurate but slower), change to:
    #     "yolov8s.pt" (small) or "yolov8m.pt" (medium), etc.
    model = YOLO("yolov8n.pt")

    # Batch sizes to try, from larger to smaller. The helper will automatically
    # fall back if an out-of-memory error occurs.
    batch_candidates = (8, 4, 2, 1)

    try:
        results, batch_used = train_with_batch_fallback(
            model=model,
            data_yaml=data_yaml,
            batch_candidates=batch_candidates,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Training failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    # Ultralytics attaches trainer state to the model; best checkpoint path is
    # accessible via model.trainer.best in recent versions.
    best_path: Path | None = None
    trainer = getattr(model, "trainer", None)
    best_attr = getattr(trainer, "best", None) if trainer is not None else None
    if isinstance(best_attr, (str, Path)):
        best_path = Path(best_attr).resolve()

    if best_path is None:
        # Fallback: infer from results.save_dir if available.
        save_dir = getattr(results, "save_dir", None)
        if save_dir is not None:
            candidate = Path(save_dir) / "weights" / "best.pt"
            if candidate.is_file():
                best_path = candidate.resolve()

    if best_path is None or not best_path.is_file():
        print(
            "Training completed but could not locate best.pt. "
            "Check the latest run directory under runs/ for weights/best.pt.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Where best.pt is saved:
    # - By default this will be under something like:
    #     runs/detect/train/weights/best.pt
    #   (the exact task subfolder is managed by Ultralytics).
    print(
        f"Training finished on CPU with batch={batch_used}. "
        f"Best model saved to: {best_path}"
    )


if __name__ == "__main__":
    main()

