#!/usr/bin/env python3
"""
One-time dev script: export Teklia yolov11-generic-page .pt to ONNX,
optionally apply dynamic INT8 quantization.

By default writes two sibling models:
  - generic-page.onnx       (full precision, default for scanner CLI)
  - generic-page-int8.onnx  (INT8 quantized)

Usage:
  pip install ultralytics onnx onnxruntime torch
  python convert_and_quantize.py [path/to/model.pt]
  # If no path given, downloads from HuggingFace Teklia/yolov11-generic-page.

  -o, --output     path for quantized ONNX (default: generic-page-int8.onnx)
  --output-fp      path for full-precision ONNX (default: generic-page.onnx)
  --fp-only        only export full-precision ONNX to -o; skip quantization
"""

import argparse
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv11 .pt to ONNX and optionally quantize to INT8")
    parser.add_argument(
        "model_pt",
        nargs="?",
        default=None,
        help="Path to model.pt (default: download from HuggingFace Teklia/yolov11-generic-page)",
    )
    parser.add_argument(
        "-o", "--output",
        default="generic-page-int8.onnx",
        help="Output path for quantized ONNX (default: generic-page-int8.onnx)",
    )
    parser.add_argument(
        "--output-fp",
        default="generic-page.onnx",
        help="Output path for full-precision ONNX (default: generic-page.onnx)",
    )
    parser.add_argument(
        "--fp-only",
        action="store_true",
        help="Only export full-precision ONNX to -o; skip INT8 quantization",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("Install onnx and onnxruntime: pip install onnx onnxruntime", file=sys.stderr)
        sys.exit(1)

    model_path = args.model_pt
    if not model_path:
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id="Teklia/yolov11-generic-page",
                filename="model.pt",
            )
        except ImportError:
            print(
                "Install huggingface_hub to auto-download, or pass path to model.pt",
                file=sys.stderr,
            )
            sys.exit(1)

    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # 1) Export .pt -> ONNX (imgsz=640 per Teklia card)
    print("Loading model and exporting to ONNX...")
    model = YOLO(model_path)
    export_path = model.export(format="onnx", imgsz=640, simplify=True)
    onnx_temp = str(export_path) if export_path else model_path.replace(".pt", ".onnx")
    if not isinstance(onnx_temp, str):
        onnx_temp = getattr(onnx_temp, "path", onnx_temp)
    if not os.path.isfile(onnx_temp):
        onnx_temp = os.path.join(os.path.dirname(model_path), "model.onnx")
    if not os.path.isfile(onnx_temp):
        print("Could not find exported ONNX path.", file=sys.stderr)
        sys.exit(1)
    print(f"Exported ONNX: {onnx_temp}")

    if args.fp_only:
        # Only write full-precision to -o
        out_fp = args.output
        os.makedirs(os.path.dirname(out_fp) or ".", exist_ok=True)
        shutil.copy2(onnx_temp, out_fp)
        if onnx_temp != out_fp and os.path.abspath(onnx_temp) != os.path.abspath(out_fp):
            try:
                os.remove(onnx_temp)
            except OSError:
                pass
        print(f"Done (full precision): {out_fp}")
        return

    # Write full-precision sibling
    os.makedirs(os.path.dirname(args.output_fp) or ".", exist_ok=True)
    shutil.copy2(onnx_temp, args.output_fp)
    print(f"Full precision: {args.output_fp}")

    # Dynamic INT8 quantization via ONNX Runtime
    print("Quantizing to INT8...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("Install onnxruntime for quantization: pip install onnxruntime", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    quantize_dynamic(
        model_input=onnx_temp,
        model_output=args.output,
        weight_type=QuantType.QInt8,
    )
    if onnx_temp != args.output and os.path.abspath(onnx_temp) != os.path.abspath(args.output):
        try:
            os.remove(onnx_temp)
        except OSError:
            pass
    print(f"Done: {args.output_fp} (default), {args.output} (INT8)")


if __name__ == "__main__":
    main()
