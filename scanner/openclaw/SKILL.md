---
name: page-scanner
description: Work with the page-scanner CLI—document page detection, A4 warp, PDF/image export. Use when building, packaging, documenting, or invoking the scanner; when configuring the default or alternate ONNX model; or when explaining CLI options and model paths.
---

# page-scanner

CLI that detects a document page in PNG/JPG with an ONNX model, warps the page to a rectangle, fits to A4, applies optional cleanup/OCR, and exports to PDF or image.

## Default model

- **Default path:** `models/seg-model.onnx` relative to the **current working directory** when the binary is run.
- **Where to expect it:** Place `seg-model.onnx` in a `models/` directory next to where you run the binary (e.g. `./models/seg-model.onnx`), or run the binary from the project root so `models/seg-model.onnx` resolves.
- **Other models:** Any ONNX model can be used by passing `--model /path/to/model.onnx`; the default is only used when `--model` is omitted.

## CLI parameters

| Option | Default | Description |
|--------|---------|-------------|
| `INPUT` | (required) | Single image (png/jpg) or folder of images |
| `--output`, `-o` | — | Output path for single image (overrides `--outdir` for that file) |
| `--format` | `pdf` | `pdf`, `img`, or `both` |
| `--outdir` | `output` | Output directory |
| `--limit` | `10` | Max images when input is a folder; `0` = unlimited |
| `--model` | `models/seg-model.onnx` | Path to ONNX model |
| `--cleanimg` | `grayscale` | `default`, `original`, `grayscale`, `bw`, `highcontrast`, `crisp`, `sharp` |
| `--ocr` | `none` | `none` or `tesseract` |
| `--llm` | off | Call OpenAI with OCR text (needs build with `--features llm` and `OPENAI_API_KEY`) |
| `--debug_bbox` | off | Write `{base}_bbox.png` with detected bbox |
| `--savemask` | off | Write `{base}_mask.png` with detection mask |

## Examples

```bash
page-scanner input.jpg
page-scanner input.jpg -o out.pdf
page-scanner folder/ --limit 50 --format both
page-scanner input.jpg --model /path/to/custom.onnx
```
