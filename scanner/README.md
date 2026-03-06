# scanner

Document page scanner CLI: detect page in PNG/JPG via a quantized ONNX YOLOv11 model, warp the page region (irregular quadrilateral when available) to a rectangle with minimal stretch, fit to A4, apply cleanup filters, export to PDF or image. Optional OCR (tesseract) and LLM (OpenAI).

**Ship target:** Linux. **Develop and test:** macOS and Linux (same codebase).

## Build

- Rust toolchain (rustup).
- No PyTorch/Ultralytics in production; the binary loads only a pre-built ONNX model.

```bash
cargo build --release
```

Optional LLM feature (OpenAI):

```bash
cargo build --release --features llm
```

## Model

Place the quantized ONNX model at:

- **Default:** `models/seg-model.onnx` (relative to the current working directory when you run `page-scanner`)
- Or pass `--model /path/to/model.onnx` to point at any compatible ONNX model

To create the model (dev only, Python):

1. Create a venv and install: `python3 -m venv .venv && .venv/bin/pip install ultralytics onnx onnxruntime torch huggingface_hub`
2. Run: `.venv/bin/python scripts/convert_and_quantize.py` (or pass path to `model.pt`). Output: `generic-page-int8.onnx` in the project dir.
3. Copy or rename `generic-page-int8.onnx` to `models/seg-model.onnx` next to the binary, or pass its path explicitly via `--model /path/to/generic-page-int8.onnx`.

## Usage

```text
scanner <INPUT>              # single image or folder
scanner input.jpg -o out.pdf
scanner folder/ --limit 50
scanner input.jpg --format pdf   # default
scanner input.jpg --format img
scanner input.jpg --format both
scanner input.jpg --cleanimg bw
scanner input.jpg --ocr tesseract
scanner input.jpg --llm           # requires --features llm and OPENAI_API_KEY
scanner input.jpg --model /path/to/model.onnx
scanner --help
```

- **Input:** Single PNG/JPG or folder (default folder limit: 10; `--limit 0` = unlimited).
- **Output:** Default `./output/`; filenames `YYYYMMDD_HHMMSS_001.pdf` (and same base for `.png`, `.txt`, `.json`).
- **Cleanup:** `--cleanimg default|bw|highcontrast|crisp|sharp`.
- **OCR:** `--ocr tesseract` (best-effort; warns if tesseract missing).
- **LLM:** `--llm` (best-effort; requires `OPENAI_API_KEY` and build with `--features llm`).

## Development (macOS)

- Build and run on macOS: `cargo run -- input.jpg`, `cargo test`.
- Install tesseract for OCR: `brew install tesseract`.
- Put `seg-model.onnx` in `./models/` (project root) or use `--model /path/to/model.onnx`.
- Config (optional): `~/.scanner/config.toml` (e.g. `default_dpi`, `default_cleanimg`, `openai_model`). CLI overrides config.

## Prebuilt binaries

This repo includes convenience binaries for common platforms under `scanner/binaries/`:

- `binaries/macos/page-scanner` – macOS build (Apple Silicon), expects `models/seg-model.onnx` alongside it (provided in `binaries/macos/models/seg-model.onnx`).
- `binaries/linux x86/page-scanner` – x86_64 Linux build, built on Ubuntu 24.04 (glibc 2.38), expects `models/seg-model.onnx` alongside it (provided in `binaries/linux x86/models/seg-model.onnx`).

For other environments, build from source with `cargo build --release` and ensure `models/seg-model.onnx` is present relative to your working directory or pass `--model`.

## OpenClaw skill

An OpenClaw skill definition for `page-scanner` lives at `scanner/openclaw/SKILL.md`. To use it:

1. Create a skill folder in your OpenClaw skills directory (for example `~/.openclaw/skills/page-scanner/`).
2. Copy the appropriate binary (`page-scanner`) from `scanner/binaries/<platform>/` into that folder.
3. Copy the matching `models/seg-model.onnx` into a `models/` subfolder next to the binary (so the binary can resolve `models/seg-model.onnx` by default).
4. Place the `SKILL.md` from `scanner/openclaw/` into that skill folder (or keep it in sync if you already initialized the skill there).
5. Follow OpenClaw’s instructions to enable or reference the `page-scanner` skill so the agent knows it can call this CLI.

## Geometry

- The pipeline warps the detected page region to a rectangle, then fits to A4. The target rectangle’s aspect is chosen from the quad’s effective width/height (average opposite edge lengths) to **minimize length/width stretch**. The result is then scaled to fit inside A4 (portrait or landscape). The model is a segmentation model; when mask decoding is implemented, the quad will be the irregular 4-point outline from the page mask; until then the quad is derived from the detection bounding box (axis-aligned).

## Assumptions

- Ship target: Linux. Develop/test on macOS and Linux.
- Model file is pre-built and placed in `~/.scanner/models/` or given via `--model`.
- Quad: currently the four corners of the best bounding box (axis-aligned); future: irregular quad from mask contour.
- A4 aspect 1∶√2; portrait/landscape from content; 300 DPI default.
- OCR and LLM are best-effort; missing tesseract or API failure does not fail the batch.
- Default output format is PDF; default folder limit is 10.
- Config is optional; CLI overrides config.
- `hf://` model download is out of scope for the initial deliverable.
