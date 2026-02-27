//! Orchestration: load image → model → geometry → cleanup → export (and OCR/LLM if enabled).

use std::path::Path;

use image::io::Reader as ImageReader;
use image::{metadata::Orientation, DynamicImage, GrayImage, ImageDecoder, ImageEncoder, RgbImage};
use crate::cli::{CleanImgMode, OutputFormat, OcrMode};
use crate::cleanup::apply_cleanup;
use crate::config::Config;
use crate::export::{write_image_png, write_pdf};
use crate::geometry::warp_quad_to_a4;
use crate::model::{quad_is_axis_aligned, Detector};
use crate::ocr::run_tesseract;

/// Process one image: load, detect, warp, cleanup, export; optionally OCR and LLM.
/// Returns Ok(()) on success; on error logs and returns Err (caller may continue batch).
pub fn process_one(
    image_path: &Path,
    detector: &mut Detector,
    outdir: &Path,
    base: &str,
    format: OutputFormat,
    cleanimg: CleanImgMode,
    dpi: u32,
    ocr_mode: OcrMode,
    use_llm: bool,
    debug_bbox: bool,
    savemask: bool,
    config: &Config,
) -> Result<(), String> {
    // Load image with EXIF orientation applied so downstream exports (PNG, mask overlay, PDF)
    // match what typical viewers show for the original.
    let img = {
        let reader = ImageReader::open(image_path)
            .map_err(|e| format!("failed to open image {}: {}", image_path.display(), e))?;
        let mut decoder = reader
            .into_decoder()
            .map_err(|e| format!("failed to create decoder for {}: {}", image_path.display(), e))?;
        let orientation = decoder
            .orientation()
            .unwrap_or(Orientation::NoTransforms);
        let mut dyn_img = DynamicImage::from_decoder(decoder)
            .map_err(|e| format!("failed to decode image {}: {}", image_path.display(), e))?;
        dyn_img.apply_orientation(orientation);
        dyn_img.to_rgb8()
    };

    let (input, meta) = Detector::letterbox_and_normalize(&img);
    let result_opt = detector
        .detect_quad(&input, &meta, savemask)
        .map_err(|e| format!("inference error: {}", e))?;

    let warped = match result_opt {
        None => {
            eprintln!(
                "warning: {}: no page detected or sanity reject, using full image",
                image_path.display()
            );
            crate::geometry::image_fit_to_a4(&img, dpi)
        }
        Some((quad, mask_opt)) => {
            eprintln!(
                "[warp] final quad passed into warp: ({:.1},{:.1}) ({:.1},{:.1}) ({:.1},{:.1}) ({:.1},{:.1})",
                quad[0].0, quad[0].1, quad[1].0, quad[1].1, quad[2].0, quad[2].1, quad[3].0, quad[3].1
            );
            if quad_is_axis_aligned(&quad) {
                eprintln!("[warp] quad is axis-aligned (bbox-only); deskew uses bbox, not mask.");
            }
            if let Some(mask) = &mask_opt {
                let overlay_path = outdir.join(format!("{}_mask.png", base));
                if let Err(e) = write_mask_overlay(&img, mask, &overlay_path) {
                    eprintln!("warning: failed to write mask overlay {}: {}", overlay_path.display(), e);
                } else {
                    eprintln!("saved mask overlay: {}", overlay_path.display());
                }
            }
            if debug_bbox {
                let mut debug_img = img.clone();
                use imageproc::drawing::draw_line_segment_mut;
                let red = image::Rgb([255u8, 0, 0]);
                for k in 0..4 {
                    let (a, b) = (quad[k], quad[(k + 1) % 4]);
                    draw_line_segment_mut(&mut debug_img, (a.0, a.1), (b.0, b.1), red);
                }
                let bbox_path = outdir.join(format!("{}_bbox.png", base));
                match std::fs::File::create(&bbox_path) {
                    Ok(mut f) => {
                        let enc = image::codecs::png::PngEncoder::new(&mut f);
                        if let Err(e) = enc.write_image(
                            debug_img.as_raw(),
                            debug_img.width(),
                            debug_img.height(),
                            image::ExtendedColorType::Rgb8,
                        ) {
                            eprintln!("warning: failed to write {}: {}", bbox_path.display(), e);
                        } else {
                            eprintln!("debug: wrote {}", bbox_path.display());
                        }
                    }
                    Err(e) => eprintln!("warning: create {}: {}", bbox_path.display(), e),
                }
            }
            match warp_quad_to_a4(&img, quad, dpi) {
                Some(w) => w,
                None => {
                    eprintln!(
                        "warning: {}: warp failed (degenerate quad?), using full image",
                        image_path.display()
                    );
                    crate::geometry::image_fit_to_a4(&img, dpi)
                }
            }
        }
    };
    let cleaned = apply_cleanup(&warped, cleanimg);

    match format {
        OutputFormat::Pdf => {
            write_pdf(&cleaned, outdir, base)
                .map_err(|e| format!("write PDF: {}", e))?;
        }
        OutputFormat::Img => {
            write_image_png(&cleaned, outdir, base)
                .map_err(|e| format!("write image: {}", e))?;
        }
        OutputFormat::Both => {
            write_pdf(&cleaned, outdir, base)
                .map_err(|e| format!("write PDF: {}", e))?;
            write_image_png(&cleaned, outdir, base)
                .map_err(|e| format!("write image: {}", e))?;
        }
    }

    if ocr_mode == OcrMode::Tesseract {
        let png_path = outdir.join(format!("{}.png", base));
        if !png_path.exists() {
            let _ = write_image_png(&cleaned, outdir, base);
        }
        let txt_path = outdir.join(format!("{}.txt", base));
        if let Some(text) = run_tesseract(&outdir.join(format!("{}.png", base))) {
            if std::fs::write(&txt_path, &text).is_err() {
                eprintln!("warning: failed to write OCR text to {}", txt_path.display());
            }
        } else {
            eprintln!("warning: tesseract not available or failed for {}", image_path.display());
        }
    }

    if use_llm {
        let ocr_text = std::fs::read_to_string(outdir.join(format!("{}.txt", base)))
            .unwrap_or_else(|_| String::new());
        let text = if ocr_text.is_empty() {
            "[No OCR text]"
        } else {
            &ocr_text
        };
        let model = config.openai_model.as_deref();
        if let Some(json) = crate::llm::call_openai(text, model) {
            let json_path = outdir.join(format!("{}.json", base));
            if std::fs::write(&json_path, serde_json::to_string_pretty(&json).unwrap_or_default()).is_err() {
                eprintln!("warning: failed to write LLM JSON to {}", json_path.display());
            }
        } else {
            eprintln!("warning: OpenAI LLM call failed for {}", image_path.display());
        }
    }

    Ok(())
}

/// Overlay detection mask on original image (semi-transparent green where mask is on) and write PNG.
fn write_mask_overlay(
    img: &RgbImage,
    mask: &GrayImage,
    path: &Path,
) -> Result<(), String> {
    if img.width() != mask.width() || img.height() != mask.height() {
        return Err(format!(
            "image size {}x{} != mask size {}x{}",
            img.width(), img.height(),
            mask.width(), mask.height()
        ));
    }
    const ALPHA: f32 = 0.45;
    const OVERLAY_R: u8 = 0;
    const OVERLAY_G: u8 = 255;
    const OVERLAY_B: u8 = 0;
    let mut out = img.clone();
    for y in 0..img.height() {
        for x in 0..img.width() {
            let m = mask.get_pixel(x, y)[0];
            if m >= 128 {
                let p = img.get_pixel(x, y);
                let r = ((1.0 - ALPHA) * p[0] as f32 + ALPHA * OVERLAY_R as f32).round() as u8;
                let g = ((1.0 - ALPHA) * p[1] as f32 + ALPHA * OVERLAY_G as f32).round() as u8;
                let b = ((1.0 - ALPHA) * p[2] as f32 + ALPHA * OVERLAY_B as f32).round() as u8;
                out.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }
    }
    let mut f = std::fs::File::create(path).map_err(|e| format!("create {}: {}", path.display(), e))?;
    let enc = image::codecs::png::PngEncoder::new(&mut f);
    enc.write_image(
        out.as_raw(),
        out.width(),
        out.height(),
        image::ExtendedColorType::Rgb8,
    ).map_err(|e| format!("write {}: {}", path.display(), e))?;
    Ok(())
}
