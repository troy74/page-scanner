//! PDF + image export; datetimestamped filenames (YYYYMMDD_HHMMSS_001).

use chrono::Local;
use image::{ImageEncoder, RgbImage};
use pdf_writer::{Content, Filter, Name, Pdf, Rect, Ref};
use std::fs;
use std::path::Path;

/// Generate base filename: YYYYMMDD_HHMMSS_XXX (XXX = 001, 002, ...).
pub fn datetimestamp_base(index: usize) -> String {
    let now = Local::now();
    let ts = now.format("%Y%m%d_%H%M%S");
    format!("{}_{:03}", ts, index.max(1))
}

/// Ensure output directory exists; return path to dir.
pub fn ensure_outdir(outdir: &Path) -> std::io::Result<&Path> {
    if !outdir.exists() {
        fs::create_dir_all(outdir)?;
    }
    Ok(outdir)
}

/// Write cleaned RGB image as PNG to outdir with base name.
pub fn write_image_png(
    img: &RgbImage,
    outdir: &Path,
    base: &str,
) -> std::io::Result<std::path::PathBuf> {
    ensure_outdir(outdir)?;
    let path = outdir.join(format!("{}.png", base));
    let mut f = fs::File::create(&path)?;
    let encoder = image::codecs::png::PngEncoder::new(&mut f);
    encoder
        .write_image(
            img.as_raw(),
            img.width(),
            img.height(),
            image::ExtendedColorType::Rgb8,
        )
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    Ok(path)
}

/// Write cleaned RGB image as JPEG to outdir with base name.
pub fn write_image_jpeg(
    img: &RgbImage,
    outdir: &Path,
    base: &str,
    quality: u8,
) -> std::io::Result<std::path::PathBuf> {
    ensure_outdir(outdir)?;
    let path = outdir.join(format!("{}.jpg", base));
    let mut f = fs::File::create(&path)?;
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut f, quality);
    encoder
        .write_image(
            img.as_raw(),
            img.width(),
            img.height(),
            image::ExtendedColorType::Rgb8,
        )
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    Ok(path)
}

/// Write single-page PDF with one image (A4 media box). Image is embedded as JPEG.
/// Coordinates are adjusted in the PDF transform so the visual orientation matches PNG.
pub fn write_pdf(
    img: &RgbImage,
    outdir: &Path,
    base: &str,
) -> std::io::Result<std::path::PathBuf> {
    ensure_outdir(outdir)?;
    let path = outdir.join(format!("{}.pdf", base));

    let mut jpeg_buf = Vec::new();
    {
        let encoder =
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg_buf, 92);
        encoder
            .write_image(
                img.as_raw(),
                img.width(),
                img.height(),
                image::ExtendedColorType::Rgb8,
            )
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    }

    let (img_w, img_h) = (img.width() as f32, img.height() as f32);
    // A4 in points: short side 595.28, long side 841.89. Choose page orientation
    // based on image orientation so landscape content uses landscape A4.
    let a4_pt_short = 595.28f32;
    let a4_pt_long = 841.89f32;
    let (a4_pt_w, a4_pt_h) = if img_w > img_h {
        // Landscape image -> landscape A4
        (a4_pt_long, a4_pt_short)
    } else {
        // Portrait or square image -> portrait A4
        (a4_pt_short, a4_pt_long)
    };
    // Scale image to fit A4 while keeping aspect
    let scale = (a4_pt_w / img_w).min(a4_pt_h / img_h);
    let draw_w = img_w * scale;
    let draw_h = img_h * scale;
    // Center on A4; PDF origin is bottom-left, y up
    let x = (a4_pt_w - draw_w) / 2.0;
    let y = (a4_pt_h - draw_h) / 2.0;

    let catalog_id = Ref::new(1);
    let page_tree_id = Ref::new(2);
    let page_id = Ref::new(3);
    let image_id = Ref::new(4);
    let content_id = Ref::new(5);

    let mut pdf = Pdf::new();

    pdf.catalog(catalog_id).pages(page_tree_id);
    pdf.pages(page_tree_id).kids([page_id]).count(1);
    pdf.page(page_id)
        .parent(page_tree_id)
        .media_box(Rect::new(0.0, 0.0, a4_pt_w, a4_pt_h))
        .contents(content_id)
        .resources()
        .x_objects()
        .pair(Name(b"Im1"), image_id);

    {
        let mut img_obj = pdf.image_xobject(image_id, &jpeg_buf);
        img_obj
            .width(img.width() as i32)
            .height(img.height() as i32)
            .bits_per_component(8)
            .filter(Filter::DctDecode);
        img_obj.color_space().device_rgb();
    }

    // Content stream: draw image. Use a positive Y scale so the PDF visual
    // orientation matches the PNG (no additional flips applied here).
    let mut content = Content::new();
    content
        .save_state()
        .transform([draw_w, 0.0, 0.0, draw_h, x, y])
        .x_object(Name(b"Im1"))
        .restore_state();

    pdf.stream(content_id, &content.finish());

    let out_bytes = pdf.finish();
    fs::write(&path, out_bytes)?;
    Ok(path)
}
