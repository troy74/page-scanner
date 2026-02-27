//! Cleanup filters: grayscale, original, default, bw, highcontrast, crisp, sharp (image crate only).

use image::{ImageBuffer, Luma, RgbImage};
use imageproc::map::map_pixels;

use crate::cli::CleanImgMode;

/// Apply the selected cleanup mode to the image.
pub fn apply_cleanup(img: &RgbImage, mode: CleanImgMode) -> RgbImage {
    match mode {
        CleanImgMode::Original => img.clone(),
        CleanImgMode::Grayscale => grayscale_clean(img),
        CleanImgMode::Default => default_clean(img),
        CleanImgMode::Bw => bw_clean(img),
        CleanImgMode::Highcontrast => highcontrast_clean(img),
        CleanImgMode::Crisp => crisp_clean(img),
        CleanImgMode::Sharp => sharp_clean(img),
    }
}

/// Mild contrast stretch + light sharpening.
fn default_clean(img: &RgbImage) -> RgbImage {
    let mut out = contrast_stretch(img, 0.02, 0.98);
    out = unsharp_mask(&out, 1.0, 1.5);
    out
}

/// Pure grayscale (no thresholding).
fn grayscale_clean(img: &RgbImage) -> RgbImage {
    let gray = image::imageops::colorops::grayscale(img);
    let (w, h) = gray.dimensions();
    let mut out = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = gray.get_pixel(x, y)[0];
            out.put_pixel(x, y, image::Rgb([v, v, v]));
        }
    }
    out
}

/// Grayscale + adaptive threshold (block-based).
fn bw_clean(img: &RgbImage) -> RgbImage {
    let gray = image::imageops::colorops::grayscale(img);
    let (w, h) = gray.dimensions();
    let block = 31.max((w.min(h) / 20) | 1);
    let thresh = adaptive_threshold(&gray, block);
    let (w, h) = thresh.dimensions();
    let mut out = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = thresh.get_pixel(x, y)[0];
            out.put_pixel(x, y, image::Rgb([v, v, v]));
        }
    }
    out
}

/// Aggressive contrast normalization (full range stretch).
fn highcontrast_clean(img: &RgbImage) -> RgbImage {
    contrast_stretch(img, 0.0, 1.0)
}

/// Mild denoise (3x3 median) + edge enhance (simple kernel).
fn crisp_clean(img: &RgbImage) -> RgbImage {
    let denoised = imageproc::filter::median_filter(img, 1, 1);
    edge_enhance(&denoised)
}

/// Unsharp mask + moderate contrast.
fn sharp_clean(img: &RgbImage) -> RgbImage {
    let mut out = unsharp_mask(img, 1.2, 2.0);
    out = contrast_stretch(&out, 0.05, 0.95);
    out
}

/// Percentile-based contrast stretch; low/high in [0,1] (e.g. 0.02, 0.98).
fn contrast_stretch(img: &RgbImage, low: f32, high: f32) -> RgbImage {
    let mut r: Vec<u8> = img.pixels().map(|p| p[0]).collect();
    let mut g: Vec<u8> = img.pixels().map(|p| p[1]).collect();
    let mut b: Vec<u8> = img.pixels().map(|p| p[2]).collect();
    r.sort_unstable();
    g.sort_unstable();
    b.sort_unstable();
    let n = r.len();
    let lo_r = r[(n as f32 * low) as usize].min(r[0]);
    let hi_r = r[(n as f32 * high) as usize].max(*r.last().unwrap_or(&255));
    let lo_g = g[(n as f32 * low) as usize].min(g[0]);
    let hi_g = g[(n as f32 * high) as usize].max(*g.last().unwrap_or(&255));
    let lo_b = b[(n as f32 * low) as usize].min(b[0]);
    let hi_b = b[(n as f32 * high) as usize].max(*b.last().unwrap_or(&255));
    let span_r = (hi_r - lo_r).max(1) as f32;
    let span_g = (hi_g - lo_g).max(1) as f32;
    let span_b = (hi_b - lo_b).max(1) as f32;
    map_pixels(img, |_x, _y, p| {
        image::Rgb([
            (((p[0] as f32 - lo_r as f32) / span_r).clamp(0.0, 1.0) * 255.0) as u8,
            (((p[1] as f32 - lo_g as f32) / span_g).clamp(0.0, 1.0) * 255.0) as u8,
            (((p[2] as f32 - lo_b as f32) / span_b).clamp(0.0, 1.0) * 255.0) as u8,
        ])
    })
}

/// Simple unsharp: blur then add (original - blur) * amount; sigma for gaussian.
fn unsharp_mask(img: &RgbImage, amount: f32, sigma: f32) -> RgbImage {
    let blurred = imageproc::filter::gaussian_blur_f32(img, sigma);
    map_pixels(img, |x, y, p| {
        let b = blurred.get_pixel(x, y);
        image::Rgb([
            (p[0] as f32 + (p[0] as f32 - b[0] as f32) * amount).clamp(0.0, 255.0) as u8,
            (p[1] as f32 + (p[1] as f32 - b[1] as f32) * amount).clamp(0.0, 255.0) as u8,
            (p[2] as f32 + (p[2] as f32 - b[2] as f32) * amount).clamp(0.0, 255.0) as u8,
        ])
    })
}

/// Block-based adaptive threshold (mean of block as threshold).
/// Uses an integral image so runtime is O(w*h) instead of O(w*h*block^2).
fn adaptive_threshold(
    gray: &ImageBuffer<Luma<u8>, Vec<u8>>,
    block_size: u32,
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (w, h) = gray.dimensions();
    let w = w as usize;
    let h = h as usize;
    // Summed-area table: sat[i,j] = sum of gray[0..i, 0..j). Size (w+1)*(h+1).
    let mut sat = vec![0u64; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row_sum = 0u64;
        for x in 0..w {
            row_sum += gray.get_pixel(x as u32, y as u32)[0] as u64;
            sat[(y + 1) * (w + 1) + (x + 1)] = sat[y * (w + 1) + (x + 1)] + row_sum;
        }
    }
    let half = (block_size / 2) as i32;
    let mut out = ImageBuffer::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let x1 = (x as i32 - half).max(0) as usize;
            let y1 = (y as i32 - half).max(0) as usize;
            let x2 = (x as i32 + half).min(w as i32 - 1).max(0) as usize;
            let y2 = (y as i32 + half).min(h as i32 - 1).max(0) as usize;
            // Rect sum = sat(y2+1,x2+1) - sat(y2+1,x1) - sat(y1,x2+1) + sat(y1,x1)
            let sum = sat[(y2 + 1) * (w + 1) + (x2 + 1)]
                .wrapping_sub(sat[(y2 + 1) * (w + 1) + x1])
                .wrapping_sub(sat[y1 * (w + 1) + (x2 + 1)])
                .wrapping_add(sat[y1 * (w + 1) + x1]);
            let count = ((x2 - x1 + 1) * (y2 - y1 + 1)) as u64;
            let mean = if count > 0 { (sum / count) as u8 } else { 128 };
            let v = gray.get_pixel(x as u32, y as u32)[0];
            // Bias the threshold so only pixels significantly darker than the local
            // mean become black. This suppresses paper texture and background noise
            // while keeping true text strokes.
            const MARGIN: u8 = 15;
            let thr = mean.saturating_sub(MARGIN);
            let out_val = if v < thr { 0 } else { 255 };
            out.put_pixel(x as u32, y as u32, Luma([out_val]));
        }
    }
    out
}

/// Simple edge enhance: laplacian-like kernel 3x3 (row-major).
fn edge_enhance(img: &RgbImage) -> RgbImage {
    let data = [
        0f32, -1f32, 0f32,
        -1f32, 5f32, -1f32,
        0f32, -1f32, 0f32,
    ];
    imageproc::filter::filter3x3(img, &data)
}
