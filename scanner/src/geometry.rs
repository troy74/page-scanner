//! Irregular quad → rectangle (minimize stretch) → A4 fit (1∶√2).
//! Quad is four corners in order (e.g. top-left, top-right, bottom-right, bottom-left);
//! target rectangle aspect is chosen from quad effective width/height to avoid stretch.

use image::{ImageBuffer, RgbImage};
use imageproc::geometric_transformations::{Interpolation, Projection};

use crate::model::{Bbox, Quad};

/// A4 aspect ratio 1∶√2. At 300 DPI: 2480×3508 (portrait) or 3508×2480 (landscape).
const A4_DPI: u32 = 300;
const A4_SHORT_MM: f32 = 210.0;
const A4_LONG_MM: f32 = 297.0;

fn sq(x: f32) -> f32 {
    x * x
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    (sq(b.0 - a.0) + sq(b.1 - a.1)).sqrt()
}

fn quad_signed_area(quad: &Quad) -> f32 {
    // Shoelace formula for 4-point polygon.
    let mut area = 0.0f32;
    for i in 0..4 {
        let j = (i + 1) % 4;
        area += quad[i].0 * quad[j].1;
        area -= quad[j].0 * quad[i].1;
    }
    area * 0.5
}

/// Normalize quad corner order:
/// - Accept 4 arbitrary points in image coordinates (origin top-left, y down).
/// - Return [top-left, top-right, bottom-right, bottom-left] with consistent winding.
/// - If winding is flipped, swap TR and BL to avoid mirroring while preserving diagonal.
fn normalize_quad(quad: Quad) -> Quad {
    // Identify corners using sum (x+y) and diff (x-y) heuristics.
    let mut tl = quad[0];
    let mut br = quad[0];
    let mut tr = quad[0];
    let mut bl = quad[0];

    let mut min_sum = quad[0].0 + quad[0].1;
    let mut max_sum = min_sum;
    let mut min_diff = quad[0].0 - quad[0].1;
    let mut max_diff = min_diff;

    for &(x, y) in &quad {
        let sum = x + y;
        let diff = x - y;

        if sum < min_sum {
            min_sum = sum;
            tl = (x, y);
        }
        if sum > max_sum {
            max_sum = sum;
            br = (x, y);
        }
        if diff < min_diff {
            min_diff = diff;
            tr = (x, y);
        }
        if diff > max_diff {
            max_diff = diff;
            bl = (x, y);
        }
    }

    let mut ordered: Quad = [tl, tr, br, bl];

    // Ensure consistent winding. For our image coordinate system (y down),
    // we treat a positive signed area as the desired orientation. If area
    // is negative, swap TR and BL, which flips left/right without
    // changing the TL<->BR diagonal.
    if quad_signed_area(&ordered) < 0.0 {
        ordered.swap(1, 3);
    }

    ordered
}

/// Clamp aspect ratio to avoid extreme stretch (e.g. very tall or very wide target rect).
const ASPECT_MIN: f32 = 0.2;
const ASPECT_MAX: f32 = 5.0;

/// Compute effective width and height of quad from opposite edge lengths (minimizes stretch).
/// Edges: 0-1 top, 1-2 right, 2-3 bottom, 3-0 left.
/// Returns (width_eff, height_eff) for aspect ratio width/height; aspect is clamped to avoid extremes.
pub fn quad_effective_size(quad: Quad) -> (f32, f32) {
    let top = dist(quad[0], quad[1]);
    let right = dist(quad[1], quad[2]);
    let bottom = dist(quad[2], quad[3]);
    let left = dist(quad[3], quad[0]);
    let w_eff = (top + bottom) * 0.5;
    let h_eff = (left + right) * 0.5;
    let raw_aspect = w_eff / h_eff.max(1e-6);
    let aspect = raw_aspect.clamp(ASPECT_MIN, ASPECT_MAX);
    let h_eff_clamped = w_eff / aspect;
    (w_eff, h_eff_clamped)
}

/// Compute A4 pixel dimensions: short = 210mm, long = 297mm at given DPI.
fn a4_pixels(dpi: u32, portrait: bool) -> (u32, u32) {
    let mm_to_inch = 25.4;
    let short_px = (A4_SHORT_MM / mm_to_inch * dpi as f32).round() as u32;
    let long_px = (A4_LONG_MM / mm_to_inch * dpi as f32).round() as u32;
    if portrait {
        (short_px, long_px)
    } else {
        (long_px, short_px)
    }
}

/// Quad as four corners (axis-aligned); re-export for convenience.
pub fn bbox_to_quad(bbox: Bbox) -> Quad {
    crate::model::bbox_to_quad(bbox)
}

/// Fit full image into A4 (scale to fit, letterbox). Use as fallback when warp fails.
pub fn image_fit_to_a4(image: &RgbImage, dpi: u32) -> RgbImage {
    let (img_w, img_h) = (image.width() as f32, image.height() as f32);
    let aspect = img_w / img_h.max(1.0);
    let portrait = aspect <= 1.0;
    let (a4_w, a4_h) = a4_pixels(dpi, portrait);
    let scale = (a4_w as f32 / img_w).min(a4_h as f32 / img_h);
    let fit_w = (img_w * scale).round() as u32;
    let fit_h = (img_h * scale).round() as u32;
    let fit_w = fit_w.min(a4_w).max(1);
    let fit_h = fit_h.min(a4_h).max(1);
    let scaled = image::imageops::resize(
        image,
        fit_w,
        fit_h,
        image::imageops::FilterType::Triangle,
    );
    let mut out = ImageBuffer::from_pixel(a4_w, a4_h, image::Rgb([255u8, 255, 255]));
    let x = (a4_w - fit_w) / 2;
    let y = (a4_h - fit_h) / 2;
    image::imageops::overlay(&mut out, &scaled, x as i64, y as i64);
    out
}

/// Warp image so the given quad maps to a rectangle with aspect = quad effective aspect
/// (minimizes length/width stretch), then fit that rectangle into A4.
/// When aspect > 1 (landscape content), uses landscape A4; otherwise portrait A4.
pub fn warp_quad_to_a4(image: &RgbImage, quad: Quad, dpi: u32) -> Option<RgbImage> {
    let quad = normalize_quad(quad);
    let (w_eff, h_eff) = quad_effective_size(quad);
    if w_eff < 1.0 || h_eff < 1.0 {
        return None;
    }
    let aspect = w_eff / h_eff;
    let portrait = aspect <= 1.0; // landscape content (aspect > 1) -> landscape A4 page
    let (a4_w, a4_h) = a4_pixels(dpi, portrait);
    let _a4_aspect = a4_w as f32 / a4_h as f32;

    // Intermediate rect: same aspect as quad (no stretch). Use area ~ A4 so resolution is good.
    let a4_area = (a4_w * a4_h) as f32;
    let rect_w = (a4_area * aspect).sqrt();
    let rect_h = rect_w / aspect;
    let rect_w = rect_w.round() as u32;
    let rect_h = rect_h.round() as u32;
    let (rect_w, rect_h) = (rect_w.max(1), rect_h.max(1));

    let target_rect: Quad = [
        (0.0, 0.0),
        (rect_w as f32, 0.0),
        (rect_w as f32, rect_h as f32),
        (0.0, rect_h as f32),
    ];
    let projection = Projection::from_control_points(quad, target_rect)?;
    let mut warped = ImageBuffer::new(rect_w, rect_h);
    imageproc::geometric_transformations::warp_into(
        image,
        &projection,
        Interpolation::Bilinear,
        image::Rgb([255u8, 255, 255]),
        &mut warped,
    );

    // Fit warped (rect_w × rect_h) into A4 (a4_w × a4_h), preserving aspect.
    let scale = (a4_w as f32 / rect_w as f32).min(a4_h as f32 / rect_h as f32);
    let fit_w = (rect_w as f32 * scale).round() as u32;
    let fit_h = (rect_h as f32 * scale).round() as u32;
    let fit_w = fit_w.min(a4_w);
    let fit_h = fit_h.min(a4_h);
    let scaled = image::imageops::resize(
        &warped,
        fit_w,
        fit_h,
        image::imageops::FilterType::Triangle,
    );
    let mut out = ImageBuffer::from_pixel(a4_w, a4_h, image::Rgb([255u8, 255, 255]));
    let x = (a4_w - fit_w) / 2;
    let y = (a4_h - fit_h) / 2;
    image::imageops::overlay(&mut out, &scaled, x as i64, y as i64);
    Some(out)
}

/// Convenience: warp from bbox (axis-aligned quad) to A4.
pub fn warp_bbox_to_a4(image: &RgbImage, bbox: Bbox, dpi: u32) -> Option<RgbImage> {
    warp_quad_to_a4(image, bbox_to_quad(bbox), dpi)
}

/// Warp from irregular quad to A4 (same as warp_quad_to_a4; alias for API clarity).
pub fn warp_irregular_quad_to_a4(
    image: &RgbImage,
    quad: crate::model::Quad,
    dpi: u32,
) -> Option<RgbImage> {
    warp_quad_to_a4(image, quad, dpi)
}
