//! ONNX session init, letterbox + normalize, run, parse outputs → best bbox or quad.
//! Supports: YOLOv11 segmentation (dim >= 37), YOLOv8 OBB (dim == 9), standard detection (dim 6 or 7).

use image::GrayImage;
use imageproc::contours::{find_contours_with_threshold, BorderType, Contour};
use imageproc::geometry::{contour_area, convex_hull, min_area_rect};
use imageproc::point::Point;
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;
use ort::Error as OrtError;
use std::path::Path;

const MODEL_INPUT_SIZE: u32 = 640;
const SCORE_THRESHOLD: f32 = 0.12;
const MIN_BBOX_FRAC: f32 = 0.10;   // reject if width or height < 10% of image
const MAX_OUTSIDE_FRAC: f32 = 0.50; // reject if >50% of unclamped box outside image
const MIN_QUAD_AREA_FRAC: f32 = 0.01; // fall back to bbox if mask quad area < 1% of image

/// Output format inferred from detection tensor shape dims[1].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectOutputType {
    /// [1, 37+, N] — segmentation (bbox + mask coeffs)
    Segmentation,
    /// [1, 9, N] — YOLOv8 OBB (cx, cy, w, h, theta, obj, class, ...)
    Obb,
    /// [1, 6, N] or [1, 7, N] — standard detection (no mask)
    Detection,
    /// Unknown shape, cannot parse
    Unknown,
}

/// Detect output type from tensor shape. Logs the result.
pub fn detect_output_type(dims: &[usize]) -> DetectOutputType {
    let ty = if dims.len() != 3 {
        DetectOutputType::Unknown
    } else {
        let dim = dims[1];
        if dim >= 37 {
            DetectOutputType::Segmentation
        } else if dim == 9 {
            DetectOutputType::Obb
        } else if dim == 6 || dim == 7 {
            DetectOutputType::Detection
        } else {
            DetectOutputType::Unknown
        }
    };
    eprintln!("[warp] detected model type: {:?} (dims[1] = {})", ty, dims.get(1).copied().unwrap_or(0));
    ty
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Letterbox metadata: scale and pad so we can map bbox back to original image.
#[derive(Debug, Clone)]
pub struct LetterboxMeta {
    pub scale: f32,
    pub pad_left: f32,
    pub pad_top: f32,
    pub orig_w: u32,
    pub orig_h: u32,
}

/// Axis-aligned bounding box in original image coordinates (xyxy).
#[derive(Debug, Clone, Copy)]
pub struct Bbox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// Four corners of a quadrilateral (TL, TR, BR, BL or irregular from mask).
pub type Quad = [(f32, f32); 4];

/// Axis-aligned bbox to quad (four corners).
pub fn bbox_to_quad(bbox: Bbox) -> Quad {
    [
        (bbox.x1, bbox.y1),
        (bbox.x2, bbox.y1),
        (bbox.x2, bbox.y2),
        (bbox.x1, bbox.y2),
    ]
}

/// Approximate area of quad (for fallback checks). Uses shoelace for simple polygon.
pub fn quad_area(quad: &Quad) -> f32 {
    let mut area = 0.0f32;
    for i in 0..4 {
        let j = (i + 1) % 4;
        area += quad[i].0 * quad[j].1;
        area -= quad[j].0 * quad[i].1;
    }
    (area / 2.0).abs()
}

/// True if quad is axis-aligned (two distinct x, two distinct y) — i.e. from bbox, not mask.
pub fn quad_is_axis_aligned(quad: &Quad) -> bool {
    let xs: std::collections::HashSet<u32> = quad
        .iter()
        .map(|p| (p.0 * 1000.0).round() as u32)
        .collect();
    let ys: std::collections::HashSet<u32> = quad
        .iter()
        .map(|p| (p.1 * 1000.0).round() as u32)
        .collect();
    xs.len() == 2 && ys.len() == 2
}

pub struct Detector {
    session: Session,
}

impl Detector {
    pub fn new(model_path: &Path) -> Result<Self, OrtError> {
        let session = Session::builder()?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }

    /// Prepare image: letterbox to 640x640, normalize to [0,1], return tensor and meta.
    pub fn letterbox_and_normalize(
        image: &image::RgbImage,
    ) -> (Array4<f32>, LetterboxMeta) {
        let (w, h) = (image.width() as f32, image.height() as f32);
        let scale = (MODEL_INPUT_SIZE as f32 / w).min(MODEL_INPUT_SIZE as f32 / h);
        let new_w = (w * scale).round() as u32;
        let new_h = (h * scale).round() as u32;
        let pad_w = MODEL_INPUT_SIZE - new_w;
        let pad_h = MODEL_INPUT_SIZE - new_h;
        let pad_left = (pad_w / 2) as f32;
        let pad_top = (pad_h / 2) as f32;

        let resized = image::imageops::resize(
            image,
            new_w,
            new_h,
            image::imageops::FilterType::Triangle,
        );
        let mut padded = image::RgbImage::new(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
        image::imageops::overlay(
            &mut padded,
            &resized,
            pad_left as i64,
            pad_top as i64,
        );

        // NCHW, normalize 0..255 -> 0..1
        let mut arr = Array4::<f32>::zeros((1, 3, MODEL_INPUT_SIZE as usize, MODEL_INPUT_SIZE as usize));
        for y in 0..MODEL_INPUT_SIZE {
            for x in 0..MODEL_INPUT_SIZE {
                let p = padded.get_pixel(x, y);
                arr[[0, 0, y as usize, x as usize]] = p[0] as f32 / 255.0;
                arr[[0, 1, y as usize, x as usize]] = p[1] as f32 / 255.0;
                arr[[0, 2, y as usize, x as usize]] = p[2] as f32 / 255.0;
            }
        }

        let meta = LetterboxMeta {
            scale,
            pad_left,
            pad_top,
            orig_w: image.width(),
            orig_h: image.height(),
        };
        (arr, meta)
    }

    /// Run inference; returns (output0 data, output0 shape) and optionally (output1 data, output1 shape).
    fn run_inference(
        &mut self,
        input: &Array4<f32>,
    ) -> Result<
        (
            (Vec<f32>, Vec<usize>),
            Option<(Vec<f32>, Vec<usize>)>,
        ),
        OrtError,
    > {
        let shape: [i64; 4] = [
            1,
            3,
            MODEL_INPUT_SIZE as i64,
            MODEL_INPUT_SIZE as i64,
        ];
        let data: Vec<f32> = input.iter().copied().collect();
        let input_value = Tensor::from_array((shape, data))?.into_dyn();
        let input_name = self
            .session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "images".to_string());
        let outputs = self.session.run(ort::inputs![input_name.as_str() => input_value])?;
        if outputs.len() == 0 {
            return Err(OrtError::new("no output"));
        }
        let (out0_shape, out0_slice) = outputs[0].try_extract_tensor::<f32>()?;
        let dims0: Vec<usize> = out0_shape.iter().map(|&d| d as usize).collect();
        let out0_data: Vec<f32> = out0_slice.to_vec();
        let out1 = if outputs.len() >= 2 {
            let (s, sl) = outputs[1].try_extract_tensor::<f32>()?;
            Some((sl.to_vec(), s.iter().map(|&d| d as usize).collect()))
        } else {
            None
        };
        Ok(((out0_data, dims0), out1))
    }

    /// Run inference and return best bbox in original image coordinates. OBB output has no bbox.
    pub fn detect(&mut self, input: &Array4<f32>, meta: &LetterboxMeta) -> Result<Option<Bbox>, OrtError> {
        let ((out0_data, dims), _) = self.run_inference(input)?;
        eprintln!("[detect] ONNX output shape: {:?}", dims);
        let output_type = detect_output_type(&dims);
        let best = match output_type {
            DetectOutputType::Obb | DetectOutputType::Unknown => None,
            DetectOutputType::Segmentation => parse_segmentation(&out0_data, &dims)?,
            DetectOutputType::Detection => parse_detection(&out0_data, &dims)?,
        };
        let bbox = match best {
            Some(det) => {
                let (lb_x1, lb_y1, lb_x2, lb_y2, score) =
                    (det.lb_x1, det.lb_y1, det.lb_x2, det.lb_y2, det.score);
                // 8) Unletterbox: 640 letterbox -> original image
                let x1_orig = (lb_x1 - meta.pad_left) / meta.scale;
                let y1_orig = (lb_y1 - meta.pad_top) / meta.scale;
                let x2_orig = (lb_x2 - meta.pad_left) / meta.scale;
                let y2_orig = (lb_y2 - meta.pad_top) / meta.scale;
                let w_img = meta.orig_w as f32;
                let h_img = meta.orig_h as f32;

                // 9) Sanity rejection BEFORE warp
                let unclamped_w = x2_orig - x1_orig;
                let unclamped_h = y2_orig - y1_orig;
                let inside_left = x1_orig.max(0.0).min(w_img);
                let inside_right = x2_orig.max(0.0).min(w_img);
                let inside_top = y1_orig.max(0.0).min(h_img);
                let inside_bottom = y2_orig.max(0.0).min(h_img);
                let inside_w = (inside_right - inside_left).max(0.0);
                let inside_h = (inside_bottom - inside_top).max(0.0);
                let inside_area = inside_w * inside_h;
                let total_area = unclamped_w * unclamped_h;
                let outside_frac = if total_area > 1e-6 {
                    (1.0 - inside_area / total_area).max(0.0)
                } else {
                    1.0
                };

                let clamped_w = (x2_orig.min(w_img) - x1_orig.max(0.0)).max(0.0);
                let clamped_h = (y2_orig.min(h_img) - y1_orig.max(0.0)).max(0.0);
                let reject_width = clamped_w < MIN_BBOX_FRAC * w_img;
                let reject_height = clamped_h < MIN_BBOX_FRAC * h_img;
                let reject_outside = outside_frac > MAX_OUTSIDE_FRAC;

                if reject_width || reject_height || reject_outside {
                    eprintln!(
                        "[detect] sanity reject: width_frac={:.2} height_frac={:.2} outside_frac={:.2} (will use full image)",
                        clamped_w / w_img,
                        clamped_h / h_img,
                        outside_frac
                    );
                    return Ok(None);
                }

                let x1 = x1_orig.max(0.0).min(w_img);
                let y1 = y1_orig.max(0.0).min(h_img);
                let x2 = x2_orig.max(0.0).min(w_img);
                let y2 = y2_orig.max(0.0).min(h_img);
                eprintln!(
                    "[detect] letterbox bbox=({:.1},{:.1},{:.1},{:.1}) orig=({:.1},{:.1},{:.1},{:.1}) score={:.3}",
                    lb_x1, lb_y1, lb_x2, lb_y2, x1, y1, x2, y2, score
                );
                Some(Bbox { x1, y1, x2, y2 })
            }
            None => None,
        };
        Ok(bbox)
    }

    /// Run inference and return best detection as a quadrilateral in original image coordinates.
    /// Supports OBB (dim 9), segmentation (dim >= 37), and standard detection (dim 6/7).
    pub fn detect_quad(
        &mut self,
        input: &Array4<f32>,
        meta: &LetterboxMeta,
        save_mask: bool,
    ) -> Result<Option<(Quad, Option<GrayImage>)>, OrtError> {
        let ((out0_data, dims), proto_opt) = self.run_inference(input)?;

        eprintln!("[warp] ONNX output0 (detections) shape: {:?}", dims);
        let output_type = detect_output_type(&dims);

        match output_type {
            DetectOutputType::Obb => {
                let (quad_lb, score) = match parse_obb(&out0_data, &dims)? {
                    Some(x) => x,
                    None => return Ok(None),
                };
                let quad_orig = unletterbox_quad(&quad_lb, meta);
                if !sanity_accept_quad(&quad_orig, meta) {
                    eprintln!("[detect] sanity reject: OBB quad failed bbox fraction or outside check");
                    return Ok(None);
                }
                eprintln!(
                    "[warp] final quad (OBB): ({:.1},{:.1}) ({:.1},{:.1}) ({:.1},{:.1}) ({:.1},{:.1}) score={:.3}",
                    quad_orig[0].0, quad_orig[0].1, quad_orig[1].0, quad_orig[1].1,
                    quad_orig[2].0, quad_orig[2].1, quad_orig[3].0, quad_orig[3].1, score
                );
                return Ok(Some((quad_orig, None)));
            }
            DetectOutputType::Detection => {
                let best = match parse_detection(&out0_data, &dims)? {
                    Some(d) => d,
                    None => return Ok(None),
                };
                let x1_orig = (best.lb_x1 - meta.pad_left) / meta.scale;
                let y1_orig = (best.lb_y1 - meta.pad_top) / meta.scale;
                let x2_orig = (best.lb_x2 - meta.pad_left) / meta.scale;
                let y2_orig = (best.lb_y2 - meta.pad_top) / meta.scale;
                let w_img = meta.orig_w as f32;
                let h_img = meta.orig_h as f32;
                let (x1, y1, x2, y2) = (
                    x1_orig.max(0.0).min(w_img),
                    y1_orig.max(0.0).min(h_img),
                    x2_orig.max(0.0).min(w_img),
                    y2_orig.max(0.0).min(h_img),
                );
                let bbox_orig = Bbox { x1, y1, x2, y2 };
                let unclamped_w = x2_orig - x1_orig;
                let unclamped_h = y2_orig - y1_orig;
                let inside_left = x1_orig.max(0.0).min(w_img);
                let inside_right = x2_orig.max(0.0).min(w_img);
                let inside_top = y1_orig.max(0.0).min(h_img);
                let inside_bottom = y2_orig.max(0.0).min(h_img);
                let inside_w = (inside_right - inside_left).max(0.0);
                let inside_h = (inside_bottom - inside_top).max(0.0);
                let inside_area = inside_w * inside_h;
                let total_area = unclamped_w * unclamped_h;
                let outside_frac = if total_area > 1e-6 {
                    (1.0 - inside_area / total_area).max(0.0)
                } else {
                    1.0
                };
                let clamped_w = (x2.min(w_img) - x1.max(0.0)).max(0.0);
                let clamped_h = (y2.min(h_img) - y1.max(0.0)).max(0.0);
                if clamped_w < MIN_BBOX_FRAC * w_img
                    || clamped_h < MIN_BBOX_FRAC * h_img
                    || outside_frac > MAX_OUTSIDE_FRAC
                {
                    eprintln!("[detect] sanity reject: detection bbox");
                    return Ok(None);
                }
                let quad_orig = bbox_to_quad(bbox_orig);
                eprintln!(
                    "[warp] final quad (detection): ({:.1},{:.1}) ({:.1},{:.1}) ({:.1},{:.1}) ({:.1},{:.1}) score={:.3}",
                    quad_orig[0].0, quad_orig[0].1, quad_orig[1].0, quad_orig[1].1,
                    quad_orig[2].0, quad_orig[2].1, quad_orig[3].0, quad_orig[3].1, best.score
                );
                return Ok(Some((quad_orig, None)));
            }
            DetectOutputType::Unknown => return Ok(None),
            DetectOutputType::Segmentation => {}
        }

        // Segmentation path (dim >= 37)
        match &proto_opt {
            Some((_, proto_shape)) => {
                eprintln!("[warp] ONNX output1 (mask prototypes) shape: {:?}", proto_shape);
            }
            None => {}
        }
        let best = match parse_segmentation(&out0_data, &dims)? {
            Some(d) => d,
            None => return Ok(None),
        };
        let (lb_x1, lb_y1, lb_x2, lb_y2) = (best.lb_x1, best.lb_y1, best.lb_x2, best.lb_y2);
        let w_img = meta.orig_w as f32;
        let h_img = meta.orig_h as f32;
        let x1_orig = (lb_x1 - meta.pad_left) / meta.scale;
        let y1_orig = (lb_y1 - meta.pad_top) / meta.scale;
        let x2_orig = (lb_x2 - meta.pad_left) / meta.scale;
        let y2_orig = (lb_y2 - meta.pad_top) / meta.scale;
        let unclamped_w = x2_orig - x1_orig;
        let unclamped_h = y2_orig - y1_orig;
        let inside_left = x1_orig.max(0.0).min(w_img);
        let inside_right = x2_orig.max(0.0).min(w_img);
        let inside_top = y1_orig.max(0.0).min(h_img);
        let inside_bottom = y2_orig.max(0.0).min(h_img);
        let inside_w = (inside_right - inside_left).max(0.0);
        let inside_h = (inside_bottom - inside_top).max(0.0);
        let inside_area = inside_w * inside_h;
        let total_area = unclamped_w * unclamped_h;
        let outside_frac = if total_area > 1e-6 {
            (1.0 - inside_area / total_area).max(0.0)
        } else {
            1.0
        };
        let clamped_w = (x2_orig.min(w_img) - x1_orig.max(0.0)).max(0.0);
        let clamped_h = (y2_orig.min(h_img) - y1_orig.max(0.0)).max(0.0);
        if clamped_w < MIN_BBOX_FRAC * w_img
            || clamped_h < MIN_BBOX_FRAC * h_img
            || outside_frac > MAX_OUTSIDE_FRAC
        {
            eprintln!(
                "[detect] sanity reject: width_frac={:.2} height_frac={:.2} outside_frac={:.2}",
                clamped_w / w_img,
                clamped_h / h_img,
                outside_frac
            );
            return Ok(None);
        }

        let bbox_orig = Bbox {
            x1: x1_orig.max(0.0).min(w_img),
            y1: y1_orig.max(0.0).min(h_img),
            x2: x2_orig.max(0.0).min(w_img),
            y2: y2_orig.max(0.0).min(h_img),
        };
        let bbox_quad = bbox_to_quad(bbox_orig);
        eprintln!(
            "[warp] selected detection: bbox (unletterbox) = ({:.1},{:.1},{:.1},{:.1}), score = {:.3}",
            bbox_orig.x1, bbox_orig.y1, bbox_orig.x2, bbox_orig.y2, best.score
        );
        eprintln!(
            "[warp] bbox_to_quad = ({:.1},{:.1}) ({:.1},{:.1}) ({:.1},{:.1}) ({:.1},{:.1})",
            bbox_quad[0].0, bbox_quad[0].1, bbox_quad[1].0, bbox_quad[1].1,
            bbox_quad[2].0, bbox_quad[2].1, bbox_quad[3].0, bbox_quad[3].1
        );

        let (quad_orig, mask_for_save): (Quad, Option<GrayImage>) = if let (Some((proto_data, proto_shape)), coeffs) =
            (proto_opt.as_ref(), best.mask_coeffs.as_slice())
        {
            if coeffs.len() == 32
                && proto_shape.len() >= 4
                && proto_shape[1] == 32
            {
                if let Some((q, mask_canvas)) = mask_to_quad(
                    proto_data,
                    proto_shape,
                    coeffs,
                    best.lb_x1,
                    best.lb_y1,
                    best.lb_x2,
                    best.lb_y2,
                    meta,
                ) {
                    let area = quad_area(&q);
                    let img_area = w_img * h_img;
                    if img_area > 0.0 && area < img_area * MIN_QUAD_AREA_FRAC {
                        eprintln!(
                            "[warp] mask quad area too small ({:.0} < {:.0}% of image), using bbox",
                            area, MIN_QUAD_AREA_FRAC * 100.0
                        );
                        (bbox_quad, if save_mask { Some(mask_canvas) } else { None })
                    } else {
                        (q, if save_mask { Some(mask_canvas) } else { None })
                    }
                } else {
                    eprintln!("[warp] mask decode or contour failed, using bbox quad");
                    (bbox_quad, None)
                }
            } else {
                eprintln!(
                    "[warp] proto shape or coeffs invalid (coeffs={}, proto_shape={:?}), using bbox",
                    coeffs.len(),
                    proto_shape
                );
                (bbox_quad, None)
            }
        } else {
            eprintln!("[warp] no mask output, using bbox quad");
            (bbox_quad, None)
        };

        Ok(Some((quad_orig, mask_for_save)))
    }
}

fn bbox_to_quad_letterbox(lb_x1: f32, lb_y1: f32, lb_x2: f32, lb_y2: f32) -> [(f32, f32); 4] {
    [
        (lb_x1, lb_y1),
        (lb_x2, lb_y1),
        (lb_x2, lb_y2),
        (lb_x1, lb_y2),
    ]
}

/// Convert OBB (cx, cy, w, h, theta) in letterbox space to four corners in letterbox space.
/// Local corners (-dx,-dy), (dx,-dy), (dx,dy), (-dx,dy); rotate by theta around (cx,cy).
fn obb_to_quad_letterbox(cx: f32, cy: f32, w: f32, h: f32, theta: f32) -> Quad {
    let dx = w / 2.0;
    let dy = h / 2.0;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let local: [(f32, f32); 4] = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)];
    let mut quad: Quad = [(0.0, 0.0); 4];
    for (k, (x, y)) in local.iter().enumerate() {
        quad[k] = (
            cx + x * cos_t - y * sin_t,
            cy + x * sin_t + y * cos_t,
        );
    }
    quad
}

/// Unletterbox a quad from 640 letterbox space to original image coordinates.
fn unletterbox_quad(quad_lb: &Quad, meta: &LetterboxMeta) -> Quad {
    let mut q: Quad = [(0.0, 0.0); 4];
    for k in 0..4 {
        q[k] = (
            (quad_lb[k].0 - meta.pad_left) / meta.scale,
            (quad_lb[k].1 - meta.pad_top) / meta.scale,
        );
    }
    q
}

/// Axis-aligned bbox of a quad (min x, min y, max x, max y).
fn quad_bbox(quad: &Quad) -> (f32, f32, f32, f32) {
    let xs = [quad[0].0, quad[1].0, quad[2].0, quad[3].0];
    let ys = [quad[0].1, quad[1].1, quad[2].1, quad[3].1];
    let x1 = xs.iter().copied().fold(f32::INFINITY, f32::min);
    let y1 = ys.iter().copied().fold(f32::INFINITY, f32::min);
    let x2 = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let y2 = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (x1, y1, x2, y2)
}

/// Sanity check for a quad in original image space: same rules as bbox (min size, max outside).
fn sanity_accept_quad(quad_orig: &Quad, meta: &LetterboxMeta) -> bool {
    let (x1, y1, x2, y2) = quad_bbox(quad_orig);
    let w_img = meta.orig_w as f32;
    let h_img = meta.orig_h as f32;
    let unclamped_w = x2 - x1;
    let unclamped_h = y2 - y1;
    let inside_left = x1.max(0.0).min(w_img);
    let inside_right = x2.max(0.0).min(w_img);
    let inside_top = y1.max(0.0).min(h_img);
    let inside_bottom = y2.max(0.0).min(h_img);
    let inside_w = (inside_right - inside_left).max(0.0);
    let inside_h = (inside_bottom - inside_top).max(0.0);
    let inside_area = inside_w * inside_h;
    let total_area = unclamped_w * unclamped_h;
    let outside_frac = if total_area > 1e-6 {
        (1.0 - inside_area / total_area).max(0.0)
    } else {
        1.0
    };
    let clamped_w = (x2.min(w_img) - x1.max(0.0)).max(0.0);
    let clamped_h = (y2.min(h_img) - y1.max(0.0)).max(0.0);
    if clamped_w < MIN_BBOX_FRAC * w_img || clamped_h < MIN_BBOX_FRAC * h_img {
        return false;
    }
    if outside_frac > MAX_OUTSIDE_FRAC {
        return false;
    }
    true
}

const MASK_THRESHOLD: f32 = 0.5;
const LETTERBOX_SIZE: f32 = 640.0;

/// Decode mask at proto resolution; crop to detection bbox in proto space; resize crop to orig bbox region; place on orig canvas; contour -> quad.
/// Returns (quad, mask_canvas) so caller can optionally save the mask overlay.
fn mask_to_quad(
    proto_data: &[f32],
    proto_shape: &[usize],
    coeffs: &[f32],
    lb_x1: f32,
    lb_y1: f32,
    lb_x2: f32,
    lb_y2: f32,
    meta: &LetterboxMeta,
) -> Option<([(f32, f32); 4], GrayImage)> {
    if proto_shape.len() < 4 || proto_shape[1] != 32 || coeffs.len() != 32 {
        return None;
    }
    let (_, _, ph, pw) = (proto_shape[0], proto_shape[1], proto_shape[2], proto_shape[3]);
    let ph_f = ph as f32;
    let pw_f = pw as f32;

    // 1) Decode mask at prototype resolution (e.g. 160x160)
    let mut mask_small = vec![0.0f32; ph * pw];
    for y in 0..ph {
        for x in 0..pw {
            let mut sum = 0.0f32;
            for c in 0..32 {
                let idx = c * ph * pw + y * pw + x;
                sum += coeffs[c] * proto_data.get(idx).copied().unwrap_or(0.0);
            }
            mask_small[y * pw + x] = sigmoid(sum);
        }
    }

    // 2) Bbox in letterbox space is (lb_x1, lb_y1, lb_x2, lb_y2) @ 640x640
    // 3) Map bbox from letterbox (640) to proto resolution (pw x ph)
    let scale_lb_to_proto_x = pw_f / LETTERBOX_SIZE;
    let scale_lb_to_proto_y = ph_f / LETTERBOX_SIZE;
    let proto_x1 = (lb_x1 * scale_lb_to_proto_x).max(0.0).min(pw_f - 1.0);
    let proto_y1 = (lb_y1 * scale_lb_to_proto_y).max(0.0).min(ph_f - 1.0);
    let proto_x2 = (lb_x2 * scale_lb_to_proto_x).max(0.0).min(pw_f);
    let proto_y2 = (lb_y2 * scale_lb_to_proto_y).max(0.0).min(ph_f);

    let crop_x1 = proto_x1.floor() as usize;
    let crop_y1 = proto_y1.floor() as usize;
    let crop_x2 = (proto_x2.ceil() as usize).min(pw);
    let crop_y2 = (proto_y2.ceil() as usize).min(ph);
    let crop_w = crop_x2.saturating_sub(crop_x1).max(1);
    let crop_h = crop_y2.saturating_sub(crop_y1).max(1);

    // 4) Crop mask_small to bbox region in proto space
    let mut crop_buf = vec![0.0f32; crop_w * crop_h];
    for oy in 0..crop_h {
        for ox in 0..crop_w {
            let sy = crop_y1 + oy;
            let sx = crop_x1 + ox;
            crop_buf[oy * crop_w + ox] = mask_small[sy * pw + sx];
        }
    }

    // 5) Orig bbox (letterbox inverse)
    let x1_orig = (lb_x1 - meta.pad_left) / meta.scale;
    let y1_orig = (lb_y1 - meta.pad_top) / meta.scale;
    let x2_orig = (lb_x2 - meta.pad_left) / meta.scale;
    let y2_orig = (lb_y2 - meta.pad_top) / meta.scale;
    let orig_bbox_w = (x2_orig - x1_orig).max(1.0);
    let orig_bbox_h = (y2_orig - y1_orig).max(1.0);

    // 6) Resize cropped mask to orig bbox size and place on full orig canvas
    let out_w = orig_bbox_w.round() as u32;
    let out_h = orig_bbox_h.round() as u32;
    let out_w = out_w.min(meta.orig_w).max(1);
    let out_h = out_h.min(meta.orig_h).max(1);

    let resized_crop = resize_mask_to_size_threshold(&crop_buf, crop_w, crop_h, out_w as usize, out_h as usize);

    let x1_px = (x1_orig.round() as i32).max(0).min(meta.orig_w as i32) as u32;
    let y1_px = (y1_orig.round() as i32).max(0).min(meta.orig_h as i32) as u32;
    let paste_w = (out_w).min(meta.orig_w.saturating_sub(x1_px));
    let paste_h = (out_h).min(meta.orig_h.saturating_sub(y1_px));

    let mut mask_orig = GrayImage::new(meta.orig_w, meta.orig_h);
    for y in 0..paste_h {
        for x in 0..paste_w {
            let p = resized_crop.get_pixel(x, y);
            mask_orig.put_pixel(x1_px + x, y1_px + y, *p);
        }
    }

    eprintln!(
        "[warp] mask decoded: proto crop {}x{} -> orig region {}x{} at ({},{}), canvas {}x{}, non-zero pixels: {}",
        crop_w, crop_h, out_w, out_h, x1_px, y1_px, mask_orig.width(), mask_orig.height(),
        mask_nonzero_count(&mask_orig)
    );
    contour_to_quad(&mask_orig).map(|q| (q, mask_orig))
}

/// Resize small float mask to out_w x out_h and threshold to 0/255.
fn resize_mask_to_size_threshold(
    small: &[f32],
    w: usize,
    h: usize,
    out_w: usize,
    out_h: usize,
) -> GrayImage {
    let mut out = GrayImage::new(out_w as u32, out_h as u32);
    for oy in 0..out_h {
        for ox in 0..out_w {
            let sx = (ox as f32 * w as f32 / out_w as f32).min((w - 1) as f32).max(0.0) as usize;
            let sy = (oy as f32 * h as f32 / out_h as f32).min((h - 1) as f32).max(0.0) as usize;
            let v = small[sy * w + sx];
            out.put_pixel(
                ox as u32,
                oy as u32,
                image::Luma([if v >= MASK_THRESHOLD { 255 } else { 0 }]),
            );
        }
    }
    out
}

fn mask_nonzero_count(mask: &GrayImage) -> u64 {
    mask.pixels()
        .filter(|p| p[0] >= 128)
        .count() as u64
}

fn resize_mask_to_size(
    small: &[f32],
    w: usize,
    h: usize,
    out_w: usize,
    out_h: usize,
) -> GrayImage {
    let mut out = GrayImage::new(out_w as u32, out_h as u32);
    for oy in 0..out_h {
        for ox in 0..out_w {
            let sx = (ox as f32 * w as f32 / out_w as f32).min((w - 1) as f32).max(0.0) as usize;
            let sy = (oy as f32 * h as f32 / out_h as f32).min((h - 1) as f32).max(0.0) as usize;
            let v = small[sy * w + sx];
            out.put_pixel(
                ox as u32,
                oy as u32,
                image::Luma([if v >= MASK_THRESHOLD { 255 } else { 0 }]),
            );
        }
    }
    out
}

/// Envelope-based quad from hull: use minAreaRect only for orientation, project all hull
/// points onto the two principal axes, take min/max along each, reconstruct quad so it
/// fully covers mask support (no inward bias). Returns [TL, TR, BR, BL] for warp.
fn contour_to_quad_envelope(hull: &[Point<i32>]) -> Option<[(f32, f32); 4]> {
    if hull.len() < 3 {
        return None;
    }
    let rect = min_area_rect(hull);
    let ox = (rect[0].x + rect[1].x + rect[2].x + rect[3].x) as f32 / 4.0;
    let oy = (rect[0].y + rect[1].y + rect[2].y + rect[3].y) as f32 / 4.0;
    let ux = (rect[1].x - rect[0].x) as f32;
    let uy = (rect[1].y - rect[0].y) as f32;
    let u_len = (ux * ux + uy * uy).sqrt().max(1e-8);
    let ux = ux / u_len;
    let uy = uy / u_len;
    let vx = (rect[3].x - rect[0].x) as f32;
    let vy = (rect[3].y - rect[0].y) as f32;
    let v_len = (vx * vx + vy * vy).sqrt().max(1e-8);
    let vx = vx / v_len;
    let vy = vy / v_len;
    let (mut min_u, mut max_u, mut min_v, mut max_v) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for p in hull {
        let px = p.x as f32;
        let py = p.y as f32;
        let u_coord = (px - ox) * ux + (py - oy) * uy;
        let v_coord = (px - ox) * vx + (py - oy) * vy;
        min_u = min_u.min(u_coord);
        max_u = max_u.max(u_coord);
        min_v = min_v.min(v_coord);
        max_v = max_v.max(v_coord);
    }
    let mut quad: [(f32, f32); 4] = [
        (ox + min_u * ux + min_v * vx, oy + min_u * uy + min_v * vy),
        (ox + max_u * ux + min_v * vx, oy + max_u * uy + min_v * vy),
        (ox + max_u * ux + max_v * vx, oy + max_u * uy + max_v * vy),
        (ox + min_u * ux + max_v * vx, oy + min_u * uy + max_v * vy),
    ];
    order_quad_corners(&mut quad);
    Some(quad)
}

/// Find largest outer contour, convex hull, then envelope-based quad (minAreaRect orientation
/// + project all hull points onto axes, min/max, reconstruct). Orders as TL/TR/BR/BL.
fn contour_to_quad(mask: &GrayImage) -> Option<[(f32, f32); 4]> {
    let contours = find_contours_with_threshold::<i32>(mask, 1);
    let outer: Vec<&Contour<i32>> = contours
        .iter()
        .filter(|c| c.border_type == BorderType::Outer)
        .collect();
    let best = outer
        .iter()
        .max_by(|a, b| {
            contour_area(&a.points)
                .partial_cmp(&contour_area(&b.points))
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
    let contour_point_count = best.points.len();
    if best.points.len() < 3 {
        eprintln!(
            "[warp] mask contour: point count {} (< 3), cannot form quad",
            contour_point_count
        );
        return None;
    }
    let hull = convex_hull(&best.points);
    let quad = contour_to_quad_envelope(&hull)?;
    eprintln!(
        "[warp] mask contour: point count {}, hull {}, envelope quad",
        contour_point_count,
        hull.len()
    );
    eprintln!(
        "[warp] final quad (mask): ({:.1},{:.1}) ({:.1},{:.1}) ({:.1},{:.1}) ({:.1},{:.1})",
        quad[0].0, quad[0].1, quad[1].0, quad[1].1, quad[2].0, quad[2].1, quad[3].0, quad[3].1
    );
    Some(quad)
}

/// Order quad as [TL, TR, BR, BL] for projection (0,0)->TL, (w,0)->TR, (w,h)->BR, (0,h)->BL.
/// Deterministic document-style ordering: TL = min(x+y), BR = max(x+y), TR = min(x-y), BL = max(x-y).
fn order_quad_corners(quad: &mut [(f32, f32); 4]) {
    let points: [(f32, f32); 4] = *quad;
    let tl = *points
        .iter()
        .min_by(|a, b| (a.0 + a.1).partial_cmp(&(b.0 + b.1)).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let br = *points
        .iter()
        .max_by(|a, b| (a.0 + a.1).partial_cmp(&(b.0 + b.1)).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let tr = *points
        .iter()
        .min_by(|a, b| (a.0 - a.1).partial_cmp(&(b.0 - b.1)).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let bl = *points
        .iter()
        .max_by(|a, b| (a.0 - a.1).partial_cmp(&(b.0 - b.1)).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    quad[0] = tl;
    quad[1] = tr;
    quad[2] = br;
    quad[3] = bl;
}

/// Best detection: bbox in letterbox space, score, and mask coefficients (length 32 when dim >= 37).
pub struct BestDetection {
    pub lb_x1: f32,
    pub lb_y1: f32,
    pub lb_x2: f32,
    pub lb_y2: f32,
    pub score: f32,
    pub mask_coeffs: Vec<f32>,
}

/// Parse YOLOv8 OBB output shape [1, 9, N]. Returns best (quad in letterbox space, score) or None.
/// Per detection: cx, cy, w, h, theta, obj, class; score = sigmoid(obj)*sigmoid(class).
fn parse_obb(data: &[f32], shape: &[usize]) -> Result<Option<(Quad, f32)>, OrtError> {
    if shape.len() != 3 || shape[1] != 9 {
        return Ok(None);
    }
    let n = shape[2];
    if n == 0 {
        return Ok(None);
    }
    let mut candidates: Vec<(f32, f32, f32, f32, f32, f32)> = Vec::with_capacity(n);
    for i in 0..n {
        let cx = data.get(0 * n + i).copied().unwrap_or(0.0);
        let cy = data.get(1 * n + i).copied().unwrap_or(0.0);
        let w = data.get(2 * n + i).copied().unwrap_or(0.0);
        let h = data.get(3 * n + i).copied().unwrap_or(0.0);
        let theta = data.get(4 * n + i).copied().unwrap_or(0.0);
        let obj_raw = data.get(5 * n + i).copied().unwrap_or(0.0);
        let class_raw = data.get(6 * n + i).copied().unwrap_or(0.0);
        let obj = sigmoid(obj_raw);
        let class_conf = sigmoid(class_raw);
        let score = obj * class_conf;
        candidates.push((cx, cy, w, h, theta, score));
    }
    candidates.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap_or(std::cmp::Ordering::Equal));
    for (idx, c) in candidates.iter().take(3).enumerate() {
        eprintln!(
            "[detect] top#{} score={:.3} OBB cx={:.1} cy={:.1} w={:.1} h={:.1} theta={:.3}",
            idx + 1, c.5, c.0, c.1, c.2, c.3, c.4
        );
    }
    let best = match candidates.into_iter().find(|c| c.5 >= SCORE_THRESHOLD) {
        Some(b) => b,
        None => return Ok(None),
    };
    let (cx, cy, w, h, theta, score) = best;
    let max_coord = [cx, cy, w, h].iter().copied().fold(0.0f32, f32::max);
    let (cx, cy, w, h) = if max_coord <= 1.0 {
        let s = MODEL_INPUT_SIZE as f32;
        (cx * s, cy * s, w * s, h * s)
    } else {
        (cx, cy, w, h)
    };
    let quad_lb = obb_to_quad_letterbox(cx, cy, w, h, theta);
    Ok(Some((quad_lb, score)))
}

/// Parse standard detection output [1, 6, N] or [1, 7, N]. No mask. Returns best bbox in letterbox space.
fn parse_detection(data: &[f32], shape: &[usize]) -> Result<Option<BestDetection>, OrtError> {
    if shape.len() != 3 {
        return Ok(None);
    }
    let (_, dim, n) = (shape[0], shape[1], shape[2]);
    if dim != 6 && dim != 7 {
        return Ok(None);
    }
    if n == 0 {
        return Ok(None);
    }
    let class_end = dim;
    let mut candidates: Vec<(f32, f32, f32, f32, f32)> = Vec::new();
    for i in 0..n {
        let cx = data.get(i).copied().unwrap_or(0.0);
        let cy = data.get(n + i).copied().unwrap_or(0.0);
        let w = data.get(2 * n + i).copied().unwrap_or(0.0);
        let h = data.get(3 * n + i).copied().unwrap_or(0.0);
        let obj_raw = data.get(4 * n + i).copied().unwrap_or(0.0);
        let obj_conf = if (0.0..=1.0).contains(&obj_raw) {
            obj_raw
        } else {
            sigmoid(obj_raw)
        };
        let max_class = if class_end > 5 {
            (5..class_end)
                .map(|c| data.get(c * n + i).copied().unwrap_or(0.0))
                .map(|v| if (0.0..=1.0).contains(&v) { v } else { sigmoid(v) })
                .fold(0.0f32, f32::max)
        } else {
            1.0
        };
        let score = obj_conf * max_class;
        if score < SCORE_THRESHOLD {
            continue;
        }
        let half_w = w / 2.0;
        let half_h = h / 2.0;
        let mut x1 = cx - half_w;
        let mut y1 = cy - half_h;
        let mut x2 = cx + half_w;
        let mut y2 = cy + half_h;
        let max_coord = [cx, cy, w, h].iter().copied().fold(0.0f32, f32::max);
        if max_coord <= 1.0 {
            let s = MODEL_INPUT_SIZE as f32;
            x1 *= s;
            y1 *= s;
            x2 *= s;
            y2 *= s;
        }
        candidates.push((x1, y1, x2, y2, score));
    }
    candidates.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
    for (idx, c) in candidates.iter().take(3).enumerate() {
        eprintln!(
            "[detect] top#{} score={:.3} letterbox_xyxy=({:.1},{:.1},{:.1},{:.1})",
            idx + 1, c.4, c.0, c.1, c.2, c.3
        );
    }
    Ok(candidates.into_iter().next().map(|(lb_x1, lb_y1, lb_x2, lb_y2, score)| BestDetection {
        lb_x1,
        lb_y1,
        lb_x2,
        lb_y2,
        score,
        mask_coeffs: vec![],
    }))
}

/// Per detection: [cx, cy, w, h, obj_conf, class_probs..., (mask_coeffs)].
/// Only for segmentation format (dim >= 37). Returns best with bbox in letterbox space and mask coeffs.
fn parse_segmentation(data: &[f32], shape: &[usize]) -> Result<Option<BestDetection>, OrtError> {
    if shape.len() != 3 {
        return Ok(None);
    }
    let (_, dim, n) = (shape[0], shape[1], shape[2]);
    if dim < 37 {
        return Ok(None);
    }

    let take = 20.min(dim);
    for (row_idx, &i) in [0, 1].iter().enumerate() {
        if i >= n {
            break;
        }
        let row: Vec<f32> = (0..take).map(|c| data.get(c * n + i).copied().unwrap_or(0.0)).collect();
        eprintln!("[detect] raw row {} (detection index {}): {:?}", row_idx, i, row);
    }

    // Format: 4 bbox + 1 obj + [class_probs] + [32 mask_coeffs]. So class_end = dim - 32 when dim >= 37.
    let num_mask_coeffs = 32;
    let class_end = dim.saturating_sub(num_mask_coeffs).max(6);
    let mut candidates: Vec<(f32, f32, f32, f32, f32, Vec<f32>)> = Vec::new();
    for i in 0..n {
        let cx = data.get(i).copied().unwrap_or(0.0);
        let cy = data.get(n + i).copied().unwrap_or(0.0);
        let w = data.get(2 * n + i).copied().unwrap_or(0.0);
        let h = data.get(3 * n + i).copied().unwrap_or(0.0);
        let obj_raw = data.get(4 * n + i).copied().unwrap_or(0.0);
        let obj_conf = if (0.0..=1.0).contains(&obj_raw) {
            obj_raw
        } else {
            sigmoid(obj_raw)
        };
        let max_class = if class_end > 5 {
            (5..class_end)
                .map(|c| data.get(c * n + i).copied().unwrap_or(0.0))
                .map(|v| if (0.0..=1.0).contains(&v) { v } else { sigmoid(v) })
                .fold(0.0f32, f32::max)
        } else {
            1.0
        };
        let score = obj_conf * max_class;
        let half_w = w / 2.0;
        let half_h = h / 2.0;
        let mut x1 = cx - half_w;
        let mut y1 = cy - half_h;
        let mut x2 = cx + half_w;
        let mut y2 = cy + half_h;
        let max_coord = [cx, cy, w, h].iter().copied().fold(0.0f32, f32::max);
        if max_coord <= 1.0 {
            let s = MODEL_INPUT_SIZE as f32;
            x1 *= s;
            y1 *= s;
            x2 *= s;
            y2 *= s;
        }
        let coeffs: Vec<f32> = if num_mask_coeffs > 0 {
            (5..5 + num_mask_coeffs)
                .map(|c| data.get(c * n + i).copied().unwrap_or(0.0))
                .collect()
        } else {
            vec![]
        };
        candidates.push((x1, y1, x2, y2, score, coeffs));
    }

    candidates.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
    for (idx, c) in candidates.iter().take(3).enumerate() {
        eprintln!(
            "[detect] top#{} score={:.3} letterbox_xyxy=({:.1},{:.1},{:.1},{:.1})",
            idx + 1,
            c.4,
            c.0,
            c.1,
            c.2,
            c.3
        );
    }

    Ok(candidates
        .into_iter()
        .find(|c| c.4 >= SCORE_THRESHOLD)
        .map(|(lb_x1, lb_y1, lb_x2, lb_y2, score, mask_coeffs)| BestDetection {
            lb_x1,
            lb_y1,
            lb_x2,
            lb_y2,
            score,
            mask_coeffs,
        }))
}
