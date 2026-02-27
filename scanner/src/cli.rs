//! CLI definitions: clap with --limit, --format, --cleanimg, --ocr, --llm, --model, etc.

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "page-scanner")]
#[command(about = "Document page scanner: detect page, warp to A4, export PDF/image")]
pub struct Cli {
    /// Input: single image (png/jpg) or folder containing images
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Output path (for single image); overrides --outdir for that file
    #[arg(short, long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Output format: pdf, img, or both
    #[arg(long, value_name = "FORMAT", default_value = "pdf")]
    pub format: OutputFormat,

    /// Output directory (default: ./output/)
    #[arg(long, value_name = "PATH", default_value = "output")]
    pub outdir: PathBuf,

    /// Max images to process when input is a folder (0 = unlimited)
    #[arg(long, value_name = "N", default_value = "10")]
    pub limit: usize,

    /// Path to ONNX model (default: models/seg-model.onnx)
    #[arg(long, value_name = "PATH")]
    pub model: Option<PathBuf>,

    /// Cleanup filter: grayscale (default), original, bw, highcontrast, crisp, sharp
    #[arg(long, value_name = "MODE", default_value = "grayscale")]
    pub cleanimg: CleanImgMode,

    /// OCR: none or tesseract
    #[arg(long, value_name = "MODE", default_value = "none")]
    pub ocr: OcrMode,

    /// Call OpenAI LLM with OCR text (requires llm feature and OPENAI_API_KEY)
    #[arg(long)]
    pub llm: bool,

    /// Write {base}_bbox.png with detected bbox drawn on input (for debugging)
    #[arg(long)]
    pub debug_bbox: bool,

    /// Write {base}_mask.png with detection mask overlaid on original image
    #[arg(long)]
    pub savemask: bool,
}

#[derive(clap::ValueEnum, Clone, Debug, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Pdf,
    Img,
    Both,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "pdf" => Ok(OutputFormat::Pdf),
            "img" | "image" => Ok(OutputFormat::Img),
            "both" => Ok(OutputFormat::Both),
            _ => Err(format!("unknown format: {}", s)),
        }
    }
}

#[derive(clap::ValueEnum, Clone, Debug, Copy, PartialEq, Eq)]
pub enum CleanImgMode {
    /// Legacy default cleanup (contrast + light sharpen)
    Default,
    /// Return the warped image without any cleanup
    Original,
    /// Convert to grayscale (no thresholding)
    Grayscale,
    Bw,
    Highcontrast,
    Crisp,
    Sharp,
}

impl std::str::FromStr for CleanImgMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "default" => Ok(CleanImgMode::Default),
            "original" => Ok(CleanImgMode::Original),
            "grayscale" | "grey" | "gray" => Ok(CleanImgMode::Grayscale),
            "bw" | "blackwhite" => Ok(CleanImgMode::Bw),
            "highcontrast" => Ok(CleanImgMode::Highcontrast),
            "crisp" => Ok(CleanImgMode::Crisp),
            "sharp" => Ok(CleanImgMode::Sharp),
            _ => Err(format!("unknown cleanimg mode: {}", s)),
        }
    }
}

#[derive(clap::ValueEnum, Clone, Debug, Copy, PartialEq, Eq)]
pub enum OcrMode {
    None,
    Tesseract,
}

impl std::str::FromStr for OcrMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(OcrMode::None),
            "tesseract" => Ok(OcrMode::Tesseract),
            _ => Err(format!("unknown ocr mode: {}", s)),
        }
    }
}
