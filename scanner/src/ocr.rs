//! Optional tesseract OCR via std::process::Command.

use std::path::Path;
use std::process::Command;

/// Run tesseract on image path; return stdout text or None on failure.
/// Equivalent to: tesseract input.png stdout -l eng
pub fn run_tesseract(image_path: &Path) -> Option<String> {
    let output = Command::new("tesseract")
        .arg(image_path)
        .arg("stdout")
        .arg("-l")
        .arg("eng")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}
