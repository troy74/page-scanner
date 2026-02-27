//! Entry: parse CLI, load config, resolve model path, run pipeline per input.

mod cli;
mod cleanup;
mod config;
mod export;
mod geometry;
mod llm;
mod model;
mod ocr;
mod pipeline;

use std::path::PathBuf;

use clap::Parser;

use cli::Cli;
use config::Config;
use model::Detector;
use pipeline::process_one;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    let config = Config::load();

    let model_path = cli
        .model
        .clone()
        .unwrap_or_else(Config::default_model_path);

    if !model_path.exists() {
        return Err(format!(
            "Model not found: {}\n\
             Place seg-model.onnx in models/ or pass --model /path/to/model.onnx",
            model_path.display()
        ));
    }

    let mut detector = Detector::new(&model_path).map_err(|e| format!("ONNX init: {}", e))?;

    let dpi = config.default_dpi.unwrap_or(300);
    let outdir = cli.outdir.clone();
    let format = cli.format;
    let cleanimg = cli.cleanimg;
    let ocr_mode = cli.ocr;
    let use_llm = cli.llm;

    let inputs = collect_inputs(&cli.input, cli.limit)?;
    if inputs.is_empty() {
        return Err("no valid image inputs (png/jpg)".to_string());
    }

    export::ensure_outdir(&outdir).map_err(|e| format!("outdir: {}", e))?;

    let n_inputs = inputs.len();
    for (i, path) in inputs.into_iter().enumerate() {
        let (outdir_use, base) = if n_inputs == 1 && cli.output.is_some() {
            let o = cli.output.as_ref().unwrap();
            (
                o.parent().unwrap_or(&outdir).to_path_buf(),
                o.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("output")
                    .to_string(),
            )
        } else {
            (outdir.clone(), export::datetimestamp_base(i + 1))
        };
        if let Err(e) = process_one(
            &path,
            &mut detector,
            &outdir_use,
            &base,
            format,
            cleanimg,
            dpi,
            ocr_mode,
            use_llm,
            cli.debug_bbox,
            cli.savemask,
            &config,
        ) {
            eprintln!("warning: {}: {}", path.display(), e);
            // continue batch
        }
    }

    Ok(())
}

/// Collect input paths: single file or folder (png/jpg), up to limit (0 = unlimited).
fn collect_inputs(input: &PathBuf, limit: usize) -> Result<Vec<PathBuf>, String> {
    if input.is_file() {
        let ext = input
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase());
        if matches!(ext.as_deref(), Some("png") | Some("jpg") | Some("jpeg")) {
            return Ok(vec![input.clone()]);
        }
        return Err(format!("unsupported input: {} (use png/jpg)", input.display()));
    }

    if input.is_dir() {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(input)
            .map_err(|e| format!("read dir {}: {}", input.display(), e))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|s| matches!(s.to_lowercase().as_str(), "png" | "jpg" | "jpeg"))
                    .unwrap_or(false)
            })
            .collect();
        paths.sort();
        if limit > 0 {
            paths.truncate(limit);
        }
        return Ok(paths);
    }

    Err(format!("input not found: {}", input.display()))
}
