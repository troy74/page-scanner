//! Load optional config from ~/.scanner/config.toml; CLI overrides config.

use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "kebab-case", default)]
pub struct Config {
    pub default_dpi: Option<u32>,
    pub default_cleanimg: Option<String>,
    pub output_dir: Option<PathBuf>,
    pub openai_model: Option<String>,
}

impl Config {
    /// Load config from ~/.scanner/config.toml if it exists.
    pub fn load() -> Self {
        let path = Self::config_path();
        if path.exists() {
            if let Ok(s) = std::fs::read_to_string(&path) {
                if let Ok(c) = toml::from_str(&s) {
                    return c;
                }
            }
        }
        Config::default()
    }

    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .map(|h| h.join(".scanner").join("config.toml"))
            .unwrap_or_else(|| PathBuf::from(".scanner/config.toml"))
    }

    pub fn models_dir() -> PathBuf {
        dirs::home_dir()
            .map(|h| h.join(".scanner").join("models"))
            .unwrap_or_else(|| PathBuf::from(".scanner/models"))
    }

    /// Default model path: models/seg-model.onnx relative to current directory.
    pub fn default_model_path() -> PathBuf {
        PathBuf::from("models/seg-model.onnx")
    }
}
