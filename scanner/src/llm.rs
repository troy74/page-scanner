//! Optional OpenAI LLM call (feature-gated); reqwest + gpt-5-mini, timeout, JSON out.

#[cfg(feature = "llm")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "llm")]
const DEFAULT_MODEL: &str = "gpt-5-mini";
#[cfg(feature = "llm")]
const TIMEOUT_SECS: u64 = 30;

#[cfg(feature = "llm")]
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
}

#[cfg(feature = "llm")]
#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[cfg(feature = "llm")]
#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Option<Vec<Choice>>,
}

#[cfg(feature = "llm")]
#[derive(Debug, Deserialize)]
struct Choice {
    message: Option<Message>,
}

#[cfg(feature = "llm")]
#[derive(Debug, Deserialize)]
struct Message {
    content: Option<String>,
}

/// Call OpenAI chat/completions with the given text; return JSON-serializable result or None.
#[cfg(feature = "llm")]
pub fn call_openai(text: &str, model: Option<&str>) -> Option<serde_json::Value> {
    let api_key = std::env::var("OPENAI_API_KEY").ok()?;
    let model = model.unwrap_or(DEFAULT_MODEL).to_string();
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(TIMEOUT_SECS))
        .build()
        .ok()?;
    let body = ChatRequest {
        model: model.clone(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: text.to_string(),
        }],
    };
    let res = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .ok()?;
    let status = res.status();
    let json: serde_json::Value = res.json().ok()?;
    if !status.is_success() {
        return None;
    }
    Some(json)
}

#[cfg(not(feature = "llm"))]
pub fn call_openai(_text: &str, _model: Option<&str>) -> Option<serde_json::Value> {
    None
}
