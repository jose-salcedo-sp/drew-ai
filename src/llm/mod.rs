//! Model clients: Bedrock embeddings, the Bedrock "why" LLM, and System One (OpenAI-compatible HTTP).

pub mod bedrock_chat;
pub mod bedrock_embed;
pub mod fakes;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Embedding dimension of the `chunks.embedding` column. Titan v2 and Cohere Embed v3 both produce 1024.
pub const EMBEDDING_DIM: usize = 1024;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelRef {
    pub id: String,
    pub version: String,
}

impl ModelRef {
    pub fn new(id: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: version.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedPurpose {
    Document,
    Query,
}

#[async_trait]
pub trait Embedder: Send + Sync {
    fn model(&self) -> ModelRef;

    /// One vector of `EMBEDDING_DIM` per input, in input order.
    async fn embed(&self, texts: &[String], purpose: EmbedPurpose)
    -> anyhow::Result<Vec<Vec<f32>>>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub system: String,
    pub user: String,
    pub max_tokens: u32,
    pub temperature: f32,
    /// When set, the model must return JSON matching this schema.
    pub json_schema: Option<serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    pub text: String,
    /// Mean token log-probability, when the backend reports it. System One confidence is derived from this.
    pub mean_logprob: Option<f32>,
}

#[async_trait]
pub trait ChatModel: Send + Sync {
    fn model(&self) -> ModelRef;

    async fn complete(&self, request: ChatRequest) -> anyhow::Result<ChatResponse>;
}
