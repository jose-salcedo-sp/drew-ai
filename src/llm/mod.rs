//! Model clients: embeddings and the "why" LLM (Bedrock, or any OpenAI-compatible server such as Ollama),
//! and System One (Kev).

pub mod bedrock_chat;
pub mod bedrock_embed;
pub mod fakes;
pub mod openai;
pub mod system_one;

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use self::{
    bedrock_chat::{BedrockChat, ChatArgs},
    bedrock_embed::{BedrockEmbedder, EmbedArgs},
    openai::{OpenAiChat, OpenAiEmbedder},
};

/// An OpenAI-compatible server when `--embed-url` is set, Bedrock otherwise.
pub async fn embedder(args: &EmbedArgs) -> eros::Result<Arc<dyn Embedder>> {
    Ok(match &args.embed_url {
        Some(url) => Arc::new(OpenAiEmbedder::new(url, &args.embed_model)),
        None => Arc::new(BedrockEmbedder::from_args(args).await?),
    })
}

/// An OpenAI-compatible server when `--explain-url` is set, Bedrock otherwise.
pub async fn explain_model(args: &ChatArgs) -> Arc<dyn ChatModel> {
    match &args.explain_url {
        Some(url) => Arc::new(OpenAiChat::new(url, &args.explain_model)),
        None => Arc::new(BedrockChat::from_args(args).await),
    }
}

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
    async fn embed(&self, texts: &[String], purpose: EmbedPurpose) -> eros::Result<Vec<Vec<f32>>>;
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
    /// Mean token log-probability, when the backend reports it.
    pub mean_logprob: Option<f32>,
}

#[async_trait]
pub trait ChatModel: Send + Sync {
    fn model(&self) -> ModelRef;

    async fn complete(&self, request: ChatRequest) -> eros::Result<ChatResponse>;
}
