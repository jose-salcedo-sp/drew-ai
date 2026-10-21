//! Model clients behind ports: [`Embedder`] for embeddings and [`ChatModel`] for the "why" LLM (Bedrock, or any
//! OpenAI-compatible server such as Ollama), and [`system_one::SystemOne`] for the "what" (Kev). This module owns
//! wire formats, timeouts and backend selection; prompts belong to the pipeline steps that send them.

pub mod bedrock_chat;
pub mod bedrock_embed;
#[cfg(any(test, feature = "testing"))]
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
///
/// # Errors
///
/// On Bedrock, when the model is not Titan Text Embeddings v2 or Cohere Embed v3.
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

/// Which model produced an output, recorded on chunks and trace entries. Two refs are the same model only if both
/// fields match, so a change of request parameters that shapes the output is a new `version`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelRef {
    /// Backend model name or id, e.g. `amazon.titan-embed-text-v2:0` or `qwen3:14b`.
    pub id: String,
    /// Set by the adapter: request parameters such as `dimensions=1024;normalize=true`, or the wire protocol.
    pub version: String,
}

impl ModelRef {
    /// See the field docs.
    pub fn new(id: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: version.into(),
        }
    }
}

/// Which side of a search a text is on; asymmetric models such as Cohere Embed v3 embed the two differently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedPurpose {
    /// A chunk being ingested.
    Document,
    /// A question being searched for.
    Query,
}

/// Turns texts into vectors for the `chunks.embedding` column and for query search.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Recorded on every chunk; chunks from different models are never mixed.
    fn model(&self) -> ModelRef;

    /// One vector of [`EMBEDDING_DIM`] per input, in input order.
    ///
    /// # Errors
    ///
    /// When the backend fails or returns the wrong number or size of vectors.
    async fn embed(&self, texts: &[String], purpose: EmbedPurpose) -> eros::Result<Vec<Vec<f32>>>;
}

/// One single-turn completion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Instructions, including the answering rules; never contains ingested text.
    pub system: String,
    /// The question and the delimited sources.
    pub user: String,
    /// Upper bound on generated tokens.
    pub max_tokens: u32,
    /// Sampling temperature; `0.0` for the most deterministic output.
    pub temperature: f32,
    /// When set, the model must return JSON matching this schema.
    pub json_schema: Option<serde_json::Value>,
}

/// A completion's output.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The reply, or the JSON document when the request had a schema.
    pub text: String,
    /// Mean token log-probability, when the backend reports it.
    pub mean_logprob: Option<f32>,
}

/// The "why" LLM used by `Explain`.
#[async_trait]
pub trait ChatModel: Send + Sync {
    /// Recorded in the trace.
    fn model(&self) -> ModelRef;

    /// # Errors
    ///
    /// When the backend fails, times out, or returns no text.
    async fn complete(&self, request: ChatRequest) -> eros::Result<ChatResponse>;
}
