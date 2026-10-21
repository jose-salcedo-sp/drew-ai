//! Hybrid retrieval: full-text + vector over current chunk versions, merged with reciprocal rank fusion.

pub mod fakes;
pub mod hybrid;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::domain::{Chunk, ChunkId};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Retrieved {
    pub chunk: Chunk,
    pub score: Relevance,
    /// Set when the chunk is not a match itself but context next to one: the id of that match.
    #[serde(default)]
    pub context_for: Option<ChunkId>,
}

/// Fused retrieval score; higher is better. Comparable only within one retrieval, and not a probability.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Relevance(pub f32);

#[async_trait]
pub trait Retriever: Send + Sync {
    /// Top hybrid matches, expanded to their parent unit and deduplicated, best first.
    async fn retrieve(&self, query: &str) -> eros::Result<Vec<Retrieved>>;
}
