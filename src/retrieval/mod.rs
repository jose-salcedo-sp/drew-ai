//! Hybrid retrieval: full-text + vector over current chunk versions, merged with reciprocal rank fusion.

pub mod fakes;
pub mod hybrid;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::domain::{Chunk, ChunkId, Score};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Retrieved {
    pub chunk: Chunk,
    /// Fused score; higher is better.
    pub score: Score,
    /// Set when the chunk is not a match itself but context next to one: the id of that match.
    #[serde(default)]
    pub context_for: Option<ChunkId>,
}

#[async_trait]
pub trait Retriever: Send + Sync {
    /// Top hybrid matches, expanded to their parent unit and deduplicated, best first.
    async fn retrieve(&self, query: &str) -> anyhow::Result<Vec<Retrieved>>;
}
