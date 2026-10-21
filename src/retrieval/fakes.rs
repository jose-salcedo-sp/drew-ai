use std::collections::HashSet;

use async_trait::async_trait;

use crate::domain::Chunk;

use super::{Relevance, Retrieved, Retriever};

/// Ranks chunks by the fraction of query words they contain. Chunks with no shared word are dropped.
pub struct KeywordRetriever {
    pub chunks: Vec<Chunk>,
}

#[async_trait]
impl Retriever for KeywordRetriever {
    async fn retrieve(&self, query: &str) -> eros::Result<Vec<Retrieved>> {
        let query = words(query);
        if query.is_empty() {
            return Ok(Vec::new());
        }
        let mut hits: Vec<_> = self
            .chunks
            .iter()
            .filter_map(|chunk| {
                let shared = query.intersection(&words(&chunk.text)).count();
                (shared > 0).then(|| Retrieved {
                    chunk: chunk.clone(),
                    score: Relevance(shared as f32 / query.len() as f32),
                    context_for: None,
                })
            })
            .collect();
        hits.sort_by(|a, b| b.score.0.total_cmp(&a.score.0));
        Ok(hits)
    }
}

fn words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}
