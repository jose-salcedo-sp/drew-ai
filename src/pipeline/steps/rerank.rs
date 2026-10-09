use std::marker::PhantomData;

use crate::{
    domain::Question,
    pipeline::{Category, Scored, Step, StepError, trace::RerankRecord},
};

use super::{Candidates, Reranked, ranked};

/// Milestone 3 stub: the top `top_k` candidates by fused score. The reranking model replaces it in milestone 5.
pub struct Rerank<TCategory> {
    top_k: usize,
    category: PhantomData<TCategory>,
}

impl<TCategory> Rerank<TCategory> {
    pub fn new(top_k: usize) -> Self {
        Self {
            top_k,
            category: PhantomData,
        }
    }
}

impl<TCategory> Default for Rerank<TCategory> {
    fn default() -> Self {
        Self::new(5)
    }
}

impl<TCategory: Category> Step for Rerank<TCategory> {
    const NAME: &'static str = "rerank";
    type Category = TCategory;
    type Input = Candidates;
    type Output = Reranked;
    type Record = RerankRecord;

    async fn run(
        &self,
        _question: &Question,
        input: Candidates,
    ) -> Result<Scored<Reranked, RerankRecord>, StepError> {
        let mut hits = input.0;
        hits.sort_by(|a, b| b.score.0.total_cmp(&a.score.0));
        hits.truncate(self.top_k);
        let record = RerankRecord {
            raw: None,
            chunks: ranked(&hits),
        };
        Ok(Scored::new(Reranked(hits), record))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domain::Score,
        pipeline::{Open, steps::fixtures::chunk},
        retrieval::Retrieved,
    };

    #[tokio::test]
    async fn keeps_the_top_five_by_fused_score() {
        let hits = [0.1, 0.7, 0.3, 0.9, 0.5, 0.2, 0.8]
            .into_iter()
            .enumerate()
            .map(|(i, score)| Retrieved {
                chunk: chunk(&format!("c{i}"), "Roles", "text"),
                score: Score(score),
            })
            .collect();

        let scored = Rerank::<Open>::default()
            .run(&Question::new("q"), Candidates(hits))
            .await
            .unwrap();

        let scores: Vec<_> = scored.output.0.iter().map(|hit| hit.score.0).collect();
        assert_eq!(scores, [0.9, 0.8, 0.7, 0.5, 0.3]);
        assert_eq!(scored.record.chunks.len(), 5);
        assert_eq!(
            scored.record.chunks[0].chunk,
            scored.output.0[0].chunk.reference()
        );
    }
}
