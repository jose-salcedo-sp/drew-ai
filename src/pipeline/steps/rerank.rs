use std::{
    collections::{BTreeMap, HashMap},
    marker::PhantomData,
    sync::Arc,
};

use futures::{StreamExt, TryStreamExt, stream};

use crate::{
    domain::{ChunkId, Question, Score},
    llm::system_one::{Decision, DecisionAnswer, DecisionRequest, SystemOne},
    pipeline::{Category, DeclineReason, Scored, Step, StepError, trace::RerankRecord},
    retrieval::Retrieved,
};

use super::{Candidates, Reranked, ranked};

pub const RERANK_PROMPT_VERSION: &str = "rerank-v1";

const QUESTION_ID: &str = "relevant";
/// Kev's smaller models are validated to 8,192 tokens; a longer section is judged on its beginning.
const MAX_STATE_CHARS: usize = 16_000;
/// Requests in flight at once; the server batches them.
const CONCURRENT_REQUESTS: usize = 8;

/// Picks the `top_k` matches that best answer the question, each followed by the context retrieval attached to it.
///
/// With System One, the best `candidates` matches by fused score are each judged by one yes/no question, and the
/// probability of yes is their score. Without it, matches keep their fused order and are unscored.
pub struct Rerank<TCategory> {
    top_k: usize,
    candidates: usize,
    system_one: Option<Arc<dyn SystemOne>>,
    category: PhantomData<TCategory>,
}

impl<TCategory> Rerank<TCategory> {
    pub fn new(top_k: usize, candidates: usize, system_one: Option<Arc<dyn SystemOne>>) -> Self {
        Self {
            top_k,
            candidates,
            system_one,
            category: PhantomData,
        }
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    pub fn candidates(&self) -> usize {
        self.candidates
    }
}

impl<TCategory> Default for Rerank<TCategory> {
    fn default() -> Self {
        Self::new(5, 25, None)
    }
}

fn state(hit: &Retrieved) -> String {
    let chunk = &hit.chunk;
    let heading = chunk
        .heading
        .as_deref()
        .map(|heading| format!(" > {heading}"))
        .unwrap_or_default();
    let text = match chunk.text.char_indices().nth(MAX_STATE_CHARS) {
        Some((end, _)) => &chunk.text[..end],
        None => &chunk.text,
    };
    format!("{}{heading}\n\n{text}", chunk.title)
}

fn request(question: &Question, hit: &Retrieved) -> DecisionRequest {
    DecisionRequest {
        state: state(hit),
        model: String::new(),
        questions: BTreeMap::from([(
            QUESTION_ID.to_owned(),
            Decision::Noul {
                instructions: format!(
                    "Does this text contain information that helps answer the question: {}",
                    question.as_str()
                ),
            },
        )]),
    }
}

async fn judge(
    system_one: &dyn SystemOne,
    request: DecisionRequest,
    chunk: &ChunkId,
) -> Result<f32, StepError> {
    let response = system_one.decide(request).await?;
    match response.answers.get(QUESTION_ID) {
        Some(DecisionAnswer::Noul { noul }) => Ok(*noul),
        other => Err(StepError::decline(DeclineReason::InvalidModelOutput(
            format!("expected a yes/no answer for {}, got {other:?}", chunk.0),
        ))),
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
        question: &Question,
        input: Candidates,
    ) -> Result<Scored<Reranked, RerankRecord>, StepError> {
        let (mut matches, context): (Vec<Retrieved>, Vec<Retrieved>) = input
            .0
            .into_iter()
            .partition(|hit| hit.context_for.is_none());
        matches.sort_by(|a, b| b.score.0.total_cmp(&a.score.0));

        let mut raw = None;
        let mut model = None;
        if let Some(system_one) = &self.system_one {
            matches.truncate(self.candidates);
            let judged = model.insert(system_one.model()).clone();
            let requests: Vec<_> = matches
                .iter()
                .map(|hit| {
                    let (system_one, request) = (system_one.clone(), request(question, hit));
                    let id = hit.chunk.id.clone();
                    async move { judge(system_one.as_ref(), request, &id).await }
                })
                .collect();
            let probabilities: Vec<f32> = stream::iter(requests)
                .buffered(CONCURRENT_REQUESTS)
                .try_collect()
                .await
                .map_err(|error| error.with_model(judged, RERANK_PROMPT_VERSION))?;
            let by_chunk: BTreeMap<&str, f32> = matches
                .iter()
                .zip(&probabilities)
                .map(|(hit, p)| (hit.chunk.id.0.as_str(), *p))
                .collect();
            raw = serde_json::to_string(&by_chunk).ok();
            for (hit, probability) in matches.iter_mut().zip(probabilities) {
                hit.score = Score(probability);
            }
            matches.sort_by(|a, b| b.score.0.total_cmp(&a.score.0));
        }
        matches.truncate(self.top_k);

        let mut context_of: HashMap<ChunkId, Vec<Retrieved>> = HashMap::new();
        for chunk in context {
            if let Some(hit) = chunk.context_for.clone() {
                context_of.entry(hit).or_default().push(chunk);
            }
        }
        let best = matches.first().map(|hit| hit.score);
        let mut sources = Vec::new();
        let mut kept_context = Vec::new();
        for hit in &matches {
            let around = context_of.remove(&hit.chunk.id).unwrap_or_default();
            sources.push(hit.clone());
            sources.extend(around.iter().cloned());
            kept_context.extend(around);
        }
        // Matches first, so eval's hit@5 reads the top five matches.
        let mut chunks = ranked(&matches);
        chunks.extend(ranked(&kept_context));
        let record = RerankRecord { raw, chunks };
        let scored = Scored::new(Reranked(sources), record);
        Ok(match (model, best) {
            (Some(model), Some(best)) => scored
                .with_score(best)
                .with_model(model, RERANK_PROMPT_VERSION),
            _ => scored,
        })
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use super::*;
    use crate::{
        domain::Chunk,
        llm::{
            ModelRef,
            system_one::{DecisionResponse, SystemOne},
        },
        pipeline::{Open, steps::fixtures::chunk},
    };

    fn hit(locator: &str, text: &str, score: f32) -> Retrieved {
        Retrieved {
            chunk: chunk(locator, "Roles", text),
            score: Score(score),
            context_for: None,
        }
    }

    fn context(of: &Retrieved, locator: &str, score: f32) -> Retrieved {
        Retrieved {
            chunk: chunk(locator, "Roles", "context"),
            score: Score(score),
            context_for: Some(of.chunk.id.clone()),
        }
    }

    fn ids(reranked: &Reranked) -> Vec<&str> {
        reranked
            .0
            .iter()
            .map(|hit| hit.chunk.heading.as_deref().unwrap())
            .collect()
    }

    /// Says yes with the probability written in the state's text, e.g. `p=0.9`.
    struct ByText;

    #[async_trait]
    impl SystemOne for ByText {
        fn model(&self) -> ModelRef {
            ModelRef::new("fake-system-one", "1")
        }

        async fn decide(&self, request: DecisionRequest) -> anyhow::Result<DecisionResponse> {
            let p = request
                .state
                .split("p=")
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|p| p.parse().ok())
                .unwrap_or(0.0);
            Ok(DecisionResponse {
                answers: BTreeMap::from([(QUESTION_ID.into(), DecisionAnswer::Noul { noul: p })]),
            })
        }
    }

    async fn rerank(rerank: Rerank<Open>, hits: Vec<Retrieved>) -> Scored<Reranked, RerankRecord> {
        rerank
            .run(&Question::new("q"), Candidates(hits))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn context_rides_with_its_match_instead_of_taking_a_slot() {
        let first = hit("a", "", 0.9);
        let second = hit("b", "", 0.8);
        let hits = vec![
            context(&first, "a-next", 0.89),
            context(&first, "a-prev", 0.89),
            first.clone(),
            second.clone(),
            hit("c", "", 0.1),
        ];

        let scored = rerank(Rerank::new(2, 25, None), hits).await;

        assert_eq!(ids(&scored.output), ["a", "a-next", "a-prev", "b"]);
        assert_eq!(scored.score, None);
        let recorded: Vec<_> = scored
            .record
            .chunks
            .iter()
            .map(|c| c.chunk.id.0.as_str())
            .collect();
        assert_eq!(
            recorded[..2],
            [first.chunk.id.0.as_str(), second.chunk.id.0.as_str()]
        );
        assert_eq!(recorded.len(), 4);
    }

    #[tokio::test]
    async fn system_one_reorders_matches_and_scores_the_step() {
        let hits = vec![
            hit("fused-best", "p=0.2", 0.9),
            hit("fused-second", "p=0.95", 0.8),
            hit("fused-third", "p=0.6", 0.7),
            hit("beyond-candidates", "p=1.0", 0.1),
        ];

        let scored = rerank(Rerank::new(2, 3, Some(Arc::new(ByText))), hits).await;

        assert_eq!(ids(&scored.output), ["fused-second", "fused-third"]);
        assert_eq!(scored.score, Some(Score(0.95)));
        assert_eq!(scored.prompt_version, Some(RERANK_PROMPT_VERSION));
        let raw: BTreeMap<String, f32> =
            serde_json::from_str(scored.record.raw.as_deref().unwrap()).unwrap();
        assert_eq!(raw.len(), 3);
        assert_eq!(scored.output.0[0].score, Score(0.95));
    }

    #[test]
    fn a_long_section_is_judged_on_its_beginning() {
        let long = Retrieved {
            chunk: Chunk {
                text: "x".repeat(MAX_STATE_CHARS * 2),
                ..chunk("long", "Roles", "")
            },
            score: Score(1.0),
            context_for: None,
        };
        let state = state(&long);
        assert!(state.starts_with("Roles > long\n\n"));
        assert_eq!(state.chars().filter(|c| *c == 'x').count(), MAX_STATE_CHARS);
    }
}
