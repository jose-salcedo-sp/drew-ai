use std::{
    collections::{BTreeMap, HashMap},
    marker::PhantomData,
    sync::Arc,
};

use futures::{StreamExt, TryStreamExt, stream};

use crate::{
    config::RerankConfig,
    domain::{ChunkId, Question, Score},
    llm::system_one::{Decision, DecisionAnswer, DecisionRequest, SystemOne},
    pipeline::{Category, DeclineReason, Scored, Step, StepError, trace::RerankRecord},
    retrieval::Retrieved,
};

use super::{Candidates, MAX_STATE_CHARS, Reranked, ranked};

pub const RERANK_PROMPT_VERSION: &str = "rerank-v2";

const QUESTION_ID: &str = "relevant";
/// Requests in flight at once; the server batches them.
const CONCURRENT_REQUESTS: usize = 8;
/// Reciprocal rank fusion constant, as in hybrid retrieval.
const RRF_K: f32 = 60.0;

/// Picks the `top_k` matches that best answer the question, each followed by the context retrieval attached to it.
///
/// With System One, the best `candidates` matches by fused score are each judged by one yes/no question. Matches
/// are then ordered by reciprocal rank fusion of their retrieval rank and their probability-of-yes rank, so the
/// judgement refines retrieval instead of replacing it; their score is the probability of yes. Without System One,
/// matches are ordered by retrieval rank alone and keep their fused score. Either way an evidence match is ranked
/// `evidence_rank_penalty` places lower, so Drew's settled answers lead unless the evidence is clearly better.
pub struct Rerank<TCategory> {
    top_k: usize,
    candidates: usize,
    evidence_rank_penalty: usize,
    threshold: Option<Score>,
    system_one: Option<Arc<dyn SystemOne>>,
    category: PhantomData<TCategory>,
}

impl<TCategory> Rerank<TCategory> {
    pub fn new(top_k: usize, candidates: usize, system_one: Option<Arc<dyn SystemOne>>) -> Self {
        Self {
            top_k,
            candidates,
            evidence_rank_penalty: RerankConfig::default().evidence_rank_penalty,
            threshold: None,
            system_one,
            category: PhantomData,
        }
    }

    pub fn from_config(
        config: &RerankConfig,
        threshold: Option<Score>,
        system_one: Option<Arc<dyn SystemOne>>,
    ) -> Self {
        Self {
            evidence_rank_penalty: config.evidence_rank_penalty,
            threshold,
            ..Self::new(config.top_k, config.candidates, system_one)
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
        Self::from_config(&RerankConfig::default(), None, None)
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

/// One list's reciprocal rank fusion term for a match at a 0-based `rank`.
fn rrf(hit: &Retrieved, rank: usize, evidence_rank_penalty: usize) -> f32 {
    let penalty = if hit.chunk.authoritative {
        0
    } else {
        evidence_rank_penalty
    };
    1.0 / (RRF_K + (rank + penalty) as f32)
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

    /// Without System One the step is unscored, so no threshold applies.
    fn threshold(&self) -> Option<Score> {
        self.threshold.filter(|_| self.system_one.is_some())
    }

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

        let penalty = self.evidence_rank_penalty;
        let mut raw = None;
        let mut model = None;
        let mut best = None;
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
            let mut by_probability: Vec<usize> = (0..matches.len()).collect();
            by_probability.sort_by(|&a, &b| probabilities[b].total_cmp(&probabilities[a]));
            let mut judged_rank = vec![0; matches.len()];
            for (rank, &index) in by_probability.iter().enumerate() {
                judged_rank[index] = rank;
            }
            let mut fused: Vec<(f32, Retrieved)> = matches
                .drain(..)
                .zip(probabilities)
                .enumerate()
                .map(|(retrieval_rank, (mut hit, probability))| {
                    let key = rrf(&hit, retrieval_rank, penalty)
                        + rrf(&hit, judged_rank[retrieval_rank], penalty);
                    hit.score = Score(probability);
                    (key, hit)
                })
                .collect();
            fused.sort_by(|a, b| b.0.total_cmp(&a.0));
            matches = fused.into_iter().map(|(_, hit)| hit).collect();
            matches.truncate(self.top_k);
            best = matches
                .iter()
                .map(|hit| hit.score)
                .max_by(|a, b| a.0.total_cmp(&b.0));
        } else {
            let mut ordered: Vec<(f32, Retrieved)> = matches
                .drain(..)
                .enumerate()
                .map(|(rank, hit)| (rrf(&hit, rank, penalty), hit))
                .collect();
            ordered.sort_by(|a, b| b.0.total_cmp(&a.0));
            matches = ordered.into_iter().map(|(_, hit)| hit).collect();
        }
        matches.truncate(self.top_k);

        let mut context_of: HashMap<ChunkId, Vec<Retrieved>> = HashMap::new();
        for chunk in context {
            if let Some(hit) = chunk.context_for.clone() {
                context_of.entry(hit).or_default().push(chunk);
            }
        }
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

        async fn decide(&self, request: DecisionRequest) -> eros::Result<DecisionResponse> {
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

    fn authoritative(hit: Retrieved) -> Retrieved {
        Retrieved {
            chunk: Chunk {
                authoritative: true,
                ..hit.chunk
            },
            ..hit
        }
    }

    #[tokio::test]
    async fn system_one_refines_retrieval_order_and_scores_the_step() {
        let hits = vec![
            hit("fused-best", "p=0.2", 0.9),
            hit("fused-second", "p=0.95", 0.8),
            hit("fused-third", "p=0.6", 0.7),
            hit("beyond-candidates", "p=1.0", 0.1),
        ];

        let scored = rerank(Rerank::new(2, 3, Some(Arc::new(ByText))), hits).await;

        // Judged ranks second, third, best; fused with retrieval ranks best, second, third.
        assert_eq!(ids(&scored.output), ["fused-second", "fused-best"]);
        assert_eq!(scored.score, Some(Score(0.95)));
        assert_eq!(scored.prompt_version, Some(RERANK_PROMPT_VERSION));
        let raw: BTreeMap<String, f32> =
            serde_json::from_str(scored.record.raw.as_deref().unwrap()).unwrap();
        assert_eq!(raw.len(), 3);
        assert_eq!(scored.output.0[0].score, Score(0.95));
        assert_eq!(scored.output.0[1].score, Score(0.2));
    }

    #[tokio::test]
    async fn authoritative_matches_lead_unless_evidence_is_clearly_better() {
        let with_system_one = || Some(Arc::new(ByText) as Arc<dyn SystemOne>);
        let close = vec![
            hit("evidence", "p=0.6", 0.9),
            authoritative(hit("settled", "p=0.5", 0.8)),
        ];
        let scored = rerank(Rerank::new(1, 25, with_system_one()), close.clone()).await;
        assert_eq!(ids(&scored.output), ["settled"]);
        let scored = rerank(Rerank::new(1, 25, None), close).await;
        assert_eq!(ids(&scored.output), ["settled"]);

        let mut far = vec![hit("evidence", "p=0.99", 0.9)];
        far.extend((0..40).map(|i| hit(&format!("filler-{i}"), "p=0.0", 0.5)));
        far.push(authoritative(hit("settled", "p=0.01", 0.1)));
        let scored = rerank(Rerank::new(1, 50, with_system_one()), far).await;
        assert_eq!(ids(&scored.output), ["evidence"]);
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
