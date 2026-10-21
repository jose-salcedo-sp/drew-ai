use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{
    domain::{CategoryKind, ChunkId, ChunkRef, Probability},
    llm::ModelRef,
    retrieval::Relevance,
};

use super::DeclineReason;

/// Flat list of step entries, ordered by start time. Persisted as JSONB in `pipeline_runs.trace`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Trace(Vec<TraceEntry>);

impl Trace {
    pub fn entries(&self) -> &[TraceEntry] {
        &self.0
    }

    pub(super) fn push(&mut self, entry: TraceEntry) {
        self.0.push(entry);
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceEntry {
    pub step: String,
    /// Set when the step ran concurrently with others, e.g. `classify_retrieve`.
    pub group: Option<String>,
    pub started_at: DateTime<Utc>,
    pub duration_ms: u64,
    pub model: Option<ModelRef>,
    pub prompt_version: Option<String>,
    pub outcome: StepOutcome,
    pub record: Option<TraceRecord>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum StepOutcome {
    Passed {
        score: Option<Probability>,
        threshold: Option<Probability>,
    },
    Declined {
        reason: DeclineReason,
        score: Option<Probability>,
        threshold: Option<Probability>,
    },
    Failed {
        error: String,
    },
}

/// One variant per step. Steps own their record type; the records here are the fields eval relies on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TraceRecord {
    Classify(ClassifyRecord),
    Retrieve(RetrieveRecord),
    Rerank(RerankRecord),
    Answer(AnswerRecord),
    Explain(ExplainRecord),
    Verify(VerifyRecord),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankedChunk {
    pub chunk: ChunkRef,
    /// Retrieval's fused score. Rerank records written before `probability` existed hold System One's
    /// probability here instead.
    pub score: Relevance,
    /// System One's probability that the chunk helps answer the question, when `Rerank` judged it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probability: Option<Probability>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassifyRecord {
    pub raw: Option<String>,
    pub category: Option<CategoryKind>,
    /// What System One chose when its probability was below `classify.min_confidence` and the run fell back to `open`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_from: Option<CategoryKind>,
}

/// Recall@50 is computed from `chunks`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrieveRecord {
    pub chunks: Vec<RankedChunk>,
}

/// Hit@5 is computed from `chunks`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RerankRecord {
    pub raw: Option<String>,
    pub chunks: Vec<RankedChunk>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnswerRecord {
    pub raw: Option<String>,
    pub parsed: Option<serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplainRecord {
    pub raw: Option<String>,
    /// The LLM flagged that cited sources contradict the System One verdict.
    pub contradicts_verdict: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerifyRecord {
    pub uncited_claims: Vec<String>,
    pub unknown_citations: Vec<ChunkId>,
}

macro_rules! into_trace_record {
    ($($record:ident => $variant:ident),* $(,)?) => {
        $(impl From<$record> for TraceRecord {
            fn from(record: $record) -> Self {
                Self::$variant(record)
            }
        })*
    };
}

into_trace_record! {
    ClassifyRecord => Classify,
    RetrieveRecord => Retrieve,
    RerankRecord => Rerank,
    AnswerRecord => Answer,
    ExplainRecord => Explain,
    VerifyRecord => Verify,
}
