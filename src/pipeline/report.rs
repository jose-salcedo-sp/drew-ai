//! What a finished run looks like to the server, the run store, and eval.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::domain::{CategoryKind, ChunkRef, Cited, Probability, Question, RunId};

use super::{DeclineReason, trace::Trace};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunReport {
    pub id: RunId,
    pub question: Question,
    pub category: Option<CategoryKind>,
    pub outcome: Outcome,
    pub trace: Trace,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Outcome {
    Answered(Answer),
    Declined {
        at: String,
        reason: DeclineReason,
        score: Option<Probability>,
        threshold: Option<Probability>,
        /// Best reranked sources, shown under the fixed decline message.
        sources: Vec<Citation>,
    },
    Failed {
        at: String,
        error: String,
    },
}

/// Built only from a verified run, so every claim and option cites a chunk in `citations` or the prompt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Answer {
    /// `None` for `open`.
    pub verdict: Option<Verdict>,
    pub why: Vec<Cited<String>>,
    /// Numbered in order: the first citation renders as `[1]`.
    pub citations: Vec<Citation>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "category", rename_all = "snake_case")]
pub enum Verdict {
    Boolean {
        yes: bool,
        confidence: Probability,
    },
    Choice {
        selected: Cited<String>,
        rejected: Vec<Rejected>,
    },
    Ranking {
        criterion: String,
        items: Vec<Cited<String>>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rejected {
    pub option: Cited<String>,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Citation {
    pub chunk: ChunkRef,
    pub title: String,
    pub url: String,
    pub excerpt: String,
}

/// Runs the whole pipeline for one question. Never errors: failures are an `Outcome`.
#[async_trait]
pub trait Answerer: Send + Sync {
    async fn answer(&self, question: Question) -> RunReport;
}

/// Returns the same outcome for every question, under a fresh run id.
pub struct CannedAnswerer {
    pub category: Option<CategoryKind>,
    pub outcome: Outcome,
}

#[async_trait]
impl Answerer for CannedAnswerer {
    async fn answer(&self, question: Question) -> RunReport {
        RunReport {
            id: RunId::new(),
            question,
            category: self.category,
            outcome: self.outcome.clone(),
            trace: Trace::default(),
        }
    }
}
