//! OpenAI chat-completions wire types, plus the `drew` extension object.

use serde::{Deserialize, Serialize};

use crate::{
    domain::{CategoryKind, Question, RunId, Score},
    pipeline::{
        DeclineReason,
        report::{Citation, Outcome, RunReport, Verdict},
        trace::{StepOutcome, Trace},
    },
    store::runs::status,
};

use super::error::ApiError;

pub const MODEL: &str = "drew-ai";

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: Option<Content>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<Part>),
}

#[derive(Debug, Deserialize)]
pub struct Part {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

impl ChatRequest {
    /// The text of the last `user` message. System messages and earlier history are ignored.
    pub fn question(&self) -> Result<Question, ApiError> {
        let message = self
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .ok_or_else(|| ApiError::bad_request("No user message in `messages`."))?;
        let text = match &message.content {
            None => String::new(),
            Some(Content::Text(text)) => text.clone(),
            Some(Content::Parts(parts)) => parts
                .iter()
                .map(|part| match (part.kind.as_str(), &part.text) {
                    ("text", Some(text)) => Ok(text.as_str()),
                    _ => Err(ApiError::bad_request(format!(
                        "Unsupported content part `{}`; only text parts are accepted.",
                        part.kind
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()?
                .join("\n"),
        };
        if text.trim().is_empty() {
            return Err(ApiError::bad_request("The last user message is empty."));
        }
        Ok(Question::new(text))
    }
}

/// Category, scores, outcome, citations, and the full trace of one run.
#[derive(Debug, Serialize)]
pub struct Drew {
    pub run_id: RunId,
    pub category: Option<CategoryKind>,
    /// `answered`, `declined`, or `failed`.
    pub outcome: &'static str,
    /// The step that declined or failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub halted_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decline_reason: Option<DeclineReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verdict: Option<Verdict>,
    pub scores: Vec<StepScore>,
    /// Numbered as in the content: the first is `[1]`. For a decline, the ranked sources.
    pub citations: Vec<Citation>,
    pub trace: Trace,
}

#[derive(Debug, Serialize)]
pub struct StepScore {
    pub step: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<String>,
    pub status: &'static str,
    pub score: Option<Score>,
    pub threshold: Option<Score>,
}

impl Drew {
    pub fn new(report: &RunReport) -> Self {
        let (halted_at, decline_reason, verdict, citations) = match &report.outcome {
            Outcome::Answered(answer) => {
                (None, None, answer.verdict.clone(), answer.citations.clone())
            }
            Outcome::Declined {
                at,
                reason,
                sources,
                ..
            } => (
                Some(at.clone()),
                Some(reason.clone()),
                None,
                sources.clone(),
            ),
            Outcome::Failed { at, .. } => (Some(at.clone()), None, None, vec![]),
        };
        let scores = report
            .trace
            .entries()
            .iter()
            .map(|entry| {
                let (status, score, threshold) = match &entry.outcome {
                    StepOutcome::Passed { score, threshold } => ("passed", *score, *threshold),
                    StepOutcome::Declined {
                        score, threshold, ..
                    } => ("declined", *score, *threshold),
                    StepOutcome::Failed { .. } => ("failed", None, None),
                };
                StepScore {
                    step: entry.step.clone(),
                    group: entry.group.clone(),
                    status,
                    score,
                    threshold,
                }
            })
            .collect();
        Self {
            run_id: report.id,
            category: report.category,
            outcome: status(&report.outcome),
            halted_at,
            decline_reason,
            verdict,
            scores,
            citations,
            trace: report.trace.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Completion {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: &'static str,
    pub choices: [CompletionChoice; 1],
    pub drew: Drew,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: String,
}

impl Completion {
    pub fn new(drew: Drew, content: String, created: i64) -> Self {
        Self {
            id: completion_id(drew.run_id),
            object: "chat.completion",
            created,
            model: MODEL,
            choices: [CompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant",
                    content,
                },
                finish_reason: "stop",
            }],
            drew,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Chunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: &'static str,
    pub choices: [ChunkChoice; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drew: Option<Drew>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Default, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

impl Chunk {
    /// The whole verified answer in one content chunk, then a final chunk carrying `finish_reason` and `drew`.
    pub fn pair(drew: Drew, content: String, created: i64) -> [Self; 2] {
        let id = completion_id(drew.run_id);
        let make = |delta, finish_reason, drew| Self {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: MODEL,
            choices: [ChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
            drew,
        };
        [
            make(
                Delta {
                    role: Some("assistant"),
                    content: Some(content),
                },
                None,
                None,
            ),
            make(Delta::default(), Some("stop"), Some(drew)),
        ]
    }
}

fn completion_id(run_id: RunId) -> String {
    format!("chatcmpl-{run_id}")
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: [ModelCard; 1],
}

#[derive(Debug, Serialize)]
pub struct ModelCard {
    pub id: &'static str,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: &'static str,
}

impl ModelList {
    pub fn drew() -> Self {
        Self {
            object: "list",
            data: [ModelCard {
                id: MODEL,
                object: "model",
                created: 0,
                owned_by: "drew",
            }],
        }
    }
}
