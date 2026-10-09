//! Typestate pipeline with a Writer-style trace. See `docs/design.md` section 5.

pub mod report;
pub mod trace;

use std::{future::Future, marker::PhantomData, time::Instant};

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::{
    domain::{CategoryKind, Question, RunId, Score},
    llm::ModelRef,
};

use self::trace::{StepOutcome, Trace, TraceEntry, TraceRecord};

pub trait Category: Send + Sync + 'static {
    /// `None` before classification.
    const KIND: Option<CategoryKind>;
}

pub struct Unclassified;
pub struct Boolean;
pub struct Choice;
pub struct Ranking;
pub struct Open;

impl Category for Unclassified {
    const KIND: Option<CategoryKind> = None;
}

macro_rules! category {
    ($($marker:ident),*) => {
        $(impl Category for $marker {
            const KIND: Option<CategoryKind> = Some(CategoryKind::$marker);
        })*
    };
}

category!(Boolean, Choice, Ranking, Open);

/// Initial state of every run: only the question is known.
#[derive(Clone, Copy, Debug)]
pub struct Asked;

pub struct Run<TCategory, TState> {
    id: RunId,
    question: Question,
    state: TState,
    trace: Trace,
    category: PhantomData<TCategory>,
}

impl Run<Unclassified, Asked> {
    pub fn new(question: Question) -> Self {
        Self {
            id: RunId::new(),
            question,
            state: Asked,
            trace: Trace::default(),
            category: PhantomData,
        }
    }
}

/// A step is generic over the categories it supports (`Answer<Boolean>` is a step, `Answer<Open>` is not),
/// rather than `Step` being generic over the category: that keeps the blanket `Pipeline` impl coherent with `Seq`.
pub trait Step: Send + Sync {
    const NAME: &'static str;
    type Category: Category;
    type Input: Send;
    type Output: Send;
    type Record: Into<TraceRecord> + Send;

    /// Minimum score to continue. A step with a threshold must return a score.
    fn threshold(&self) -> Option<Score> {
        None
    }

    fn run(
        &self,
        question: &Question,
        input: Self::Input,
    ) -> impl Future<Output = Result<Scored<Self::Output, Self::Record>, StepError>> + Send;
}

pub struct Scored<TOutput, TRecord> {
    pub output: TOutput,
    pub record: TRecord,
    pub score: Option<Score>,
    pub model: Option<ModelRef>,
    pub prompt_version: Option<&'static str>,
}

impl<TOutput, TRecord> Scored<TOutput, TRecord> {
    pub fn new(output: TOutput, record: TRecord) -> Self {
        Self {
            output,
            record,
            score: None,
            model: None,
            prompt_version: None,
        }
    }

    pub fn with_score(self, score: Score) -> Self {
        Self {
            score: Some(score),
            ..self
        }
    }

    pub fn with_model(self, model: ModelRef, prompt_version: &'static str) -> Self {
        Self {
            model: Some(model),
            prompt_version: Some(prompt_version),
            ..self
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum DeclineReason {
    LowConfidence,
    /// Model output failed DTO deserialization or `TryFrom` conversion.
    InvalidModelOutput(String),
    NoSources,
    Unverified(String),
}

/// A step that cannot produce its output. The record, when present, keeps the raw model output in the trace.
#[derive(Debug)]
pub struct StepError {
    pub kind: StepErrorKind,
    pub record: Option<TraceRecord>,
}

#[derive(Debug)]
pub enum StepErrorKind {
    Decline(DeclineReason),
    Failed(anyhow::Error),
}

impl StepError {
    pub fn decline(reason: DeclineReason) -> Self {
        Self {
            kind: StepErrorKind::Decline(reason),
            record: None,
        }
    }

    pub fn with_record(self, record: impl Into<TraceRecord>) -> Self {
        Self {
            record: Some(record.into()),
            ..self
        }
    }
}

impl<E: Into<anyhow::Error>> From<E> for StepError {
    fn from(error: E) -> Self {
        Self {
            kind: StepErrorKind::Failed(error.into()),
            record: None,
        }
    }
}

#[derive(Debug)]
pub struct Halted {
    pub id: RunId,
    pub question: Question,
    pub trace: Trace,
    /// `None` when the run halted before classification.
    pub category: Option<CategoryKind>,
    pub at: &'static str,
    pub halt: Halt,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Halt {
    Declined {
        reason: DeclineReason,
        score: Option<Score>,
        threshold: Option<Score>,
    },
    Failed {
        error: String,
    },
}

impl<TCategory: Category, TState: Send> Run<TCategory, TState> {
    pub fn id(&self) -> RunId {
        self.id
    }

    pub fn question(&self) -> &Question {
        &self.question
    }

    pub fn state(&self) -> &TState {
        &self.state
    }

    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    pub fn into_parts(self) -> (RunId, Question, TState, Trace) {
        (self.id, self.question, self.state, self.trace)
    }

    /// Times the step, appends its trace entry, checks its threshold, and continues or halts.
    pub async fn then<TStep>(self, step: &TStep) -> Result<Run<TCategory, TStep::Output>, Halted>
    where
        TStep: Step<Category = TCategory, Input = TState>,
    {
        let Self {
            id,
            question,
            state,
            mut trace,
            ..
        } = self;
        let (entry, result) = execute(step, &question, state, None).await;
        trace.push(entry);
        match result {
            Ok(output) => Ok(Run {
                id,
                question,
                state: output,
                trace,
                category: PhantomData,
            }),
            Err(halt) => Err(Halted {
                id,
                question,
                trace,
                category: TCategory::KIND,
                at: TStep::NAME,
                halt,
            }),
        }
    }

    /// Runs two steps concurrently on the same state. Both entries are traced even when one halts.
    pub async fn both<TLeft, TRight>(
        self,
        group: &'static str,
        left: &TLeft,
        right: &TRight,
    ) -> Result<Run<TCategory, (TLeft::Output, TRight::Output)>, Halted>
    where
        TState: Clone,
        TLeft: Step<Category = TCategory, Input = TState>,
        TRight: Step<Category = TCategory, Input = TState>,
    {
        let Self {
            id,
            question,
            state,
            mut trace,
            ..
        } = self;
        let ((left_entry, left_result), (right_entry, right_result)) = futures::join!(
            execute(left, &question, state.clone(), Some(group)),
            execute(right, &question, state, Some(group)),
        );
        trace.push(left_entry);
        trace.push(right_entry);
        let halted = |trace, at, halt| Halted {
            id,
            question: question.clone(),
            trace,
            category: TCategory::KIND,
            at,
            halt,
        };
        match (left_result, right_result) {
            (Ok(left), Ok(right)) => Ok(Run {
                id,
                question: question.clone(),
                state: (left, right),
                trace,
                category: PhantomData,
            }),
            (Err(halt), _) => Err(halted(trace, TLeft::NAME, halt)),
            (_, Err(halt)) => Err(halted(trace, TRight::NAME, halt)),
        }
    }

    /// Restructures the state without a step. Not traced, so only for plumbing between steps.
    #[expect(dead_code, reason = "first caller is the classification join")]
    pub(crate) fn map<TNext: Send>(self, f: impl FnOnce(TState) -> TNext) -> Run<TCategory, TNext> {
        Run {
            id: self.id,
            question: self.question,
            state: f(self.state),
            trace: self.trace,
            category: PhantomData,
        }
    }

    /// Re-tags the run with the category decided by `Classify`. Only the classification join may call this.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "first caller is the classification join")
    )]
    pub(crate) fn classified<TNext: Category>(self) -> Run<TNext, TState> {
        Run {
            id: self.id,
            question: self.question,
            state: self.state,
            trace: self.trace,
            category: PhantomData,
        }
    }
}

async fn execute<TStep: Step>(
    step: &TStep,
    question: &Question,
    input: TStep::Input,
    group: Option<&'static str>,
) -> (TraceEntry, Result<TStep::Output, Halt>) {
    let started_at = Utc::now();
    let started = Instant::now();
    let result = step.run(question, input).await;
    let duration_ms = started.elapsed().as_millis() as u64;
    let threshold = step.threshold();

    let entry = |outcome, record, model, prompt_version: Option<&'static str>| TraceEntry {
        step: TStep::NAME.to_owned(),
        group: group.map(str::to_owned),
        started_at,
        duration_ms,
        model,
        prompt_version: prompt_version.map(str::to_owned),
        outcome,
        record,
    };

    match result {
        Ok(scored) => {
            let halt = match (scored.score, threshold) {
                (Some(score), Some(threshold)) if score < threshold => Some(Halt::Declined {
                    reason: DeclineReason::LowConfidence,
                    score: Some(score),
                    threshold: Some(threshold),
                }),
                (None, Some(_)) => Some(Halt::Failed {
                    error: format!("{} has a threshold but returned no score", TStep::NAME),
                }),
                _ => None,
            };
            let outcome = match &halt {
                None => StepOutcome::Passed {
                    score: scored.score,
                    threshold,
                },
                Some(halt) => outcome_of(halt),
            };
            let entry = entry(
                outcome,
                Some(scored.record.into()),
                scored.model,
                scored.prompt_version,
            );
            match halt {
                None => (entry, Ok(scored.output)),
                Some(halt) => (entry, Err(halt)),
            }
        }
        Err(error) => {
            let halt = match error.kind {
                StepErrorKind::Decline(reason) => Halt::Declined {
                    reason,
                    score: None,
                    threshold,
                },
                StepErrorKind::Failed(error) => Halt::Failed {
                    error: format!("{error:#}"),
                },
            };
            (
                entry(outcome_of(&halt), error.record, None, None),
                Err(halt),
            )
        }
    }
}

fn outcome_of(halt: &Halt) -> StepOutcome {
    match halt.clone() {
        Halt::Declined {
            reason,
            score,
            threshold,
        } => StepOutcome::Declined {
            reason,
            score,
            threshold,
        },
        Halt::Failed { error } => StepOutcome::Failed { error },
    }
}

/// A typed path through the state machine. Every `Step` is a one-transition pipeline.
pub trait Pipeline: Send + Sync {
    type Category: Category;
    type Input: Send;
    type Output: Send;

    fn drive(
        &self,
        run: Run<Self::Category, Self::Input>,
    ) -> impl Future<Output = Result<Run<Self::Category, Self::Output>, Halted>> + Send;
}

impl<TStep: Step> Pipeline for TStep {
    type Category = TStep::Category;
    type Input = TStep::Input;
    type Output = TStep::Output;

    async fn drive(
        &self,
        run: Run<TStep::Category, TStep::Input>,
    ) -> Result<Run<TStep::Category, TStep::Output>, Halted> {
        run.then(self).await
    }
}

/// Composes two pipelines; only a valid pipeline when the first one's output is the second one's input.
///
/// ```compile_fail
/// use drew::{domain::Question, pipeline::{trace::VerifyRecord, *}};
///
/// struct A;
/// struct B;
/// struct ToA;
///
/// impl Step for ToA {
///     const NAME: &'static str = "to_a";
///     type Category = Open;
///     type Input = B;
///     type Output = A;
///     type Record = VerifyRecord;
///
///     async fn run(&self, _: &Question, _: B) -> Result<Scored<A, VerifyRecord>, StepError> {
///         unimplemented!()
///     }
/// }
///
/// async fn twice(run: Run<Open, B>) {
///     // ToA outputs A, but the second ToA needs B.
///     let _ = Seq(ToA, ToA).drive(run).await;
/// }
/// ```
pub struct Seq<TFirst, TRest>(pub TFirst, pub TRest);

impl<TFirst, TRest> Pipeline for Seq<TFirst, TRest>
where
    TFirst: Pipeline,
    TRest: Pipeline<Category = TFirst::Category, Input = TFirst::Output>,
{
    type Category = TFirst::Category;
    type Input = TFirst::Input;
    type Output = TRest::Output;

    async fn drive(
        &self,
        run: Run<TFirst::Category, TFirst::Input>,
    ) -> Result<Run<TFirst::Category, TRest::Output>, Halted> {
        self.1.drive(self.0.drive(run).await?).await
    }
}

#[cfg(test)]
mod tests;
