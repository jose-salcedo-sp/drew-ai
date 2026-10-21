//! Re-applies candidate thresholds to the scores recorded in each run's trace, without calling any model.

use std::{
    collections::{BTreeMap, HashMap},
    fmt,
};

use serde::{Deserialize, Serialize};

use crate::{
    domain::{CategoryKind, Probability},
    pipeline::{DeclineReason, report::Outcome, trace::StepOutcome},
};

use super::metrics::{CATEGORIES, EvalRun, Metric, Results, cell, percent, scope_name};

pub const MIN_ANSWERED_PRECISION: f64 = 0.95;
pub const MAX_DECLINE_RATE: f64 = 0.40;

/// One entry of the candidates file (a JSON array).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub name: String,
    /// Step name (as recorded in the trace) to minimum score. A step not listed has no threshold.
    #[serde(default)]
    pub thresholds: BTreeMap<String, Probability>,
    /// Overrides `thresholds` for runs the pipeline classified into that category.
    #[serde(default)]
    pub per_category: HashMap<CategoryKind, BTreeMap<String, Probability>>,
}

impl Candidate {
    fn threshold(&self, category: Option<CategoryKind>, step: &str) -> Option<Probability> {
        category
            .and_then(|category| self.per_category.get(&category))
            .and_then(|thresholds| thresholds.get(step))
            .or_else(|| self.thresholds.get(step))
            .copied()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuneReport {
    pub candidates: Vec<CandidateReport>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateReport {
    pub name: String,
    pub meets_launch_gate: bool,
    pub overall: TuneStats,
    /// Grouped by the golden (expected) category, in `CATEGORIES` order.
    pub per_category: Vec<CategoryTuneStats>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CategoryTuneStats {
    pub category: CategoryKind,
    #[serde(flatten)]
    pub stats: TuneStats,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuneStats {
    pub runs: usize,
    pub answered: usize,
    pub declined: usize,
    /// Declined runs that originally halted on low confidence at a step this candidate would pass.
    /// What follows that step was never recorded, so they stay declined.
    pub unresolved: usize,
    pub failed: usize,
    pub decline_rate: Metric,
    pub answered_precision: Metric,
}

#[derive(Clone, Debug, PartialEq)]
enum Replayed {
    Answered { correct: Option<bool> },
    Declined { unresolved: bool },
    Failed,
}

/// The outcome `run` would have had under `candidate`, as far as its recorded scores tell.
fn replay(run: &EvalRun, candidate: &Candidate) -> Replayed {
    let report = &run.report;
    for entry in report.trace.entries() {
        let score = match &entry.outcome {
            StepOutcome::Passed {
                score: Some(score), ..
            }
            | StepOutcome::Declined {
                score: Some(score), ..
            } => score,
            _ => continue,
        };
        if let Some(threshold) = candidate.threshold(report.category, &entry.step)
            && *score < threshold
        {
            return Replayed::Declined { unresolved: false };
        }
    }
    match &report.outcome {
        Outcome::Answered(answer) => Replayed::Answered {
            correct: run.golden.expected.matches(answer.verdict.as_ref()),
        },
        Outcome::Declined { reason, .. } => Replayed::Declined {
            unresolved: matches!(reason, DeclineReason::LowConfidence),
        },
        Outcome::Failed { .. } => Replayed::Failed,
    }
}

pub fn tune(results: &Results, candidates: &[Candidate]) -> TuneReport {
    TuneReport {
        candidates: candidates
            .iter()
            .map(|candidate| {
                let replayed: Vec<(CategoryKind, Replayed)> = results
                    .runs
                    .iter()
                    .map(|run| (run.golden.expected.category(), replay(run, candidate)))
                    .collect();
                let stats = |category: Option<CategoryKind>| {
                    TuneStats::of(
                        replayed
                            .iter()
                            .filter(|(c, _)| category.is_none_or(|category| *c == category))
                            .map(|(_, replayed)| replayed),
                    )
                };
                let overall = stats(None);
                CandidateReport {
                    name: candidate.name.clone(),
                    meets_launch_gate: overall.meets_launch_gate(),
                    overall,
                    per_category: CATEGORIES
                        .into_iter()
                        .map(|category| CategoryTuneStats {
                            category,
                            stats: stats(Some(category)),
                        })
                        .collect(),
                }
            })
            .collect(),
    }
}

impl TuneStats {
    fn of<'a>(replayed: impl Iterator<Item = &'a Replayed> + Clone) -> Self {
        let count = |pick: fn(&Replayed) -> bool| replayed.clone().filter(|r| pick(r)).count();
        Self {
            runs: replayed.clone().count(),
            answered: count(|r| matches!(r, Replayed::Answered { .. })),
            declined: count(|r| matches!(r, Replayed::Declined { .. })),
            unresolved: count(|r| matches!(r, Replayed::Declined { unresolved: true })),
            failed: count(|r| matches!(r, Replayed::Failed)),
            decline_rate: Metric::rate(
                replayed
                    .clone()
                    .map(|r| matches!(r, Replayed::Declined { .. })),
            ),
            answered_precision: Metric::rate(replayed.filter_map(|r| match r {
                Replayed::Answered { correct } => *correct,
                _ => None,
            })),
        }
    }

    pub fn meets_launch_gate(&self) -> bool {
        let precision = self.answered_precision.value;
        let decline = self.decline_rate.value;
        precision.is_some_and(|p| p >= MIN_ANSWERED_PRECISION)
            && decline.is_some_and(|d| d <= MAX_DECLINE_RATE)
    }
}

impl fmt::Display for TuneReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:<20} {:<8} {:>5} {:>8} {:>8} {:>10} {:>6} {:>14} {:>14}  gate",
            "candidate",
            "scope",
            "runs",
            "answered",
            "declined",
            "unresolved",
            "failed",
            "precision",
            "decline rate",
        )?;
        for candidate in &self.candidates {
            let rows = std::iter::once((None, &candidate.overall)).chain(
                candidate
                    .per_category
                    .iter()
                    .map(|entry| (Some(entry.category), &entry.stats)),
            );
            for (category, stats) in rows {
                let (name, gate) = match category {
                    None if candidate.meets_launch_gate => (candidate.name.as_str(), "  PASS"),
                    None => (candidate.name.as_str(), ""),
                    Some(_) => ("", ""),
                };
                writeln!(
                    f,
                    "{:<20} {:<8} {:>5} {:>8} {:>8} {:>10} {:>6} {:>14} {:>14}{gate}",
                    name,
                    scope_name(category),
                    stats.runs,
                    stats.answered,
                    stats.declined,
                    stats.unresolved,
                    stats.failed,
                    cell(stats.answered_precision),
                    cell(stats.decline_rate),
                )?;
            }
        }
        let passing: Vec<&str> = self
            .candidates
            .iter()
            .filter(|candidate| candidate.meets_launch_gate)
            .map(|candidate| candidate.name.as_str())
            .collect();
        write!(
            f,
            "Launch gate (answered precision >= {}, decline rate <= {}): {}",
            percent(Some(MIN_ANSWERED_PRECISION)),
            percent(Some(MAX_DECLINE_RATE)),
            if passing.is_empty() {
                "no candidate meets it".to_owned()
            } else {
                format!("met by {}", passing.join(", "))
            }
        )
    }
}
