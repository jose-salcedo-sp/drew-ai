//! Per-step metrics over golden-set runs, and the baseline regression gate.

use std::{
    collections::{BTreeMap, HashSet},
    fmt,
};

use futures::{StreamExt, stream};
use serde::{Deserialize, Serialize};

use crate::{
    domain::{CategoryKind, ChunkId, ChunkRef, Provenance},
    pipeline::{
        report::{Answer, Answerer, Outcome, RunReport, Verdict},
        trace::{ClassifyRecord, RerankRecord, RetrieveRecord, TraceRecord},
    },
};

use super::golden::GoldenQuestion;

pub const RECALL_K: usize = 50;
pub const HIT_K: usize = 5;

pub const CATEGORIES: [CategoryKind; 4] = [
    CategoryKind::Boolean,
    CategoryKind::Choice,
    CategoryKind::Ranking,
    CategoryKind::Open,
];

/// The `drew eval run` output file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Results {
    pub metrics: MetricsReport,
    pub runs: Vec<EvalRun>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvalRun {
    pub golden: GoldenQuestion,
    /// Derived from `golden` and `report`; kept for reading the file, never read back.
    pub scores: RunScores,
    pub report: RunReport,
}

/// One run's contribution to each metric. `None` means the run does not count toward that metric.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RunScores {
    pub classified_correctly: Option<bool>,
    pub recall_at_50: Option<f64>,
    pub hit_at_5: Option<bool>,
    pub answer_correct: Option<bool>,
    pub declined: bool,
    pub failed: bool,
    pub citations_valid: Option<bool>,
}

/// Answers every golden question, at most `concurrency` at a time, and scores the runs.
pub async fn evaluate(
    answerer: &dyn Answerer,
    golden: Vec<GoldenQuestion>,
    concurrency: usize,
) -> Results {
    let runs = stream::iter(golden)
        .map(|golden| async move {
            let report = answerer.answer(golden.question.clone()).await;
            EvalRun::new(golden, report)
        })
        .buffered(concurrency.max(1))
        .collect()
        .await;
    Results::new(runs)
}

impl Results {
    pub fn new(runs: Vec<EvalRun>) -> Self {
        Self {
            metrics: MetricsReport::of(&runs),
            runs,
        }
    }
}

impl EvalRun {
    pub fn new(golden: GoldenQuestion, report: RunReport) -> Self {
        Self {
            scores: RunScores::of(&golden, &report),
            golden,
            report,
        }
    }
}

impl RunScores {
    pub fn of(golden: &GoldenQuestion, report: &RunReport) -> Self {
        let expected = golden.expected.category();
        let classify = last_record(report, |record| match record {
            TraceRecord::Classify(record) => Some(record),
            _ => None,
        });
        let retrieve = last_record(report, |record| match record {
            TraceRecord::Retrieve(record) => Some(record),
            _ => None,
        });
        let rerank = last_record(report, |record| match record {
            TraceRecord::Rerank(record) => Some(record),
            _ => None,
        });
        let relevant: HashSet<&ChunkId> = golden.relevant_chunks.iter().collect();

        Self {
            classified_correctly: classified_correctly(report, classify, expected),
            recall_at_50: retrieve
                .filter(|_| !relevant.is_empty())
                .map(|record| recall(&relevant, record)),
            hit_at_5: rerank
                .filter(|_| !relevant.is_empty())
                .map(|record| hit(&relevant, record)),
            answer_correct: match &report.outcome {
                Outcome::Answered(answer) => golden.expected.matches(answer.verdict.as_ref()),
                _ => None,
            },
            declined: matches!(report.outcome, Outcome::Declined { .. }),
            failed: matches!(report.outcome, Outcome::Failed { .. }),
            citations_valid: match &report.outcome {
                Outcome::Answered(answer) => Some(citations_valid(answer, retrieve, rerank)),
                _ => None,
            },
        }
    }
}

fn last_record<'a, T>(
    report: &'a RunReport,
    pick: impl Fn(&'a TraceRecord) -> Option<&'a T>,
) -> Option<&'a T> {
    report
        .trace
        .entries()
        .iter()
        .rev()
        .find_map(|entry| entry.record.as_ref().and_then(&pick))
}

/// The final category wins (a `choice` can be reclassified as `open`); a run that halted at
/// `Classify` falls back to what the step recorded.
fn classified_correctly(
    report: &RunReport,
    classify: Option<&ClassifyRecord>,
    expected: CategoryKind,
) -> Option<bool> {
    if report.category.is_none() && classify.is_none() {
        return None;
    }
    let predicted = report
        .category
        .or_else(|| classify.and_then(|record| record.category));
    Some(predicted == Some(expected))
}

fn recall(relevant: &HashSet<&ChunkId>, record: &RetrieveRecord) -> f64 {
    let found: HashSet<&ChunkId> = record
        .chunks
        .iter()
        .take(RECALL_K)
        .map(|ranked| &ranked.chunk.id)
        .filter(|id| relevant.contains(id))
        .collect();
    found.len() as f64 / relevant.len() as f64
}

fn hit(relevant: &HashSet<&ChunkId>, record: &RerankRecord) -> bool {
    record
        .chunks
        .iter()
        .take(HIT_K)
        .any(|ranked| relevant.contains(&ranked.chunk.id))
}

/// Retrieved means returned by `Retrieve` or by `Rerank`, which also sees parent expansions.
fn citations_valid(
    answer: &Answer,
    retrieve: Option<&RetrieveRecord>,
    rerank: Option<&RerankRecord>,
) -> bool {
    let retrieved: HashSet<&ChunkRef> = retrieve
        .map(|record| &record.chunks)
        .into_iter()
        .chain(rerank.map(|record| &record.chunks))
        .flatten()
        .map(|ranked| &ranked.chunk)
        .collect();
    let retrieved_ids: HashSet<&ChunkId> = retrieved.iter().map(|chunk| &chunk.id).collect();

    let citations_ok = answer
        .citations
        .iter()
        .all(|citation| retrieved.contains(&citation.chunk));
    let provenance_ok = cited_chunks(answer).all(|id| retrieved_ids.contains(id));
    citations_ok && provenance_ok
}

fn cited_chunks(answer: &Answer) -> impl Iterator<Item = &ChunkId> {
    let verdict_sources: Vec<&[Provenance]> = match &answer.verdict {
        None | Some(Verdict::Boolean { .. }) => vec![],
        Some(Verdict::Choice { selected, rejected }) => std::iter::once(selected)
            .chain(rejected.iter().map(|rejected| &rejected.option))
            .map(|cited| cited.sources.as_slice())
            .collect(),
        Some(Verdict::Ranking { items, .. }) => {
            items.iter().map(|item| item.sources.as_slice()).collect()
        }
    };
    answer
        .why
        .iter()
        .map(|claim| claim.sources.as_slice())
        .chain(verdict_sources)
        .flatten()
        .filter_map(|source| match source {
            Provenance::Chunk(id) => Some(id),
            Provenance::UserPrompt(_) => None,
        })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricsReport {
    pub overall: Metrics,
    /// Grouped by the golden (expected) category, in `CATEGORIES` order.
    pub per_category: Vec<CategoryMetrics>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CategoryMetrics {
    pub category: CategoryKind,
    #[serde(flatten)]
    pub metrics: Metrics,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    pub runs: usize,
    pub classification_accuracy: Metric,
    pub retrieval_recall_at_50: Metric,
    pub rerank_hit_at_5: Metric,
    pub answer_precision: Metric,
    pub decline_rate: Metric,
    pub failure_rate: Metric,
    pub citation_validity: Metric,
}

/// A mean over the `n` runs that count toward it; `value` is `None` when `n` is 0.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Metric {
    pub value: Option<f64>,
    pub n: usize,
}

impl Metric {
    pub fn mean(values: impl IntoIterator<Item = f64>) -> Self {
        let (sum, n) = values
            .into_iter()
            .fold((0.0, 0), |(sum, n), value| (sum + value, n + 1));
        Self {
            value: (n > 0).then(|| sum / n as f64),
            n,
        }
    }

    pub fn rate(hits: impl IntoIterator<Item = bool>) -> Self {
        Self::mean(hits.into_iter().map(|hit| if hit { 1.0 } else { 0.0 }))
    }
}

impl MetricsReport {
    pub fn of(runs: &[EvalRun]) -> Self {
        let scores = |category: Option<CategoryKind>| {
            runs.iter()
                .filter(move |run| category.is_none_or(|c| run.golden.expected.category() == c))
                .map(|run| &run.scores)
        };
        Self {
            overall: Metrics::of(scores(None)),
            per_category: CATEGORIES
                .into_iter()
                .map(|category| CategoryMetrics {
                    category,
                    metrics: Metrics::of(scores(Some(category))),
                })
                .collect(),
        }
    }

    pub fn category(&self, category: CategoryKind) -> Option<&Metrics> {
        self.per_category
            .iter()
            .find(|entry| entry.category == category)
            .map(|entry| &entry.metrics)
    }
}

impl Metrics {
    pub fn of<'a>(scores: impl Iterator<Item = &'a RunScores> + Clone) -> Self {
        Self {
            runs: scores.clone().count(),
            classification_accuracy: Metric::rate(
                scores.clone().filter_map(|s| s.classified_correctly),
            ),
            retrieval_recall_at_50: Metric::mean(scores.clone().filter_map(|s| s.recall_at_50)),
            rerank_hit_at_5: Metric::rate(scores.clone().filter_map(|s| s.hit_at_5)),
            answer_precision: Metric::rate(scores.clone().filter_map(|s| s.answer_correct)),
            decline_rate: Metric::rate(scores.clone().map(|s| s.declined)),
            failure_rate: Metric::rate(scores.clone().map(|s| s.failed)),
            citation_validity: Metric::rate(scores.filter_map(|s| s.citations_valid)),
        }
    }

    pub fn get(&self, name: MetricName) -> Metric {
        match name {
            MetricName::ClassificationAccuracy => self.classification_accuracy,
            MetricName::RetrievalRecallAt50 => self.retrieval_recall_at_50,
            MetricName::RerankHitAt5 => self.rerank_hit_at_5,
            MetricName::AnswerPrecision => self.answer_precision,
            MetricName::DeclineRate => self.decline_rate,
            MetricName::FailureRate => self.failure_rate,
            MetricName::CitationValidity => self.citation_validity,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricName {
    ClassificationAccuracy,
    #[serde(rename = "retrieval_recall_at_50")]
    RetrievalRecallAt50,
    #[serde(rename = "rerank_hit_at_5")]
    RerankHitAt5,
    AnswerPrecision,
    DeclineRate,
    FailureRate,
    CitationValidity,
}

impl MetricName {
    pub const ALL: [Self; 7] = [
        Self::ClassificationAccuracy,
        Self::RetrievalRecallAt50,
        Self::RerankHitAt5,
        Self::AnswerPrecision,
        Self::DeclineRate,
        Self::FailureRate,
        Self::CitationValidity,
    ];

    pub fn higher_is_better(self) -> bool {
        !matches!(self, Self::DeclineRate | Self::FailureRate)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::ClassificationAccuracy => "classify",
            Self::RetrievalRecallAt50 => "recall@50",
            Self::RerankHitAt5 => "hit@5",
            Self::AnswerPrecision => "precision",
            Self::DeclineRate => "declined",
            Self::FailureRate => "failed",
            Self::CitationValidity => "citations",
        }
    }
}

/// A previous results file (or any JSON with its `metrics`), plus per-metric tolerances.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Baseline {
    pub metrics: MetricsReport,
    /// Allowed drop (or rise, for decline and failure rates) before a metric counts as regressed. Missing means 0.
    #[serde(default)]
    pub tolerances: BTreeMap<MetricName, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Regression {
    /// `None` for the overall metrics.
    pub category: Option<CategoryKind>,
    pub metric: MetricName,
    pub baseline: f64,
    /// `None` when no run counted toward the metric this time.
    pub current: Option<f64>,
    pub tolerance: f64,
}

/// Every metric, overall and per category, that is worse than the baseline by more than its tolerance.
pub fn regressions(baseline: &Baseline, current: &MetricsReport) -> Vec<Regression> {
    let scopes = std::iter::once((
        None,
        Some(&baseline.metrics.overall),
        Some(&current.overall),
    ))
    .chain(CATEGORIES.into_iter().map(|category| {
        (
            Some(category),
            baseline.metrics.category(category),
            current.category(category),
        )
    }));
    let mut found = Vec::new();
    for (category, base, now) in scopes {
        let Some(base) = base else { continue };
        for metric in MetricName::ALL {
            let Some(baseline_value) = base.get(metric).value else {
                continue;
            };
            let current_value = now.and_then(|now| now.get(metric).value);
            let tolerance = baseline.tolerances.get(&metric).copied().unwrap_or(0.0);
            let regressed = match current_value {
                None => true,
                Some(value) if metric.higher_is_better() => value < baseline_value - tolerance,
                Some(value) => value > baseline_value + tolerance,
            };
            if regressed {
                found.push(Regression {
                    category,
                    metric,
                    baseline: baseline_value,
                    current: current_value,
                    tolerance,
                });
            }
        }
    }
    found
}

pub(super) fn scope_name(category: Option<CategoryKind>) -> &'static str {
    match category {
        None => "overall",
        Some(CategoryKind::Boolean) => "boolean",
        Some(CategoryKind::Choice) => "choice",
        Some(CategoryKind::Ranking) => "ranking",
        Some(CategoryKind::Open) => "open",
    }
}

pub(super) fn percent(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| format!("{:.1}%", value * 100.0))
}

pub(super) fn cell(metric: Metric) -> String {
    format!("{} ({})", percent(metric.value), metric.n)
}

impl fmt::Display for MetricsReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:<8} {:>5}", "scope", "runs")?;
        for metric in MetricName::ALL {
            write!(f, " {:>14}", metric.label())?;
        }
        writeln!(f)?;
        let rows = std::iter::once((None, &self.overall)).chain(
            self.per_category
                .iter()
                .map(|entry| (Some(entry.category), &entry.metrics)),
        );
        for (category, metrics) in rows {
            write!(f, "{:<8} {:>5}", scope_name(category), metrics.runs)?;
            for metric in MetricName::ALL {
                write!(f, " {:>14}", cell(metrics.get(metric)))?;
            }
            writeln!(f)?;
        }
        write!(
            f,
            "Each cell is value (runs counted). Recall@{RECALL_K} and hit@{HIT_K} count runs with \
             relevant chunks whose step ran; precision counts answered runs with a gradable answer."
        )
    }
}

impl fmt::Display for Regression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}: {} -> {} (tolerance {:.1} pts)",
            scope_name(self.category),
            self.metric.label(),
            percent(Some(self.baseline)),
            percent(self.current),
            self.tolerance * 100.0,
        )
    }
}
