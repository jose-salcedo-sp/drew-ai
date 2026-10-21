//! Every pipeline tunable in one JSON file (`config/drew.json`), so `drew eval` can print and record what it ran with.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    domain::{CategoryKind, Probability},
    retrieval::hybrid::HybridConfig,
};

pub const DEFAULT_PATH: &str = "config/drew.json";

/// Steps whose score `Run::then` checks against a threshold. `classify` has `classify.min_confidence` instead,
/// which falls back to `open` rather than declining; `explain` and `verify` are not scored.
pub const THRESHOLDED_STEPS: [&str; 2] = ["rerank", "answer"];

#[derive(Clone, Debug, Default, clap::Args)]
pub struct ConfigArgs {
    /// Pipeline configuration JSON. Unset: `config/drew.json` when it exists, else built-in defaults.
    #[arg(long = "config", env = "DREW_CONFIG")]
    pub path: Option<PathBuf>,
}

impl ConfigArgs {
    /// An empty `DREW_CONFIG`, as in `.env.example`, counts as unset.
    pub fn load(&self) -> eros::Result<PipelineConfig> {
        match self
            .path
            .as_ref()
            .filter(|path| !path.as_os_str().is_empty())
        {
            Some(path) => PipelineConfig::read(path),
            None if Path::new(DEFAULT_PATH).exists() => {
                PipelineConfig::read(Path::new(DEFAULT_PATH))
            }
            None => {
                tracing::warn!("no {DEFAULT_PATH} and no --config; using built-in defaults");
                Ok(PipelineConfig::default())
            }
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PipelineConfig {
    pub retrieval: HybridConfig,
    pub classify: ClassifyConfig,
    pub rerank: RerankConfig,
    /// Minimum score per step, keyed by the step's trace name; the shape of a `drew eval tune` candidate.
    pub thresholds: BTreeMap<String, Probability>,
    /// Overrides `thresholds` for runs classified into that category.
    pub per_category: PerCategory,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClassifyConfig {
    /// Below this probability, a `boolean`, `choice` or `ranking` classification falls back to `open`.
    pub min_confidence: Option<Probability>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RerankConfig {
    /// Matches kept for `Answer` and `Explain`.
    pub top_k: usize,
    /// Best matches by fused retrieval score that System One judges.
    pub candidates: usize,
    /// An evidence (non-authoritative) match is ranked as if this many places lower.
    pub evidence_rank_penalty: usize,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            candidates: 25,
            evidence_rank_penalty: 10,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PerCategory {
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub boolean: BTreeMap<String, Probability>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub choice: BTreeMap<String, Probability>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub ranking: BTreeMap<String, Probability>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub open: BTreeMap<String, Probability>,
}

impl PerCategory {
    fn get(&self, category: CategoryKind) -> &BTreeMap<String, Probability> {
        match category {
            CategoryKind::Boolean => &self.boolean,
            CategoryKind::Choice => &self.choice,
            CategoryKind::Ranking => &self.ranking,
            CategoryKind::Open => &self.open,
        }
    }

    fn all(&self) -> impl Iterator<Item = &BTreeMap<String, Probability>> {
        [&self.boolean, &self.choice, &self.ranking, &self.open].into_iter()
    }
}

impl PipelineConfig {
    pub fn read(path: &Path) -> eros::Result<Self> {
        read(path)
    }

    pub fn parse(text: &str) -> eros::Result<Self> {
        let config: Self = serde_json::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> eros::Result<()> {
        eros::ensure!(
            self.rerank.top_k > 0 && self.rerank.candidates >= self.rerank.top_k,
            "rerank needs 0 < top_k <= candidates"
        );
        for thresholds in std::iter::once(&self.thresholds).chain(self.per_category.all()) {
            for step in thresholds.keys() {
                eros::ensure!(
                    THRESHOLDED_STEPS.contains(&step.as_str()),
                    "no threshold applies to step `{step}`; thresholded steps are {THRESHOLDED_STEPS:?} \
                     (classify uses classify.min_confidence)"
                );
            }
        }
        Ok(())
    }

    /// The category's override, else the general threshold.
    pub fn threshold(&self, category: CategoryKind, step: &str) -> Option<Probability> {
        self.per_category
            .get(category)
            .get(step)
            .or_else(|| self.thresholds.get(step))
            .copied()
    }
}

// `#[eros::context]` on an associated function without `self` calls its inner function unqualified.
#[eros::context("reading pipeline config {}", path.display())]
fn read(path: &Path) -> eros::Result<PipelineConfig> {
    let text = fs::read_to_string(path)?;
    PipelineConfig::parse(&text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_config_parses() {
        let config = PipelineConfig::parse(include_str!("../config/drew.json")).unwrap();
        assert!(config.classify.min_confidence.is_some());
        assert!(config.threshold(CategoryKind::Boolean, "answer").is_some());
    }

    #[test]
    fn an_empty_path_is_unset_and_a_missing_one_is_an_error() {
        let empty = ConfigArgs {
            path: Some(PathBuf::new()),
        };
        assert!(empty.load().is_ok());
        let missing = ConfigArgs {
            path: Some("no/such/drew.json".into()),
        };
        assert!(missing.load().is_err());
    }

    #[test]
    fn empty_object_is_the_defaults() {
        assert_eq!(
            PipelineConfig::parse("{}").unwrap(),
            PipelineConfig::default()
        );
    }

    #[test]
    fn category_override_wins_over_the_general_threshold() {
        let config = PipelineConfig::parse(
            r#"{"thresholds": {"answer": 0.6}, "per_category": {"boolean": {"answer": 0.8}}}"#,
        )
        .unwrap();
        assert_eq!(
            config.threshold(CategoryKind::Boolean, "answer"),
            Some(Probability::new(0.8).unwrap())
        );
        assert_eq!(
            config.threshold(CategoryKind::Open, "answer"),
            Some(Probability::new(0.6).unwrap())
        );
        assert_eq!(config.threshold(CategoryKind::Open, "rerank"), None);
    }

    #[test]
    fn typos_and_unthresholded_steps_are_rejected() {
        for bad in [
            r#"{"thresold": {}}"#,
            r#"{"rerank": {"top": 5}}"#,
            r#"{"thresholds": {"classify": 0.5}}"#,
            r#"{"per_category": {"boolean": {"explain": 0.5}}}"#,
            r#"{"rerank": {"top_k": 10, "candidates": 5}}"#,
        ] {
            assert!(PipelineConfig::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn probabilities_outside_zero_to_one_are_rejected_on_every_load_path() {
        for bad in [
            r#"{"thresholds": {"answer": 1.5}}"#,
            r#"{"thresholds": {"rerank": -0.1}}"#,
            r#"{"thresholds": {"answer": 1e39}}"#,
            r#"{"per_category": {"boolean": {"answer": 1.01}}}"#,
            r#"{"classify": {"min_confidence": 1.2}}"#,
            r#"{"classify": {"min_confidence": -0.5}}"#,
        ] {
            assert!(PipelineConfig::parse(bad).is_err(), "{bad}");
            assert!(
                serde_json::from_str::<PipelineConfig>(bad).is_err(),
                "{bad}"
            );
        }
        assert!(PipelineConfig::parse(r#"{"thresholds": {"answer": 1.0}}"#).is_ok());
    }
}
