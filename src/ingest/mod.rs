//! Batch ingestion: connectors fetch changed source units; the runner versions, embeds, and stores their chunks.

pub mod confluence;
pub mod jira;
pub mod slack;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::{
    domain::{Document, SourceKind},
    store::DbArgs,
};

#[async_trait]
pub trait Source: Send + Sync {
    fn kind(&self) -> SourceKind;

    /// Every source unit updated after `since`, or all of them when `since` is `None`.
    async fn fetch_changed(&self, since: Option<DateTime<Utc>>) -> anyhow::Result<Vec<Document>>;
}

#[derive(Debug, clap::Args)]
pub struct Args {
    #[command(flatten)]
    pub db: DbArgs,
    #[command(flatten)]
    pub confluence: confluence::Args,
    #[command(flatten)]
    pub jira: jira::Args,
    #[command(flatten)]
    pub slack: slack::Args,
}

pub async fn run(_args: Args) -> anyhow::Result<()> {
    anyhow::bail!("drew ingest is not implemented yet")
}

pub mod fakes {
    use async_trait::async_trait;
    use chrono::{DateTime, Utc};

    use super::Source;
    use crate::domain::{Document, SourceKind};

    /// Returns its documents whose `updated_at` is after `since`.
    pub struct StaticSource {
        pub kind: SourceKind,
        pub documents: Vec<Document>,
    }

    #[async_trait]
    impl Source for StaticSource {
        fn kind(&self) -> SourceKind {
            self.kind
        }

        async fn fetch_changed(
            &self,
            since: Option<DateTime<Utc>>,
        ) -> anyhow::Result<Vec<Document>> {
            Ok(self
                .documents
                .iter()
                .filter(|d| since.is_none_or(|since| d.updated_at > since))
                .cloned()
                .collect())
        }
    }
}
