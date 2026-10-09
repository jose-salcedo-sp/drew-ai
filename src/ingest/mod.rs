//! Batch ingestion: connectors fetch changed source units; the runner versions, embeds, and stores their chunks.

pub mod confluence;
pub mod jira;
pub mod slack;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::PgPool;

use crate::{
    domain::{Document, SourceKind},
    llm::{
        Embedder, ModelRef,
        bedrock_embed::{BedrockEmbedder, EmbedArgs},
    },
    store::{
        self, DbArgs,
        chunks::{ChunkStore, Stored},
    },
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
    pub embed: EmbedArgs,
    /// Re-ingest every source unit from scratch with the configured embedding model, replacing chunks embedded with any other model.
    #[arg(long)]
    pub reembed: bool,
    #[command(flatten)]
    pub confluence: confluence::Args,
    #[command(flatten)]
    pub jira: jira::Args,
    #[command(flatten)]
    pub slack: slack::Args,
}

pub async fn run(args: Args) -> anyhow::Result<()> {
    let pool = store::connect(&args.db).await?;
    let embedder = BedrockEmbedder::from_args(&args.embed).await?;

    let mut sources: Vec<Box<dyn Source>> = Vec::new();
    if let Some(connector) = confluence::Connector::from_args(&args.confluence)? {
        sources.push(Box::new(connector));
    }
    if let Some(connector) = jira::Connector::from_args(&args.jira)? {
        sources.push(Box::new(connector));
    }
    if let Some(connector) = slack::Connector::from_args(&args.slack)? {
        sources.push(Box::new(connector));
    }
    anyhow::ensure!(!sources.is_empty(), "no sources are configured");

    let reports = ingest(&pool, &embedder, &sources, args.reembed).await?;
    let failed: Vec<&str> = reports
        .iter()
        .filter(|r| r.error.is_some())
        .map(|r| r.source.as_str())
        .collect();
    anyhow::ensure!(
        failed.is_empty(),
        "ingest failed for: {}",
        failed.join(", ")
    );
    Ok(())
}

#[derive(Debug, thiserror::Error)]
#[error(
    "current chunks were embedded with {current:?} but the configured embedding model is {configured:?}; \
     mixing embedding models is not allowed, so pass --reembed to re-ingest everything with the configured model"
)]
pub struct EmbeddingModelMismatch {
    pub current: Vec<ModelRef>,
    pub configured: ModelRef,
}

#[derive(Debug)]
pub struct SourceReport {
    pub source: SourceKind,
    pub documents: u64,
    pub stored: Stored,
    /// Set when the source failed; its cursor did not advance. Counts cover what was stored before the failure.
    pub error: Option<anyhow::Error>,
}

/// Ingests every source independently: a failing source neither stops the others nor advances its cursor.
/// Fails before touching any source with `EmbeddingModelMismatch` when current chunks use another model and `reembed` is off.
pub async fn ingest(
    pool: &PgPool,
    embedder: &dyn Embedder,
    sources: &[Box<dyn Source>],
    reembed: bool,
) -> anyhow::Result<Vec<SourceReport>> {
    let started_at = Utc::now();
    let store = ChunkStore::new(pool.clone());
    let model = embedder.model();

    let current = store.current_embedding_models().await?;
    if !reembed && current.iter().any(|m| *m != model) {
        return Err(EmbeddingModelMismatch {
            current,
            configured: model,
        }
        .into());
    }

    let mut reports = Vec::new();
    for source in sources {
        let kind = source.kind();
        let mut report = SourceReport {
            source: kind,
            documents: 0,
            stored: Stored::default(),
            error: None,
        };
        let mut result =
            ingest_source(&store, embedder, source.as_ref(), reembed, &mut report).await;
        if result.is_ok() {
            result = store.set_cursor(kind, started_at).await;
        }
        match result {
            Ok(()) => tracing::info!(
                source = kind.as_str(),
                documents = report.documents,
                new_versions = report.stored.new_versions,
                superseded = report.stored.superseded,
                "source ingested"
            ),
            Err(e) => {
                tracing::error!(
                    source = kind.as_str(),
                    documents = report.documents,
                    new_versions = report.stored.new_versions,
                    superseded = report.stored.superseded,
                    error = format!("{e:#}"),
                    "source failed"
                );
                report.error = Some(e);
            }
        }
        reports.push(report);
    }

    if reembed && reports.iter().all(|r| r.error.is_none()) {
        let superseded = store.supersede_other_models(&model).await?;
        if superseded > 0 {
            tracing::warn!(
                superseded,
                "superseded chunks embedded with other models that this run did not re-ingest"
            );
        }
    }
    Ok(reports)
}

async fn ingest_source(
    store: &ChunkStore,
    embedder: &dyn Embedder,
    source: &dyn Source,
    reembed: bool,
    report: &mut SourceReport,
) -> anyhow::Result<()> {
    let kind = source.kind();
    let since = if reembed {
        None
    } else {
        store.cursor(kind).await?
    };
    let documents = source.fetch_changed(since).await?;
    for document in &documents {
        anyhow::ensure!(
            document.source == kind,
            "{} source returned document {}",
            kind.as_str(),
            document.id().0
        );
        report.stored += store.store(document, embedder).await?;
        report.documents += 1;
    }
    Ok(())
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
