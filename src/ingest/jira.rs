//! Jira: an issue plus Drew's comments is one unit.

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::domain::{Document, SourceKind};

use super::Source;

#[derive(Debug, clap::Args)]
pub struct Args {}

pub struct Connector {}

impl Connector {
    /// `None` when this source is not configured.
    pub fn from_args(_args: &Args) -> anyhow::Result<Option<Self>> {
        Ok(None)
    }
}

#[async_trait]
impl Source for Connector {
    fn kind(&self) -> SourceKind {
        SourceKind::Jira
    }

    async fn fetch_changed(&self, _since: Option<DateTime<Utc>>) -> anyhow::Result<Vec<Document>> {
        anyhow::bail!("Jira connector is not implemented yet")
    }
}
