//! Slack public channels named by Drew: one thread is one unit; Drew's messages are authoritative.

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
        SourceKind::Slack
    }

    async fn fetch_changed(&self, _since: Option<DateTime<Utc>>) -> anyhow::Result<Vec<Document>> {
        anyhow::bail!("Slack connector is not implemented yet")
    }
}
