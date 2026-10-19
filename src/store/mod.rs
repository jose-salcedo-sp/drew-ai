//! Postgres + pgvector. Schema lives in `migrations/`; queries use runtime-checked `sqlx::query`, not the macros.

pub mod chunks;
pub mod fakes;
pub mod runs;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, migrate::Migrator, postgres::PgPoolOptions};

use crate::{
    domain::{RunId, UserId},
    pipeline::report::RunReport,
};

pub static MIGRATOR: Migrator = sqlx::migrate!("./migrations");

#[derive(Clone, Debug, clap::Args)]
pub struct DbArgs {
    #[arg(long, env = "DATABASE_URL")]
    pub database_url: String,
}

/// Connects and applies pending migrations.
pub async fn connect(args: &DbArgs) -> eros::Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&args.database_url)
        .await?;
    MIGRATOR.run(&pool).await?;
    Ok(pool)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredRun {
    pub user: UserId,
    pub created_at: DateTime<Utc>,
    pub report: RunReport,
}

/// Every run is persisted: answered, declined, and failed.
#[async_trait]
pub trait RunStore: Send + Sync {
    async fn save(&self, user: &UserId, report: &RunReport) -> eros::Result<()>;

    async fn get(&self, id: RunId) -> eros::Result<Option<StoredRun>>;
}
