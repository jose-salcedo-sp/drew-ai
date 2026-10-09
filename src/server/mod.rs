//! OpenAI-compatible HTTP API: `/v1/models`, `/v1/chat/completions`, `/v1/runs/{id}`.

use crate::store::DbArgs;

#[derive(Debug, clap::Args)]
pub struct Args {
    #[command(flatten)]
    pub db: DbArgs,
    #[arg(long, env = "DREW_BIND", default_value = "0.0.0.0:8080")]
    pub bind: String,
}

pub async fn run(_args: Args) -> anyhow::Result<()> {
    anyhow::bail!("drew serve is not implemented yet")
}
