//! `drew eval`: golden-set metrics per step, and threshold tuning over persisted runs.

#[derive(Debug, clap::Args)]
pub struct Args {}

pub async fn run(_args: Args) -> anyhow::Result<()> {
    anyhow::bail!("drew eval is not implemented yet")
}
