use clap::{Parser, Subcommand};
use drew::{eval, ingest, server};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "drew",
    about = "Drew AI: answers about SAP and ARM internals, with provenance"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the OpenAI-compatible API.
    Serve(server::Args),
    /// Pull changed content from the configured sources.
    Ingest(ingest::Args),
    /// Run the golden set and report per-step metrics.
    Eval(eval::Args),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    match Cli::parse().command {
        Command::Serve(args) => server::run(args).await,
        Command::Ingest(args) => ingest::run(args).await,
        Command::Eval(args) => eval::run(args).await,
    }
}
