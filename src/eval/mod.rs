//! `drew eval`: golden-set metrics per step, and threshold tuning over persisted runs.

pub mod golden;
pub mod metrics;
pub mod tune;

use std::{fs, path::PathBuf, sync::Arc};

use anyhow::Context;

use crate::{
    pipeline::{
        answerer::{PipelineAnswerer, PipelineArgs},
        report::Answerer,
    },
    store::{self, DbArgs},
};

use self::{
    metrics::{Baseline, Results},
    tune::Candidate,
};

#[derive(Debug, clap::Args)]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Answer every golden question and report per-step metrics.
    Run(RunCommand),
    /// Re-apply candidate thresholds to a results file's recorded scores.
    Tune(TuneArgs),
}

#[derive(Debug, clap::Args)]
pub struct RunCommand {
    #[command(flatten)]
    pub files: RunArgs,
    #[command(flatten)]
    pub db: DbArgs,
    #[command(flatten)]
    pub pipeline: PipelineArgs,
}

#[derive(Debug, clap::Args)]
pub struct RunArgs {
    /// Golden set, JSONL.
    #[arg(long)]
    pub golden: PathBuf,
    /// Where to write the results JSON.
    #[arg(long)]
    pub out: PathBuf,
    /// Results JSON (optionally with `tolerances`) to gate against; any regression exits non-zero.
    #[arg(long)]
    pub baseline: Option<PathBuf>,
    /// Questions answered at once.
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,
}

#[derive(Debug, clap::Args)]
pub struct TuneArgs {
    /// Results JSON written by `drew eval run`.
    #[arg(long)]
    pub results: PathBuf,
    /// JSON array of candidates: `{"name", "thresholds": {step: min}, "per_category": {category: {step: min}}}`.
    #[arg(long)]
    pub candidates: PathBuf,
    /// Where to write the tuning report JSON.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

pub async fn run(args: Args) -> anyhow::Result<()> {
    match args.command {
        Command::Run(args) => {
            let pool = store::connect(&args.db).await?;
            let answerer = PipelineAnswerer::from_args(pool, &args.pipeline).await?;
            run_golden(Arc::new(answerer), args.files).await
        }
        Command::Tune(args) => run_tune(args),
    }
}

pub async fn run_golden(answerer: Arc<dyn Answerer>, args: RunArgs) -> anyhow::Result<()> {
    let text = fs::read_to_string(&args.golden)
        .with_context(|| format!("reading {}", args.golden.display()))?;
    let golden = golden::parse(&text).with_context(|| format!("in {}", args.golden.display()))?;
    let baseline: Option<Baseline> = args.baseline.as_deref().map(read_json).transpose()?;

    let results = metrics::evaluate(answerer.as_ref(), golden, args.concurrency).await;
    write_json(&args.out, &results)?;
    println!("{}", results.metrics);
    println!("Results written to {}", args.out.display());

    let Some(baseline) = baseline else {
        return Ok(());
    };
    let regressions = metrics::regressions(&baseline, &results.metrics);
    if regressions.is_empty() {
        println!("No regressions against the baseline.");
        return Ok(());
    }
    println!("Regressions against the baseline:");
    for regression in &regressions {
        println!("  {regression}");
    }
    anyhow::bail!("{} metric(s) regressed beyond tolerance", regressions.len())
}

pub fn run_tune(args: TuneArgs) -> anyhow::Result<()> {
    let results: Results = read_json(&args.results)?;
    let candidates: Vec<Candidate> = read_json(&args.candidates)?;
    let report = tune::tune(&results, &candidates);
    println!("{report}");
    if let Some(out) = &args.out {
        write_json(out, &report)?;
        println!("Tuning report written to {}", out.display());
    }
    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> anyhow::Result<T> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

fn write_json(path: &std::path::Path, value: &impl serde::Serialize) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(value)?;
    fs::write(path, json + "\n").with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests;
