//! `drew eval`: golden-set metrics per step, and threshold tuning over persisted runs.

pub mod golden;
pub mod metrics;
pub mod progress;
pub mod tune;

use std::{
    fs,
    io::Write as _,
    path::PathBuf,
    process::{Command as Process, Stdio},
    sync::Arc,
};

use chrono::Utc;

use crate::{
    pipeline::{
        answerer::{PipelineAnswerer, PipelineArgs},
        report::Answerer,
    },
    store::{self, DbArgs},
};

use self::{
    metrics::{Baseline, Results, RunSetup},
    tune::Candidate,
};

#[derive(Debug, clap::Args)]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, clap::Subcommand)]
#[expect(clippy::large_enum_variant, reason = "parsed once at startup")]
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

pub async fn run(args: Args) -> eros::Result<()> {
    match args.command {
        Command::Run(args) => {
            let pool = store::connect(&args.db).await?;
            let answerer = PipelineAnswerer::from_args(pool, &args.pipeline).await?;
            let setup = RunSetup {
                started_at: Utc::now(),
                finished_at: None,
                golden: args.files.golden.display().to_string(),
                git: git_describe(),
                config: answerer.config().clone(),
                models: answerer.models(),
            };
            run_golden(Arc::new(answerer), args.files, Some(setup)).await
        }
        Command::Tune(args) => run_tune(args),
    }
}

/// `setup`, when known, is printed before the first question and recorded in the results file.
pub async fn run_golden(
    answerer: Arc<dyn Answerer>,
    args: RunArgs,
    setup: Option<RunSetup>,
) -> eros::Result<()> {
    let golden = read_golden(&args.golden)?;
    let baseline: Option<Baseline> = args.baseline.as_deref().map(read_json).transpose()?;

    if let Some(setup) = &setup {
        print_setup(setup);
    }
    let bar = progress::Bar::new(golden.len(), args.concurrency);
    let mut results = metrics::evaluate(answerer.as_ref(), golden, args.concurrency, &bar).await;
    bar.finish();
    results.setup = setup.map(|setup| RunSetup {
        finished_at: Some(Utc::now()),
        ..setup
    });
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
    eros::bail!("{} metric(s) regressed beyond tolerance", regressions.len())
}

pub fn run_tune(args: TuneArgs) -> eros::Result<()> {
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

/// Through `jq -C` when it is installed, so the config reads like any other JSON in the terminal.
fn print_setup(setup: &RunSetup) {
    let json = serde_json::json!({
        "golden": setup.golden,
        "git": setup.git,
        "models": setup.models,
        "config": setup.config,
    });
    let pretty = serde_json::to_string_pretty(&json)
        .expect("the setup is plain data, which always serializes");
    println!("{}", console::style("    Setup").green().bold());
    let colour = if console::colors_enabled() {
        "-C"
    } else {
        "-M"
    };
    let jq = Process::new("jq")
        .args([colour, "."])
        .stdin(Stdio::piped())
        .stdout(Stdio::inherit())
        .stderr(Stdio::null())
        .spawn()
        .and_then(|mut jq| {
            if let Some(mut stdin) = jq.stdin.take() {
                stdin.write_all(pretty.as_bytes())?;
            }
            jq.wait()
        });
    if !jq.is_ok_and(|status| status.success()) {
        println!("{pretty}");
    }
}

fn git_describe() -> Option<String> {
    let output = Process::new("git")
        .args(["describe", "--always", "--dirty"])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

#[eros::context("reading golden set {}", path.display())]
fn read_golden(path: &std::path::Path) -> eros::Result<Vec<golden::GoldenQuestion>> {
    let text = fs::read_to_string(path)?;
    Ok(golden::parse(&text)?)
}

#[eros::context("reading {}", path.display())]
fn read_json<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> eros::Result<T> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

#[eros::context("writing {}", path.display())]
fn write_json(path: &std::path::Path, value: &impl serde::Serialize) -> eros::Result<()> {
    let json = serde_json::to_string_pretty(value)?;
    Ok(fs::write(path, json + "\n")?)
}

#[cfg(test)]
mod tests;
