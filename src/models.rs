//! `drew models check`: one call to each model backend `drew serve` and `drew eval run` would use, with the same
//! arguments, so a wrong URL, key, model id or AWS profile shows in seconds instead of mid-run.

use std::{fmt, time::Duration};

use aws_config::{BehaviorVersion, timeout::TimeoutConfig};
use aws_sdk_bedrockruntime::{
    config::{Credentials, ProvideCredentials},
    error::DisplayErrorContext,
};
use eros::{Context, ensure};
use serde_json::{Value, json};
use tokio::time::Instant;

use crate::{
    llm::{
        self, ChatModel, ChatRequest, EMBEDDING_DIM, EmbedPurpose, Embedder, ModelRef,
        backend::{ChatBackendKind, EmbedBackendKind, SystemOneBackendKind},
        system_one::{Decision, DecisionRequest, SystemOne},
    },
    pipeline::answerer::PipelineArgs,
};

/// Bounds loading the AWS config and resolving credentials, which may query instance metadata.
const AWS_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const AWS_CREDENTIALS_TIMEOUT: Duration = Duration::from_secs(15);
/// `Explain`'s own budget: a reasoning effort that spends it all before answering fails here too.
const PROBE_MAX_TOKENS: u32 = 2048;
const QUESTION_ID: &str = "check";

/// `drew models` options.
#[derive(Debug, clap::Args)]
pub struct Args {
    /// `check`.
    #[command(subcommand)]
    pub command: Command,
}

/// The `drew models` subcommands.
#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Call the embedder, the Explain LLM and System One once each, and report each one's model, latency and
    /// outcome; exits non-zero when any fails.
    Check(PipelineArgs),
}

/// A model port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Port {
    /// [`Embedder`].
    Embeddings,
    /// [`ChatModel`] for `Explain`.
    Explain,
    /// [`SystemOne`].
    SystemOne,
}

impl fmt::Display for Port {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Embeddings => "embeddings",
            Self::Explain => "explain",
            Self::SystemOne => "system one",
        })
    }
}

/// One call to one backend.
#[derive(Debug)]
pub struct Probe {
    /// The port called.
    pub port: Port,
    /// The adapter's model, or the configured name and `not built` when the adapter could not be built.
    pub model: ModelRef,
    /// Time spent in the call, zero when the adapter could not be built.
    pub latency: Duration,
    /// The call's outcome.
    pub outcome: eros::Result<()>,
}

/// What `drew models check` found.
#[derive(Debug)]
pub struct Check {
    /// Backend selections worth knowing before reading the probes, such as Bedrock and its AWS region.
    pub notes: Vec<String>,
    /// One per configured port, in port order.
    pub probes: Vec<Probe>,
}

impl Check {
    /// How many probes failed.
    pub fn failed(&self) -> usize {
        self.probes
            .iter()
            .filter(|probe| probe.outcome.is_err())
            .count()
    }
}

impl fmt::Display for Check {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for note in &self.notes {
            writeln!(f, "{note}")?;
        }
        for probe in &self.probes {
            let ModelRef { id, version } = &probe.model;
            let latency = probe.latency;
            match &probe.outcome {
                Ok(()) => writeln!(f, "{:<10}  {id} ({version})  {latency:.1?}  ok", probe.port)?,
                Err(error) => {
                    writeln!(
                        f,
                        "{:<10}  {id} ({version})  {latency:.1?}  error",
                        probe.port
                    )?;
                    writeln!(f, "{error:#?}")?;
                }
            }
        }
        Ok(())
    }
}

/// Runs `drew models`.
///
/// # Errors
///
/// When any backend fails; each failure is printed with the report.
pub async fn run(args: Args) -> eros::Result<()> {
    match args.command {
        Command::Check(args) => {
            let check = check(&args).await;
            print!("{check}");
            let failed = check.failed();
            ensure!(
                failed == 0,
                "{failed} of {} model backends failed; see above",
                check.probes.len()
            );
            Ok(())
        }
    }
}

/// Calls each backend `args` selects once, concurrently. System One is not probed when its backend is `none`.
pub async fn check(args: &PipelineArgs) -> Check {
    let (bedrock, embeddings, explain, system_one) = tokio::join!(
        bedrock_note(args),
        probe_embeddings(args),
        probe_explain(args),
        probe_system_one(args),
    );
    let mut notes: Vec<String> = bedrock.into_iter().collect();
    let mut probes = vec![embeddings, explain];
    match system_one {
        Some(probe) => probes.push(probe),
        None => notes.push(
            "system one: none (DREW_SYSTEM_ONE_BACKEND=none); every question is classified open"
                .to_owned(),
        ),
    }
    Check { notes, probes }
}

/// Which ports use Bedrock, the AWS region, and which provider resolved credentials.
async fn bedrock_note(args: &PipelineArgs) -> Option<String> {
    let ports: Vec<&str> = [
        (args.embed.embed_backend == EmbedBackendKind::Bedrock)
            .then_some("embeddings (DREW_EMBED_BACKEND=bedrock)"),
        (args.chat.explain_backend == ChatBackendKind::Bedrock)
            .then_some("explain (DREW_EXPLAIN_BACKEND=bedrock)"),
    ]
    .into_iter()
    .flatten()
    .collect();
    if ports.is_empty() {
        return None;
    }
    let config = aws_config::defaults(BehaviorVersion::latest())
        .timeout_config(
            TimeoutConfig::builder()
                .connect_timeout(AWS_CONNECT_TIMEOUT)
                .build(),
        )
        .load()
        .await;
    let region = config
        .region()
        .map_or_else(|| "unset".to_owned(), ToString::to_string);
    let credentials = match config.credentials_provider() {
        None => "none configured".to_owned(),
        Some(provider) => {
            match tokio::time::timeout(AWS_CREDENTIALS_TIMEOUT, provider.provide_credentials())
                .await
            {
                Ok(Ok(credentials)) => format!("from {}", provider_name(&credentials)),
                Ok(Err(error)) => format!("none resolved: {}", DisplayErrorContext(error)),
                Err(_) => format!("none resolved within {AWS_CREDENTIALS_TIMEOUT:?}"),
            }
        }
    };
    Some(format!(
        "Bedrock selected for {}: AWS region {region}, credentials {credentials}",
        ports.join(" and ")
    ))
}

fn provider_name(credentials: &Credentials) -> String {
    // `Credentials` names its provider only in `Debug`, which redacts the secret; nothing else from it is printed.
    let debug = format!("{credentials:?}");
    debug
        .split_once("provider_name: \"")
        .and_then(|(_, rest)| rest.split_once('"'))
        .map_or_else(
            || "an unnamed provider".to_owned(),
            |(name, _)| format!("the {name} provider"),
        )
}

fn not_built(port: Port, model: &str, error: eros::ErrorUnion) -> Probe {
    Probe {
        port,
        model: ModelRef::new(model, "not built"),
        latency: Duration::ZERO,
        outcome: Err(error),
    }
}

async fn probe_embeddings(args: &PipelineArgs) -> Probe {
    let embedder = match llm::embedder(&args.embed).await {
        Ok(embedder) => embedder,
        Err(error) => return not_built(Port::Embeddings, &args.embed.embed_model, error),
    };
    let start = Instant::now();
    let outcome = embed_once(&*embedder).await;
    Probe {
        port: Port::Embeddings,
        model: embedder.model(),
        latency: start.elapsed(),
        outcome,
    }
}

#[eros::context("embedding one query")]
async fn embed_once(embedder: &dyn Embedder) -> eros::Result<()> {
    let vectors = embedder
        .embed(&["drew models check".to_owned()], EmbedPurpose::Query)
        .await?;
    let dimensions: Vec<usize> = vectors.iter().map(Vec::len).collect();
    ensure!(
        dimensions == [EMBEDDING_DIM],
        "expected one {EMBEDDING_DIM}-dimension vector, got dimensions {dimensions:?}"
    );
    Ok(())
}

async fn probe_explain(args: &PipelineArgs) -> Probe {
    let chat = match llm::explain_model(&args.chat).await {
        Ok(chat) => chat,
        Err(error) => return not_built(Port::Explain, &args.chat.explain_model, error),
    };
    let start = Instant::now();
    let outcome = explain_once(&*chat).await;
    Probe {
        port: Port::Explain,
        model: chat.model(),
        latency: start.elapsed(),
        outcome,
    }
}

#[eros::context("asking for a one-field JSON reply")]
async fn explain_once(chat: &dyn ChatModel) -> eros::Result<()> {
    let response = chat
        .complete(ChatRequest {
            system: "Reply with JSON that matches the schema.".to_owned(),
            user: "Is this message a connectivity check? Answer in `ok`.".to_owned(),
            max_tokens: PROBE_MAX_TOKENS,
            temperature: 0.0,
            json_schema: Some(json!({
                "type": "object",
                "properties": { "ok": { "type": "boolean" } },
                "required": ["ok"],
                "additionalProperties": false,
            })),
        })
        .await?;
    let reply: Value = serde_json::from_str(&response.text).context("the reply is not JSON")?;
    ensure!(
        reply["ok"].is_boolean(),
        "the reply does not match the schema: {}",
        response.text
    );
    Ok(())
}

async fn probe_system_one(args: &PipelineArgs) -> Option<Probe> {
    if args.system_one.system_one_backend == SystemOneBackendKind::None {
        return None;
    }
    let model = args
        .system_one
        .system_one_model
        .as_deref()
        .unwrap_or("unset");
    let system_one = match llm::system_one_client(&args.system_one) {
        Ok(Some(system_one)) => system_one,
        Ok(None) => return None,
        Err(error) => return Some(not_built(Port::SystemOne, model, error)),
    };
    let start = Instant::now();
    let outcome = decide_once(&*system_one).await;
    Some(Probe {
        port: Port::SystemOne,
        model: system_one.model(),
        latency: start.elapsed(),
        outcome,
    })
}

#[eros::context("asking one yes/no question")]
async fn decide_once(system_one: &dyn SystemOne) -> eros::Result<()> {
    let response = system_one
        .decide(DecisionRequest {
            state: "This message is a connectivity check from drew models check.".to_owned(),
            model: String::new(),
            questions: [(
                QUESTION_ID.to_owned(),
                Decision::Noul {
                    instructions: "Is this message a connectivity check?".to_owned(),
                },
            )]
            .into(),
        })
        .await?;
    response.yes_no(QUESTION_ID)?;
    Ok(())
}
