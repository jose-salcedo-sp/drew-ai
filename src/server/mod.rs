//! OpenAI-compatible HTTP API: `/v1/models`, `/v1/chat/completions`, `/v1/runs/{id}`.

pub mod auth;
pub mod error;
pub mod openai;
pub mod render;

#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod tests;

use std::{convert::Infallible, sync::Arc};

use anyhow::Context;
use axum::{
    Json, Router,
    extract::{Path, State, rejection::JsonRejection},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use chrono::{DateTime, Utc};
use serde::Serialize;
use uuid::Uuid;

use crate::{
    domain::{RunId, UserId},
    pipeline::{
        answerer::{PipelineAnswerer, PipelineArgs},
        report::{Answerer, Outcome, RunReport},
    },
    store::{self, DbArgs, RunStore, StoredRun, runs::PgRunStore},
};

use self::{
    auth::{ApiKeys, User},
    error::ApiError,
    openai::{ChatRequest, Chunk, Completion, Drew, ModelList},
};

#[derive(Debug, clap::Args)]
pub struct Args {
    #[command(flatten)]
    pub db: DbArgs,
    #[arg(long, env = "DREW_BIND", default_value = "0.0.0.0:8080")]
    pub bind: String,
    /// Per-user bearer keys: `key:user_id` pairs, comma-separated.
    #[arg(long, env = "DREW_API_KEYS", hide_env_values = true)]
    pub api_keys: ApiKeys,
    #[command(flatten)]
    pub pipeline: PipelineArgs,
}

#[derive(Clone)]
pub struct AppState {
    pub answerer: Arc<dyn Answerer>,
    pub runs: Arc<dyn RunStore>,
    pub keys: Arc<ApiKeys>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/runs/{id}", get(get_run))
        .fallback(unknown_endpoint)
        .with_state(state)
}

pub async fn run(args: Args) -> anyhow::Result<()> {
    let pool = store::connect(&args.db).await?;
    let answerer = PipelineAnswerer::from_args(pool.clone(), &args.pipeline).await?;
    let state = AppState {
        answerer: Arc::new(answerer),
        runs: Arc::new(PgRunStore::new(pool)),
        keys: Arc::new(args.api_keys),
    };
    let listener = tokio::net::TcpListener::bind(&args.bind)
        .await
        .with_context(|| format!("binding {}", args.bind))?;
    tracing::info!(addr = %listener.local_addr()?, "drew serve listening");
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

/// As PID 1 in a container the process gets no default SIGTERM handler, so it has to exit on SIGTERM itself.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => {
                tracing::warn!(%error, "no SIGTERM handler");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
}

async fn models(_: User) -> Json<ModelList> {
    Json(ModelList::drew())
}

async fn chat_completions(
    State(state): State<AppState>,
    User(user): User,
    body: Result<Json<ChatRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = body.map_err(|rejection| ApiError::bad_request(rejection.body_text()))?;
    let question = request.question()?;

    let report = state.answerer.answer(question).await;
    let run_id = report.id;
    if let Err(error) = state.runs.save(&user, &report).await {
        tracing::error!(%run_id, error = format!("{error:#}"), "run not persisted");
        return Err(ApiError::internal(
            "run_not_persisted",
            format!("Run {run_id} could not be persisted."),
        )
        .with_run(run_id));
    }

    let drew = Drew::new(&report);
    tracing::info!(%run_id, user = user.0, outcome = drew.outcome, "run finished");
    let content = match &report.outcome {
        Outcome::Answered(answer) => render::answer(answer),
        Outcome::Declined { sources, .. } => render::declined(sources),
        Outcome::Failed { at, error } => {
            tracing::error!(%run_id, at, error, "run failed");
            return Err(
                ApiError::internal("run_failed", format!("Run {run_id} failed at {at}."))
                    .with_run(run_id)
                    .with_drew(drew),
            );
        }
    };

    let created = Utc::now().timestamp();
    if request.stream != Some(true) {
        return Ok(Json(Completion::new(drew, content, created)).into_response());
    }
    let events: Vec<Result<Event, Infallible>> = Chunk::pair(drew, content, created)
        .iter()
        .map(|chunk| {
            let data = serde_json::to_string(chunk).expect("chunk serializes");
            Ok(Event::default().data(data))
        })
        .chain([Ok(Event::default().data("[DONE]"))])
        .collect();
    Ok(Sse::new(futures::stream::iter(events)).into_response())
}

/// A stored run as returned by `GET /v1/runs/{id}`.
#[derive(Serialize)]
struct RunView {
    #[serde(flatten)]
    report: RunReport,
    status: &'static str,
    user: UserId,
    created_at: DateTime<Utc>,
}

impl From<StoredRun> for RunView {
    fn from(run: StoredRun) -> Self {
        Self {
            status: store::runs::status(&run.report.outcome),
            report: run.report,
            user: run.user,
            created_at: run.created_at,
        }
    }
}

async fn get_run(
    State(state): State<AppState>,
    User(user): User,
    Path(id): Path<String>,
) -> Result<Json<RunView>, ApiError> {
    let not_found = || ApiError::not_found("Run not found.");
    let run_id = Uuid::parse_str(&id)
        .map(RunId::from)
        .map_err(|_| not_found())?;
    let run = state.runs.get(run_id).await.map_err(|error| {
        tracing::error!(%run_id, error = format!("{error:#}"), "run not loaded");
        ApiError::internal("run_store_unavailable", "The run could not be loaded.")
    })?;
    match run {
        // Someone else's run is indistinguishable from a missing one.
        Some(run) if run.user == user => Ok(Json(run.into())),
        _ => Err(not_found()),
    }
}

async fn unknown_endpoint() -> ApiError {
    ApiError::not_found("Unknown endpoint.")
}
