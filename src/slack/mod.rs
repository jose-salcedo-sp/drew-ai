//! Developer-facing Slack bot over Socket Mode, sharing the HTTP API's answerer and run store.

pub mod api;
#[cfg(any(test, feature = "testing"))]
pub mod fakes;
pub mod render;
pub mod socket;

#[cfg(test)]
mod tests;

use std::{sync::Arc, time::Duration};

use serde_json::Value;
use tokio::{
    sync::{Semaphore, watch},
    task::{JoinError, JoinSet},
};

use crate::{
    domain::{Question, UserId},
    pipeline::report::{Outcome, RunReport},
    server::AppState,
};

use self::{
    api::{SlackApi, SlackClient},
    socket::{Backoff, Connector, WebSocketConnector},
};

#[derive(Debug, clap::Args)]
pub struct SlackArgs {
    #[arg(long, env = "SLACK_APP_TOKEN", hide_env_values = true)]
    pub app_token: Option<String>,
    #[arg(long, env = "SLACK_BOT_TOKEN", hide_env_values = true)]
    pub bot_token: Option<String>,
}

pub struct Bot {
    api: Arc<dyn SlackApi>,
    connector: Arc<dyn Connector>,
}

impl Bot {
    /// An empty token counts as unset, as in `.env.example`.
    pub fn from_args(args: &SlackArgs) -> eros::Result<Option<Self>> {
        let set = |token: &Option<String>| token.clone().filter(|t| !t.trim().is_empty());
        match (set(&args.app_token), set(&args.bot_token)) {
            (None, None) => Ok(None),
            (Some(app), Some(bot)) => Ok(Some(Self::new(
                Arc::new(SlackClient::new(app, bot)?),
                Arc::new(WebSocketConnector),
            ))),
            _ => {
                eros::bail!("SLACK_APP_TOKEN and SLACK_BOT_TOKEN must both be set or both unset")
            }
        }
    }

    pub fn new(api: Arc<dyn SlackApi>, connector: Arc<dyn Connector>) -> Self {
        Self { api, connector }
    }

    /// Reconnects until `shutdown` turns true or its sender is dropped, then waits up to
    /// `DRAIN_TIMEOUT` for in-flight events.
    pub async fn run(self, state: AppState, mut shutdown: watch::Receiver<bool>) {
        let mut events = EventTasks::new(MAX_CONCURRENT_EVENTS);
        let mut backoff = Backoff::default();
        loop {
            let on_event = |event| {
                let state = state.clone();
                let api = Arc::clone(&self.api);
                events.spawn(async move { handle_event(&state, api.as_ref(), event).await });
            };
            let ended = tokio::select! {
                () = stopped(&mut shutdown) => break,
                ended = self.session(&mut backoff, on_event) => ended,
            };
            events.reap();
            let delay = match ended {
                Ok(()) => continue,
                Err(error) => {
                    tracing::warn!(error = format!("{error:#?}"), "slack socket reconnecting");
                    backoff.next_delay()
                }
            };
            tokio::select! {
                () = stopped(&mut shutdown) => break,
                () = tokio::time::sleep(delay) => {}
            }
        }
        events.drain(DRAIN_TIMEOUT).await;
    }

    async fn session(
        &self,
        backoff: &mut Backoff,
        on_event: impl FnMut(Value) + Send,
    ) -> eros::Result<()> {
        let url = self.api.open_connection().await?;
        let connection = self.connector.connect(&url).await?;
        socket::run_session(connection, backoff, on_event).await
    }
}

async fn stopped(shutdown: &mut watch::Receiver<bool>) {
    // An `Err` means the sender is gone, which also stops the bot.
    let _ = shutdown.wait_for(|stop| *stop).await;
}

/// Each event runs a whole pipeline, which shares the database pool with the HTTP API.
const MAX_CONCURRENT_EVENTS: usize = 4;
/// Fits, with the HTTP drain alongside it, in Kubernetes' default 30 s termination grace period.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(20);

/// Event tasks: every one is tracked, and at most `limit` run at once while the rest wait.
struct EventTasks {
    tasks: JoinSet<eros::Result<()>>,
    permits: Arc<Semaphore>,
}

impl EventTasks {
    fn new(limit: usize) -> Self {
        Self {
            tasks: JoinSet::new(),
            permits: Arc::new(Semaphore::new(limit)),
        }
    }

    fn spawn(&mut self, task: impl Future<Output = eros::Result<()>> + Send + 'static) {
        self.reap();
        let permits = Arc::clone(&self.permits);
        self.tasks.spawn(async move {
            let _permit = permits
                .acquire_owned()
                .await
                .expect("the event semaphore is never closed");
            task.await
        });
    }

    fn reap(&mut self) {
        while let Some(finished) = self.tasks.try_join_next() {
            report(finished);
        }
    }

    async fn drain(mut self, within: Duration) {
        let all_done = async {
            while let Some(finished) = self.tasks.join_next().await {
                report(finished);
            }
        };
        if tokio::time::timeout(within, all_done).await.is_err() {
            tracing::warn!(
                abandoned = self.tasks.len(),
                "slack events still running at shutdown were abandoned after {within:?}"
            );
            self.tasks.abort_all();
        }
    }
}

fn report(finished: Result<eros::Result<()>, JoinError>) {
    match finished {
        Ok(Ok(())) => {}
        Ok(Err(error)) => tracing::warn!(error = format!("{error:#?}"), "slack event failed"),
        Err(error) if error.is_panic() => {
            tracing::error!(error = format!("{error:#?}"), "slack event task panicked");
        }
        Err(error) => tracing::warn!(error = format!("{error:#?}"), "slack event task cancelled"),
    }
}

pub(crate) async fn handle_event(
    state: &AppState,
    api: &dyn SlackApi,
    event: Value,
) -> eros::Result<()> {
    if event.get("bot_id").is_some() || event.get("subtype").is_some() {
        return Ok(());
    }
    let Some(kind) = event.get("type").and_then(Value::as_str) else {
        return Ok(());
    };
    let (question, channel, thread_ts) = match kind {
        "app_mention" => {
            let text = event.get("text").and_then(Value::as_str).unwrap_or("");
            let question = strip_mentions(text);
            if question.is_empty() {
                return Ok(());
            }
            let channel = event
                .get("channel")
                .and_then(Value::as_str)
                .ok_or_else(|| eros::error!("app_mention missing channel"))?;
            let thread = event
                .get("thread_ts")
                .or_else(|| event.get("ts"))
                .and_then(Value::as_str);
            (question, channel, thread)
        }
        "message" if event.get("channel_type").and_then(Value::as_str) == Some("im") => {
            let text = event.get("text").and_then(Value::as_str).unwrap_or("");
            if text.trim().is_empty() {
                return Ok(());
            }
            let channel = event
                .get("channel")
                .and_then(Value::as_str)
                .ok_or_else(|| eros::error!("dm missing channel"))?;
            let thread = event.get("thread_ts").and_then(Value::as_str);
            (text.trim().to_owned(), channel, thread)
        }
        _ => return Ok(()),
    };
    let slack_user = event
        .get("user")
        .and_then(Value::as_str)
        .ok_or_else(|| eros::error!("event missing user"))?;
    answer_in_slack(
        state,
        api,
        slack_user,
        channel,
        thread_ts,
        Question::new(question),
    )
    .await
}

async fn answer_in_slack(
    state: &AppState,
    api: &dyn SlackApi,
    slack_user: &str,
    channel: &str,
    thread_ts: Option<&str>,
    question: Question,
) -> eros::Result<()> {
    let placeholder_ts = api
        .post_message(channel, thread_ts, "Looking that up...")
        .await?;
    let report = state.answerer.answer(question).await;
    let user = UserId(format!("slack:{slack_user}"));
    let text = persist_and_render(state, &user, &report).await;
    api.update_message(channel, &placeholder_ts, &text).await?;
    Ok(())
}

/// Slack replies go to shared channels, so a failure shows only the run id; the detail goes to the log.
async fn persist_and_render(state: &AppState, user: &UserId, report: &RunReport) -> String {
    let run_id = report.id;
    if let Err(error) = state.runs.save(user, report).await {
        tracing::error!(%run_id, error = format!("{error:#?}"), "slack run not persisted");
        return format!("Run {run_id} could not be completed.");
    }
    let id = run_id.to_string();
    match &report.outcome {
        Outcome::Answered(answer) => render::answer(answer, &id),
        Outcome::Declined { sources, .. } => render::declined(sources, &id),
        Outcome::Failed { at, error } => {
            tracing::error!(%run_id, at, error, "slack run failed");
            format!("Run {id} failed at {at}.")
        }
    }
}

/// Removes Slack user mentions (`<@U…>`) from mention text.
pub fn strip_mentions(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '<' {
            let mut token = String::from("<");
            while let Some(&next) = chars.peek() {
                token.push(next);
                chars.next();
                if next == '>' {
                    break;
                }
            }
            if token.starts_with("<@") {
                continue;
            }
            out.push_str(&token);
        } else {
            out.push(c);
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}
