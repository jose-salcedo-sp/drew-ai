//! Developer-facing Slack bot over Socket Mode, sharing the HTTP API's answerer and run store.

pub mod api;
pub mod fakes;
pub mod render;
pub mod socket;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use serde_json::Value;

use crate::{
    domain::{Question, UserId},
    pipeline::report::{Outcome, RunReport},
    server::AppState,
};

use self::api::{SlackApi, SlackClient};

#[derive(Debug, clap::Args)]
pub struct SlackArgs {
    #[arg(long, env = "SLACK_APP_TOKEN", hide_env_values = true)]
    pub app_token: Option<String>,
    #[arg(long, env = "SLACK_BOT_TOKEN", hide_env_values = true)]
    pub bot_token: Option<String>,
}

pub struct Bot {
    api: Arc<dyn SlackApi>,
}

impl Bot {
    pub fn from_args(args: &SlackArgs) -> anyhow::Result<Option<Self>> {
        match (&args.app_token, &args.bot_token) {
            (None, None) => Ok(None),
            (Some(app), Some(bot)) => Ok(Some(Self {
                api: Arc::new(SlackClient::new(app.clone(), bot.clone())),
            })),
            _ => {
                anyhow::bail!("SLACK_APP_TOKEN and SLACK_BOT_TOKEN must both be set or both unset")
            }
        }
    }

    pub fn with_api(api: Arc<dyn SlackApi>) -> Self {
        Self { api }
    }

    pub async fn run(
        self,
        state: AppState,
        shutdown: tokio::sync::watch::Receiver<bool>,
    ) -> anyhow::Result<()> {
        let api = self.api.clone();
        let mut backoff = std::time::Duration::from_secs(1);
        loop {
            if *shutdown.borrow() {
                return Ok(());
            }
            let session_state = state.clone();
            let api = api.clone();
            let shutdown_rx = shutdown.clone();
            let result = socket::run_socket(
                api.clone(),
                move |event| {
                    let state = session_state.clone();
                    let api = api.clone();
                    tokio::spawn(async move {
                        if let Err(error) = handle_event(&state, api.as_ref(), event).await {
                            tracing::warn!(error = format!("{error:#}"), "slack event failed");
                        }
                    });
                },
                shutdown_rx,
            )
            .await;
            match result {
                Ok(()) => return Ok(()),
                Err(error) => {
                    tracing::warn!(error = format!("{error:#}"), "slack socket reconnecting");
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(std::time::Duration::from_secs(60));
                }
            }
        }
    }
}

pub(crate) async fn handle_event(
    state: &AppState,
    api: &dyn SlackApi,
    event: Value,
) -> anyhow::Result<()> {
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
                .ok_or_else(|| anyhow::anyhow!("app_mention missing channel"))?;
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
                .ok_or_else(|| anyhow::anyhow!("dm missing channel"))?;
            let thread = event.get("thread_ts").and_then(Value::as_str);
            (text.trim().to_owned(), channel, thread)
        }
        _ => return Ok(()),
    };
    let slack_user = event
        .get("user")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("event missing user"))?;
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
) -> anyhow::Result<()> {
    let placeholder_ts = api
        .post_message(channel, thread_ts, "Looking that up...")
        .await?;
    let report = state.answerer.answer(question).await;
    let run_id = report.id;
    let user = UserId(format!("slack:{slack_user}"));
    let text = match persist_and_render(state, &user, &report).await {
        Ok(content) => content,
        Err(error) => {
            tracing::error!(%run_id, error = format!("{error:#}"), "slack run failed");
            format!("Run {run_id} could not be completed: {error:#}")
        }
    };
    api.update_message(channel, &placeholder_ts, &text).await?;
    Ok(())
}

async fn persist_and_render(
    state: &AppState,
    user: &UserId,
    report: &RunReport,
) -> anyhow::Result<String> {
    state.runs.save(user, report).await?;
    let id = report.id.to_string();
    Ok(match &report.outcome {
        Outcome::Answered(answer) => render::answer(answer, &id),
        Outcome::Declined { sources, .. } => render::declined(sources, &id),
        Outcome::Failed { at, error } => format!("Run {id} failed at {at}: {error}"),
    })
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
