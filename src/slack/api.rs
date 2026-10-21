//! Slack Web API client for Socket Mode and chat updates.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::{Client as HttpClient, StatusCode};
use serde::{Deserialize, de::DeserializeOwned};

const SLACK_API: &str = "https://slack.com/api/";
/// Web API methods answer in about a second; this bounds a hung call so its event task ends.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

#[async_trait]
pub trait SlackApi: Send + Sync {
    async fn open_connection(&self) -> eros::Result<String>;

    async fn post_message(
        &self,
        channel: &str,
        thread_ts: Option<&str>,
        text: &str,
    ) -> eros::Result<String>;

    async fn update_message(&self, channel: &str, ts: &str, text: &str) -> eros::Result<()>;
}

pub struct SlackClient {
    http: HttpClient,
    base_url: String,
    app_token: String,
    bot_token: String,
}

impl SlackClient {
    pub fn new(app_token: String, bot_token: String) -> eros::Result<Self> {
        Ok(Self {
            http: http_client()?,
            base_url: SLACK_API.into(),
            app_token,
            bot_token,
        })
    }

    /// Sends every call under `base_url` (ending in `/`) instead of `https://slack.com/api/`.
    pub fn with_base_url(self, base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            ..self
        }
    }

    /// Calls a Web API method. A non-200 status or `ok: false` is an error naming Slack's reason.
    #[eros::context("Slack {}", method)]
    async fn call<T: DeserializeOwned>(
        &self,
        method: &str,
        token: &str,
        form: &[(&str, &str)],
    ) -> eros::Result<T> {
        let response = self
            .http
            .post(format!("{}{method}", self.base_url))
            .bearer_auth(token)
            .form(form)
            .send()
            .await?;
        let status = response.status();
        let body = response.text().await?;
        eros::ensure!(status == StatusCode::OK, "returned {status}: {body}");
        let reply: ApiOk = serde_json::from_str(&body)?;
        if !reply.ok {
            eros::bail!("{}", reply.error.as_deref().unwrap_or("unknown error"));
        }
        Ok(serde_json::from_str(&body)?)
    }
}

#[eros::context("build the Slack HTTP client")]
fn http_client() -> eros::Result<HttpClient> {
    Ok(HttpClient::builder()
        .timeout(REQUEST_TIMEOUT)
        .connect_timeout(CONNECT_TIMEOUT)
        .build()?)
}

#[derive(Deserialize)]
struct ApiOk {
    ok: bool,
    error: Option<String>,
}

#[async_trait]
impl SlackApi for SlackClient {
    async fn open_connection(&self) -> eros::Result<String> {
        #[derive(Deserialize)]
        struct Open {
            url: Option<String>,
        }
        let open: Open = self
            .call("apps.connections.open", &self.app_token, &[])
            .await?;
        open.url
            .ok_or_else(|| eros::error!("apps.connections.open missing url"))
    }

    async fn post_message(
        &self,
        channel: &str,
        thread_ts: Option<&str>,
        text: &str,
    ) -> eros::Result<String> {
        #[derive(Deserialize)]
        struct Posted {
            ts: Option<String>,
        }
        let mut form = vec![("channel", channel), ("text", text)];
        if let Some(ts) = thread_ts {
            form.push(("thread_ts", ts));
        }
        let posted: Posted = self
            .call("chat.postMessage", &self.bot_token, &form)
            .await?;
        posted
            .ts
            .ok_or_else(|| eros::error!("chat.postMessage missing ts"))
    }

    async fn update_message(&self, channel: &str, ts: &str, text: &str) -> eros::Result<()> {
        let form = [("channel", channel), ("ts", ts), ("text", text)];
        self.call::<serde::de::IgnoredAny>("chat.update", &self.bot_token, &form)
            .await?;
        Ok(())
    }
}
