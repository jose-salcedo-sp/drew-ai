//! Slack Web API client for Socket Mode and chat updates.

use async_trait::async_trait;
use reqwest::{Client as HttpClient, StatusCode};
use serde::Deserialize;

const SLACK_API: &str = "https://slack.com/api/";

#[async_trait]
pub trait SlackApi: Send + Sync {
    async fn open_connection(&self) -> anyhow::Result<String>;

    async fn post_message(
        &self,
        channel: &str,
        thread_ts: Option<&str>,
        text: &str,
    ) -> anyhow::Result<String>;

    async fn update_message(&self, channel: &str, ts: &str, text: &str) -> anyhow::Result<()>;
}

pub struct SlackClient {
    http: HttpClient,
    app_token: String,
    bot_token: String,
}

impl SlackClient {
    pub fn new(app_token: String, bot_token: String) -> Self {
        Self {
            http: HttpClient::new(),
            app_token,
            bot_token,
        }
    }
}

#[derive(Deserialize)]
struct ApiOk {
    ok: bool,
    error: Option<String>,
}

#[async_trait]
impl SlackApi for SlackClient {
    async fn open_connection(&self) -> anyhow::Result<String> {
        #[derive(Deserialize)]
        struct Open {
            #[serde(flatten)]
            base: ApiOk,
            url: Option<String>,
        }
        let body: Open = self
            .http
            .post(format!("{SLACK_API}apps.connections.open"))
            .bearer_auth(&self.app_token)
            .send()
            .await?
            .json()
            .await?;
        if !body.base.ok {
            anyhow::bail!(
                "apps.connections.open: {}",
                body.base.error.unwrap_or_else(|| "unknown".into())
            );
        }
        body.url
            .ok_or_else(|| anyhow::anyhow!("apps.connections.open missing url"))
    }

    async fn post_message(
        &self,
        channel: &str,
        thread_ts: Option<&str>,
        text: &str,
    ) -> anyhow::Result<String> {
        #[derive(Deserialize)]
        struct Posted {
            #[serde(flatten)]
            base: ApiOk,
            ts: Option<String>,
        }
        let mut form = vec![("channel", channel), ("text", text)];
        if let Some(ts) = thread_ts {
            form.push(("thread_ts", ts));
        }
        let response = self
            .http
            .post(format!("{SLACK_API}chat.postMessage"))
            .bearer_auth(&self.bot_token)
            .form(&form)
            .send()
            .await?;
        let status = response.status();
        let body: Posted = response.json().await?;
        if status != StatusCode::OK || !body.base.ok {
            anyhow::bail!(
                "chat.postMessage: {}",
                body.base.error.unwrap_or_else(|| status.to_string())
            );
        }
        body.ts
            .ok_or_else(|| anyhow::anyhow!("chat.postMessage missing ts"))
    }

    async fn update_message(&self, channel: &str, ts: &str, text: &str) -> anyhow::Result<()> {
        #[derive(Deserialize)]
        struct Updated {
            #[serde(flatten)]
            base: ApiOk,
        }
        let form = [("channel", channel), ("ts", ts), ("text", text)];
        let response = self
            .http
            .post(format!("{SLACK_API}chat.update"))
            .bearer_auth(&self.bot_token)
            .form(&form)
            .send()
            .await?;
        let status = response.status();
        let body: Updated = response.json().await?;
        if status != StatusCode::OK || !body.base.ok {
            anyhow::bail!(
                "chat.update: {}",
                body.base.error.unwrap_or_else(|| status.to_string())
            );
        }
        Ok(())
    }
}
