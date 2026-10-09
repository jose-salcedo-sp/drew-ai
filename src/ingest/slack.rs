//! Slack public channels named by Drew: one thread is one unit; Drew's messages are authoritative.

use std::{
    collections::{BTreeMap, HashMap},
    convert::Infallible,
    fmt,
    str::FromStr,
    sync::{Mutex, PoisonError},
    time::Duration,
};

use anyhow::{Context, bail};
use async_trait::async_trait;
use chrono::{DateTime, TimeDelta, Utc};
use reqwest::{StatusCode, header::RETRY_AFTER};
use serde::{Deserialize, de::DeserializeOwned};

use crate::domain::{ChunkDraft, Document, Labels, Sensitivity, SourceKind};

use super::Source;

const PAGE_LIMIT: &str = "200";
const MAX_ATTEMPTS: u32 = 5;
const TITLE_CHARS: usize = 80;
/// Channel housekeeping and deleted-message placeholders; they carry no content.
const NOISE_SUBTYPES: &[&str] = &[
    "channel_join",
    "channel_leave",
    "bot_add",
    "bot_remove",
    "channel_topic",
    "channel_purpose",
    "channel_name",
    "channel_archive",
    "channel_unarchive",
    "tombstone",
];

#[derive(Debug, clap::Args)]
pub struct Args {
    /// Bot token with the `channels:read`, `channels:history`, and `users:read` scopes.
    #[arg(long, env = "SLACK_BOT_TOKEN", hide_env_values = true)]
    pub slack_bot_token: Option<Token>,
    /// Comma-separated ids of the public channels to ingest.
    #[arg(long, env = "SLACK_CHANNELS")]
    pub slack_channels: Option<String>,
    /// Workspace URL used for permalinks, e.g. `https://sailpoint.slack.com`.
    #[arg(long, env = "SLACK_WORKSPACE_URL")]
    pub slack_workspace_url: Option<String>,
    /// Drew's Slack user id.
    #[arg(long, env = "DREW_SLACK_USER_ID")]
    pub drew_slack_user_id: Option<String>,
    /// How far before `since` to rescan history for threads with new replies.
    #[arg(long, env = "SLACK_LOOKBACK_DAYS", default_value_t = 30)]
    pub slack_lookback_days: u32,
    #[arg(long, env = "SLACK_API_URL", default_value = "https://slack.com/api")]
    pub slack_api_url: String,
}

#[derive(Clone)]
pub struct Token(String);

impl FromStr for Token {
    type Err = Infallible;

    fn from_str(token: &str) -> Result<Self, Infallible> {
        Ok(Self(token.to_owned()))
    }
}

impl fmt::Debug for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Token(<redacted>)")
    }
}

pub struct Connector {
    client: reqwest::Client,
    api_url: String,
    token: Token,
    channels: Vec<String>,
    workspace_url: String,
    drew_user_id: String,
    lookback: TimeDelta,
    display_names: Mutex<HashMap<String, String>>,
}

impl Connector {
    /// `None` when this source is not configured.
    pub fn from_args(args: &Args) -> anyhow::Result<Option<Self>> {
        let non_empty = |value: &Option<String>| {
            value
                .as_deref()
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(str::to_owned)
        };
        let token = args
            .slack_bot_token
            .clone()
            .filter(|t| !t.0.trim().is_empty());
        let channels: Vec<String> = args
            .slack_channels
            .iter()
            .flat_map(|c| c.split(','))
            .map(str::trim)
            .filter(|c| !c.is_empty())
            .map(str::to_owned)
            .collect();
        let workspace_url = non_empty(&args.slack_workspace_url);
        let drew_user_id = non_empty(&args.drew_slack_user_id);

        let missing: Vec<&str> = [
            ("SLACK_BOT_TOKEN", token.is_none()),
            ("SLACK_CHANNELS", channels.is_empty()),
            ("SLACK_WORKSPACE_URL", workspace_url.is_none()),
            ("DREW_SLACK_USER_ID", drew_user_id.is_none()),
        ]
        .into_iter()
        .filter_map(|(name, missing)| missing.then_some(name))
        .collect();
        let (Some(token), Some(workspace_url), Some(drew_user_id)) =
            (token, workspace_url, drew_user_id)
        else {
            if missing.len() == 4 {
                return Ok(None);
            }
            bail!(
                "Slack is partially configured; also set {}",
                missing.join(", ")
            );
        };
        if channels.is_empty() {
            bail!("Slack is partially configured; also set SLACK_CHANNELS");
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("building the Slack HTTP client")?;
        Ok(Some(Self {
            client,
            api_url: args.slack_api_url.trim_end_matches('/').to_owned(),
            token,
            channels,
            workspace_url: workspace_url.trim_end_matches('/').to_owned(),
            drew_user_id,
            lookback: TimeDelta::days(i64::from(args.slack_lookback_days)),
            display_names: Mutex::default(),
        }))
    }

    async fn call<T: DeserializeOwned>(
        &self,
        method: &str,
        query: &[(&str, &str)],
    ) -> anyhow::Result<T> {
        let url = format!("{}/{method}", self.api_url);
        let mut attempt = 1;
        let response = loop {
            let response = self
                .client
                .get(&url)
                .bearer_auth(&self.token.0)
                .query(query)
                .send()
                .await
                .with_context(|| format!("calling Slack {method}"))?;
            if response.status() != StatusCode::TOO_MANY_REQUESTS || attempt == MAX_ATTEMPTS {
                break response;
            }
            let wait = retry_after(&response);
            tracing::warn!(
                method,
                attempt,
                wait_secs = wait.as_secs(),
                "Slack rate limit; retrying"
            );
            tokio::time::sleep(wait).await;
            attempt += 1;
        };
        let body = response
            .error_for_status()
            .with_context(|| format!("calling Slack {method}"))?
            .bytes()
            .await
            .with_context(|| format!("reading Slack {method} response"))?;
        let envelope: Envelope = serde_json::from_slice(&body)
            .with_context(|| format!("parsing Slack {method} response"))?;
        if !envelope.ok {
            return Err(ApiError {
                method: method.to_owned(),
                code: envelope.error.unwrap_or_else(|| "unknown_error".into()),
                needed: envelope.needed,
            }
            .into());
        }
        serde_json::from_slice(&body).with_context(|| format!("parsing Slack {method} response"))
    }

    /// Every message across all cursor pages of `conversations.history` or `conversations.replies`.
    async fn messages(
        &self,
        method: &str,
        params: &[(&str, &str)],
    ) -> anyhow::Result<Vec<Message>> {
        let mut messages = Vec::new();
        let mut cursor = String::new();
        loop {
            let mut query = params.to_vec();
            query.push(("limit", PAGE_LIMIT));
            if !cursor.is_empty() {
                query.push(("cursor", &cursor));
            }
            let page: MessagesPage = self.call(method, &query).await?;
            messages.extend(page.messages);
            match page.response_metadata.next_cursor {
                Some(next) if !next.is_empty() => cursor = next,
                _ => return Ok(messages),
            }
        }
    }

    async fn check_public(&self, channel: &str) -> anyhow::Result<()> {
        let info: ChannelInfo = match self
            .call("conversations.info", &[("channel", channel)])
            .await
        {
            Ok(info) => info,
            Err(error) if api_code(&error) == Some("channel_not_found") => bail!(
                "Slack channel {channel} was not found or is not visible to the bot; only public channels are supported"
            ),
            Err(error) => return Err(error),
        };
        let info = info.channel;
        let excluded = if info.is_im {
            Some("a direct message")
        } else if info.is_mpim {
            Some("a group direct message")
        } else if info.is_private || info.is_group || !info.is_channel {
            Some("a private channel")
        } else {
            None
        };
        if let Some(kind) = excluded {
            bail!("Slack channel {channel} is {kind}; only public channels can be ingested");
        }
        if info.is_member == Some(false) {
            bail!(
                "the Slack bot is not a member of #{} ({channel}); invite it to the channel",
                info.name.as_deref().unwrap_or(channel)
            );
        }
        Ok(())
    }

    async fn channel_documents(
        &self,
        channel: &str,
        since: Option<DateTime<Utc>>,
    ) -> anyhow::Result<Vec<Document>> {
        let oldest = since.map(|since| slack_ts(since - self.lookback));
        let mut params = vec![("channel", channel)];
        if let Some(oldest) = &oldest {
            params.push(("oldest", oldest));
        }
        let history = self.messages("conversations.history", &params).await?;

        let mut threads = BTreeMap::new();
        for parent in history {
            if parent.is_reply() {
                continue;
            }
            let posted = ts_time(&parent.ts)?;
            let messages = if parent.reply_count > 0 {
                let latest = match &parent.latest_reply {
                    Some(ts) => ts_time(ts)?,
                    None => posted,
                };
                if since.is_some_and(|since| latest <= since) {
                    continue;
                }
                self.messages(
                    "conversations.replies",
                    &[("channel", channel), ("ts", &parent.ts)],
                )
                .await?
            } else if parent.user.as_deref() == Some(&self.drew_user_id) && !parent.is_noise() {
                if since.is_some_and(|since| posted <= since) {
                    continue;
                }
                vec![parent.clone()]
            } else {
                continue;
            };
            if let Some(document) = self.thread_document(channel, &parent.ts, messages).await? {
                threads.insert(posted, document);
            }
        }
        Ok(threads.into_values().collect())
    }

    async fn thread_document(
        &self,
        channel: &str,
        thread_ts: &str,
        messages: Vec<Message>,
    ) -> anyhow::Result<Option<Document>> {
        // `conversations.replies` repeats the parent on every page.
        let mut by_time = BTreeMap::new();
        for message in messages.into_iter().filter(|m| !m.is_noise()) {
            by_time.insert(ts_time(&message.ts)?, message);
        }
        let Some(updated_at) = by_time.keys().next_back().copied() else {
            return Ok(None);
        };

        let mut users = Vec::new();
        for message in by_time.values() {
            users.extend(message.user.clone());
            users.extend(mentioned_users(&message.text).map(str::to_owned));
        }
        users.sort();
        users.dedup();
        let mut names = HashMap::new();
        for user in users {
            let name = self.display_name(&user).await?;
            names.insert(user, name);
        }

        let mut title = None;
        let mut lines = Vec::new();
        let mut authoritative = false;
        for message in by_time.values() {
            authoritative |= message.user.as_deref() == Some(&self.drew_user_id);
            let text = render(&message.text, &names);
            if text.is_empty() {
                continue;
            }
            let author = match &message.user {
                Some(user) => names[user].as_str(),
                None => message.bot_name(),
            };
            title.get_or_insert_with(|| truncate_title(&text));
            lines.push(format!("{author}: {text}"));
        }
        let Some(title) = title else {
            return Ok(None);
        };

        let url = format!(
            "{}/archives/{channel}/p{}",
            self.workspace_url,
            thread_ts.replace('.', "")
        );
        Ok(Some(Document {
            source: SourceKind::Slack,
            external_id: format!("{channel}:{thread_ts}"),
            title,
            url: url.clone(),
            updated_at,
            labels: Labels {
                acl: vec![format!("slack-channel:{channel}")],
                customer: None,
                sensitivity: Sensitivity::Internal,
            },
            chunks: vec![ChunkDraft {
                locator: "thread".into(),
                heading: None,
                url,
                text: lines.join("\n"),
                authoritative,
            }],
        }))
    }

    async fn display_name(&self, user: &str) -> anyhow::Result<String> {
        if user == self.drew_user_id {
            return Ok("Drew".into());
        }
        if let Some(name) = self.cached_names().get(user) {
            return Ok(name.clone());
        }
        let name = match self.call::<UserInfo>("users.info", &[("user", user)]).await {
            Ok(info) => info.user.display_name().unwrap_or(user).to_owned(),
            Err(error)
                if matches!(
                    api_code(&error),
                    Some("user_not_found" | "user_not_visible")
                ) =>
            {
                user.to_owned()
            }
            Err(error) => return Err(error),
        };
        self.cached_names().insert(user.to_owned(), name.clone());
        Ok(name)
    }

    fn cached_names(&self) -> std::sync::MutexGuard<'_, HashMap<String, String>> {
        self.display_names
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

#[async_trait]
impl Source for Connector {
    fn kind(&self) -> SourceKind {
        SourceKind::Slack
    }

    async fn fetch_changed(&self, since: Option<DateTime<Utc>>) -> anyhow::Result<Vec<Document>> {
        for channel in &self.channels {
            self.check_public(channel).await?;
        }
        let mut documents = Vec::new();
        for channel in &self.channels {
            documents.extend(
                self.channel_documents(channel, since)
                    .await
                    .with_context(|| format!("ingesting Slack channel {channel}"))?,
            );
        }
        Ok(documents)
    }
}

#[derive(Debug, thiserror::Error)]
#[error(
    "Slack {method} failed: {code}{}",
    .needed.as_ref().map(|scope| format!(" (missing scope {scope})")).unwrap_or_default()
)]
struct ApiError {
    method: String,
    code: String,
    needed: Option<String>,
}

fn api_code(error: &anyhow::Error) -> Option<&str> {
    error.downcast_ref::<ApiError>().map(|e| e.code.as_str())
}

fn retry_after(response: &reqwest::Response) -> Duration {
    let secs = response
        .headers()
        .get(RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(1);
    Duration::from_secs(secs)
}

#[derive(Deserialize)]
struct Envelope {
    ok: bool,
    error: Option<String>,
    needed: Option<String>,
}

#[derive(Deserialize)]
struct ChannelInfo {
    channel: ChannelFlags,
}

#[derive(Deserialize)]
struct ChannelFlags {
    name: Option<String>,
    #[serde(default)]
    is_channel: bool,
    #[serde(default)]
    is_group: bool,
    #[serde(default)]
    is_im: bool,
    #[serde(default)]
    is_mpim: bool,
    #[serde(default)]
    is_private: bool,
    is_member: Option<bool>,
}

#[derive(Deserialize)]
struct MessagesPage {
    messages: Vec<Message>,
    #[serde(default)]
    response_metadata: ResponseMetadata,
}

#[derive(Default, Deserialize)]
struct ResponseMetadata {
    next_cursor: Option<String>,
}

#[derive(Clone, Deserialize)]
struct Message {
    ts: String,
    #[serde(default)]
    text: String,
    user: Option<String>,
    subtype: Option<String>,
    thread_ts: Option<String>,
    #[serde(default)]
    reply_count: u32,
    latest_reply: Option<String>,
    username: Option<String>,
    bot_profile: Option<BotProfile>,
}

impl Message {
    /// A reply that also appears in channel history (`thread_broadcast`).
    fn is_reply(&self) -> bool {
        self.thread_ts.as_ref().is_some_and(|t| *t != self.ts)
    }

    fn is_noise(&self) -> bool {
        self.subtype
            .as_deref()
            .is_some_and(|s| NOISE_SUBTYPES.contains(&s))
    }

    fn bot_name(&self) -> &str {
        self.bot_profile
            .as_ref()
            .and_then(|b| b.name.as_deref())
            .or(self.username.as_deref())
            .filter(|n| !n.is_empty())
            .unwrap_or("bot")
    }
}

#[derive(Clone, Deserialize)]
struct BotProfile {
    name: Option<String>,
}

#[derive(Deserialize)]
struct UserInfo {
    user: User,
}

#[derive(Deserialize)]
struct User {
    name: Option<String>,
    real_name: Option<String>,
    #[serde(default)]
    profile: Profile,
}

#[derive(Default, Deserialize)]
struct Profile {
    display_name: Option<String>,
    real_name: Option<String>,
}

impl User {
    fn display_name(&self) -> Option<&str> {
        [
            &self.profile.display_name,
            &self.profile.real_name,
            &self.real_name,
            &self.name,
        ]
        .into_iter()
        .filter_map(|n| n.as_deref())
        .find(|n| !n.trim().is_empty())
    }
}

/// Parses `"1790899200.000100"` (seconds and microseconds) into a time.
fn ts_time(ts: &str) -> anyhow::Result<DateTime<Utc>> {
    let parse = || {
        let (secs, frac) = ts.split_once('.').unwrap_or((ts, ""));
        let micros: u32 = format!("{frac:0<6}").get(..6)?.parse().ok()?;
        DateTime::from_timestamp(secs.parse().ok()?, micros * 1_000)
    };
    parse().with_context(|| format!("invalid Slack timestamp {ts:?}"))
}

fn slack_ts(time: DateTime<Utc>) -> String {
    format!("{}.{:06}", time.timestamp(), time.timestamp_subsec_micros())
}

enum Piece<'a> {
    Text(&'a str),
    /// The inside of `<...>`.
    Markup(&'a str),
}

/// Slack escapes literal `<`, `>`, and `&`, so every `<` opens markup.
fn pieces(mut text: &str) -> Vec<Piece<'_>> {
    let mut pieces = Vec::new();
    while let Some(start) = text.find('<') {
        let Some(len) = text[start..].find('>') else {
            break;
        };
        if start > 0 {
            pieces.push(Piece::Text(&text[..start]));
        }
        pieces.push(Piece::Markup(&text[start + 1..start + len]));
        text = &text[start + len + 1..];
    }
    if !text.is_empty() {
        pieces.push(Piece::Text(text));
    }
    pieces
}

fn mentioned_users(text: &str) -> impl Iterator<Item = &str> {
    pieces(text).into_iter().filter_map(|piece| match piece {
        Piece::Markup(markup) => markup
            .strip_prefix('@')
            .map(|m| m.split('|').next().unwrap_or(m)),
        Piece::Text(_) => None,
    })
}

/// Slack mrkdwn to plain text, with user mentions replaced by `names`.
fn render(text: &str, names: &HashMap<String, String>) -> String {
    let mut out = String::new();
    for piece in pieces(text) {
        match piece {
            Piece::Text(text) => out.push_str(&unescape(text)),
            Piece::Markup(markup) => out.push_str(&render_markup(markup, names)),
        }
    }
    out.trim().to_owned()
}

fn render_markup(markup: &str, names: &HashMap<String, String>) -> String {
    let (target, label) = match markup.split_once('|') {
        Some((target, label)) => (target, Some(unescape(label)).filter(|l| !l.is_empty())),
        None => (markup, None),
    };
    if let Some(user) = target.strip_prefix('@') {
        return format!("@{}", names.get(user).map_or(user, String::as_str));
    }
    if let Some(channel) = target.strip_prefix('#') {
        return format!("#{}", label.unwrap_or_else(|| channel.to_owned()));
    }
    if let Some(special) = target.strip_prefix('!') {
        return match special {
            "here" | "channel" | "everyone" => format!("@{special}"),
            _ => label
                .unwrap_or_else(|| format!("@{}", special.split('^').next().unwrap_or(special))),
        };
    }
    let url = unescape(target);
    if let Some(address) = url.strip_prefix("mailto:") {
        return label.unwrap_or_else(|| address.to_owned());
    }
    let without_scheme = url.split_once("://").map_or(url.as_str(), |(_, rest)| rest);
    match label {
        Some(label) if label != url && label != without_scheme => format!("{label} ({url})"),
        _ => url,
    }
}

fn unescape(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

fn truncate_title(text: &str) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.chars().count() <= TITLE_CHARS {
        return text;
    }
    let cut: String = text.chars().take(TITLE_CHARS - 1).collect();
    format!("{}…", cut.trim_end())
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use serde_json::{Value, json};
    use wiremock::{
        Mock, MockBuilder, MockServer, ResponseTemplate,
        matchers::{header, method, path, query_param, query_param_is_missing},
    };

    use super::*;

    const CHANNEL: &str = "C0SAPHELP";
    const HISTORY_PAGE_2: &str = "bmV4dF90czoxNzkwMjk0NDAw";
    const AGR_USERS_PAGE_2: &str = "cmVwbGllczI=";
    /// 2026-10-01T00:00:00Z
    const SINCE: i64 = 1_790_812_800;
    /// `SINCE` minus the default 30-day lookback.
    const OLDEST: &str = "1788220800.000000";

    fn fixture(name: &str) -> Value {
        let path = format!(
            "{}/tests/fixtures/slack/{name}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        serde_json::from_str(&text).unwrap()
    }

    fn slack(api_method: &str) -> MockBuilder {
        Mock::given(method("GET"))
            .and(path(format!("/{api_method}")))
            .and(header("authorization", "Bearer xoxb-test"))
    }

    fn ok(body: Value) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(body)
    }

    fn unconfigured() -> Args {
        Args {
            slack_bot_token: None,
            slack_channels: None,
            slack_workspace_url: None,
            drew_slack_user_id: None,
            slack_lookback_days: 30,
            slack_api_url: "https://slack.com/api".into(),
        }
    }

    fn args(server: &MockServer) -> Args {
        Args {
            slack_bot_token: Some("xoxb-test".parse().unwrap()),
            slack_channels: Some(CHANNEL.into()),
            slack_workspace_url: Some("https://sailpoint.slack.com/".into()),
            drew_slack_user_id: Some("U0DREW".into()),
            slack_api_url: server.uri(),
            ..unconfigured()
        }
    }

    fn connector(args: &Args) -> Connector {
        Connector::from_args(args).unwrap().unwrap()
    }

    fn time(ts: &str) -> DateTime<Utc> {
        ts_time(ts).unwrap()
    }

    /// Mounts the `sap-help` channel; history must be requested with exactly `oldest`.
    async fn mount_channel(server: &MockServer, oldest: Option<&str>) {
        slack("conversations.info")
            .and(query_param("channel", CHANNEL))
            .respond_with(ok(fixture("conversations.info")))
            .mount(server)
            .await;

        for (cursor, page) in [
            (None, "conversations.history.page1"),
            (Some(HISTORY_PAGE_2), "conversations.history.page2"),
        ] {
            let mock = slack("conversations.history")
                .and(query_param("channel", CHANNEL))
                .and(query_param("limit", "200"));
            let mock = match cursor {
                Some(cursor) => mock.and(query_param("cursor", cursor)),
                None => mock.and(query_param_is_missing("cursor")),
            };
            let mock = match oldest {
                Some(oldest) => mock.and(query_param("oldest", oldest)),
                None => mock.and(query_param_is_missing("oldest")),
            };
            mock.respond_with(ok(fixture(page)))
                .expect(1)
                .mount(server)
                .await;
        }

        let replies = |ts: &str| {
            slack("conversations.replies")
                .and(query_param("channel", CHANNEL))
                .and(query_param("ts", ts))
        };
        replies("1790899200.000100")
            .and(query_param_is_missing("cursor"))
            .respond_with(ok(fixture("conversations.replies.agr-users.page1")))
            .expect(1)
            .mount(server)
            .await;
        replies("1790899200.000100")
            .and(query_param("cursor", AGR_USERS_PAGE_2))
            .respond_with(ok(fixture("conversations.replies.agr-users.page2")))
            .expect(1)
            .mount(server)
            .await;
        replies("1790294400.000500")
            .respond_with(ok(fixture("conversations.replies.su53")))
            .expect(1)
            .mount(server)
            .await;
        replies("1790200000.000700")
            .respond_with(ok(fixture("conversations.replies.role-buffer")))
            .expect(u64::from(oldest.is_none()))
            .mount(server)
            .await;

        for (user, name) in [("U0ALICE", "alice"), ("U0BOB", "bob")] {
            slack("users.info")
                .and(query_param("user", user))
                .respond_with(ok(fixture(&format!("users.info.{name}"))))
                .expect(1)
                .mount(server)
                .await;
        }
    }

    #[test]
    fn unconfigured_is_none() {
        assert!(Connector::from_args(&unconfigured()).unwrap().is_none());
        let blank = Args {
            slack_bot_token: Some("".parse().unwrap()),
            slack_channels: Some(" , ".into()),
            ..unconfigured()
        };
        assert!(Connector::from_args(&blank).unwrap().is_none());
    }

    #[test]
    fn partial_config_names_what_is_missing() {
        let token_only = Args {
            slack_bot_token: Some("xoxb-test".parse().unwrap()),
            ..unconfigured()
        };
        let error = Connector::from_args(&token_only).err().unwrap().to_string();
        assert!(
            error.contains("SLACK_CHANNELS, SLACK_WORKSPACE_URL, DREW_SLACK_USER_ID"),
            "{error}"
        );

        let no_channels = Args {
            slack_bot_token: Some("xoxb-test".parse().unwrap()),
            slack_channels: Some(",".into()),
            slack_workspace_url: Some("https://sailpoint.slack.com".into()),
            drew_slack_user_id: Some("U0DREW".into()),
            ..unconfigured()
        };
        let error = Connector::from_args(&no_channels)
            .err()
            .unwrap()
            .to_string();
        assert!(error.ends_with("also set SLACK_CHANNELS"), "{error}");
    }

    #[test]
    fn cli_flags_configure_the_connector() {
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            slack: Args,
        }
        let cli = Cli::try_parse_from([
            "drew",
            "--slack-bot-token",
            "xoxb-secret",
            "--slack-channels",
            "C1, C2",
            "--slack-workspace-url",
            "https://sailpoint.slack.com",
            "--drew-slack-user-id",
            "U0DREW",
            "--slack-lookback-days",
            "7",
        ])
        .unwrap();
        assert!(!format!("{:?}", cli.slack).contains("xoxb-secret"));
        let connector = connector(&cli.slack);
        assert_eq!(connector.channels, ["C1", "C2"]);
        assert_eq!(connector.lookback, TimeDelta::days(7));
    }

    #[tokio::test]
    async fn since_rescans_the_lookback_window_and_keeps_threads_with_new_replies() {
        let server = MockServer::start().await;
        mount_channel(&server, Some(OLDEST)).await;

        let since = DateTime::from_timestamp(SINCE, 0).unwrap();
        let documents = connector(&args(&server))
            .fetch_changed(Some(since))
            .await
            .unwrap();

        let ids: Vec<_> = documents.iter().map(|d| d.external_id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "C0SAPHELP:1790294400.000500",
                "C0SAPHELP:1790899200.000100",
                "C0SAPHELP:1791072000.000100",
            ]
        );

        let [su53, agr_users, reminder] = documents.as_slice() else {
            unreachable!()
        };

        assert_eq!(
            agr_users.url,
            "https://sailpoint.slack.com/archives/C0SAPHELP/p1790899200000100"
        );
        assert_eq!(agr_users.updated_at, time("1790985600.000400"));
        assert_eq!(agr_users.title.chars().count(), TITLE_CHARS);
        assert!(
            agr_users
                .title
                .starts_with("How does @Bob Builder map roles in AGR_USERS? See #basis & SAP Help"),
            "{}",
            agr_users.title
        );
        assert!(agr_users.title.ends_with('…'));
        assert_eq!(
            agr_users.labels,
            Labels {
                acl: vec!["slack-channel:C0SAPHELP".into()],
                customer: None,
                sensitivity: Sensitivity::Internal,
            }
        );
        assert_eq!(
            agr_users.chunks,
            [ChunkDraft {
                locator: "thread".into(),
                heading: None,
                url: agr_users.url.clone(),
                text: "alice: How does @Bob Builder map roles in AGR_USERS? See #basis & SAP Help (https://help.sap.com/agr)\n\
                       Drew: @alice AGR_USERS is filled by the PFCG user comparison (PFUD).\n\
                       Drew: Also see SAP Note 123.\n\
                       Bob Builder: thanks @Drew!"
                    .into(),
                authoritative: true,
            }]
        );

        assert_eq!(su53.updated_at, time("1790900000.000600"));
        assert!(!su53.chunks[0].authoritative);
        assert_eq!(
            su53.chunks[0].text,
            "Bob Builder: Is SU53 reliable for missing authorizations?\n\
             Jira Cloud: Linked ARM-1 (https://jira.example.com/browse/ARM-1)\n\
             alice: Mostly. Trace with ST01 when it lies."
        );

        assert_eq!(
            reminder.title,
            "Reminder: `S_TCODE` runs before `S_PROGRAM` & is cached <per session>."
        );
        assert_eq!(reminder.updated_at, time("1791072000.000100"));
        assert!(reminder.chunks[0].authoritative);
        assert_eq!(
            reminder.chunks[0].text,
            "Drew: Reminder: `S_TCODE` runs before `S_PROGRAM` & is cached <per session>."
        );
    }

    #[tokio::test]
    async fn full_fetch_includes_every_thread_and_drews_standalone_messages() {
        let server = MockServer::start().await;
        mount_channel(&server, None).await;

        let documents = connector(&args(&server)).fetch_changed(None).await.unwrap();

        let ids: Vec<_> = documents.iter().map(|d| d.external_id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "C0SAPHELP:1790000000.001000",
                "C0SAPHELP:1790200000.000700",
                "C0SAPHELP:1790294400.000500",
                "C0SAPHELP:1790899200.000100",
                "C0SAPHELP:1791072000.000100",
            ]
        );
        assert_eq!(documents[0].chunks[0].text, "Drew: USR02 holds logon data.");
        assert_eq!(
            documents[1].chunks[0].text,
            "Bob Builder: Where is the role buffer kept?\n\
             Drew: In the user buffer; refresh it with SU56."
        );
        assert!(documents[1].chunks[0].authoritative);
    }

    #[tokio::test]
    async fn rejects_channels_that_are_not_public() {
        let cases = [
            (
                json!({"ok": true, "channel": {"id": "G0X", "is_channel": false, "is_group": true, "is_private": true}}),
                "G0X is a private channel",
            ),
            (
                json!({"ok": true, "channel": {"id": "G0X", "is_channel": true, "is_private": true}}),
                "G0X is a private channel",
            ),
            (
                json!({"ok": true, "channel": {"id": "G0X", "is_im": true, "is_private": true}}),
                "G0X is a direct message",
            ),
            (
                json!({"ok": true, "channel": {"id": "G0X", "is_mpim": true, "is_group": true, "is_private": true}}),
                "G0X is a group direct message",
            ),
            (
                json!({"ok": false, "error": "channel_not_found"}),
                "G0X was not found or is not visible to the bot",
            ),
        ];
        for (info, expected) in cases {
            let server = MockServer::start().await;
            slack("conversations.info")
                .and(query_param("channel", CHANNEL))
                .respond_with(ok(fixture("conversations.info")))
                .mount(&server)
                .await;
            slack("conversations.info")
                .and(query_param("channel", "G0X"))
                .respond_with(ok(info))
                .mount(&server)
                .await;
            slack("conversations.history")
                .respond_with(ok(json!({"ok": true, "messages": []})))
                .expect(0)
                .mount(&server)
                .await;

            let args = Args {
                slack_channels: Some(format!("{CHANNEL},G0X")),
                ..args(&server)
            };
            let error = connector(&args).fetch_changed(None).await.unwrap_err();
            assert!(error.to_string().contains(expected), "{error:#}");
        }
    }

    #[tokio::test]
    async fn retries_after_rate_limit() {
        let server = MockServer::start().await;
        slack("conversations.info")
            .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "1"))
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;
        slack("conversations.info")
            .respond_with(ok(fixture("conversations.info")))
            .expect(1)
            .mount(&server)
            .await;
        slack("conversations.history")
            .respond_with(ok(json!({"ok": true, "messages": [], "has_more": false})))
            .expect(1)
            .mount(&server)
            .await;

        let documents = connector(&args(&server)).fetch_changed(None).await.unwrap();
        assert!(documents.is_empty());
    }

    #[tokio::test]
    async fn api_errors_name_the_missing_scope() {
        let server = MockServer::start().await;
        slack("conversations.info")
            .respond_with(ok(json!({
                "ok": false,
                "error": "missing_scope",
                "needed": "channels:read",
                "provided": "users:read"
            })))
            .mount(&server)
            .await;

        let error = connector(&args(&server))
            .fetch_changed(None)
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Slack conversations.info failed: missing_scope (missing scope channels:read)"
        );
    }

    #[test]
    fn renders_mentions_channels_and_links() {
        let names = HashMap::from([("U1".to_owned(), "alice".to_owned())]);
        assert_eq!(
            render("hi <@U1|old-name> and <@U9>", &names),
            "hi @alice and @U9"
        );
        assert_eq!(
            render(
                "<!here> <!subteam^S1|@basis-team> <#C1> <#C2|general> <#C3|>",
                &names
            ),
            "@here @basis-team #C1 #general #C3"
        );
        assert_eq!(
            render(
                "<mailto:a@example.com|a@example.com> <https://x.example> <http://sap.com|sap.com> <https://me.sap.com/notes/1?a=1&amp;b=2|Note 1>",
                &names
            ),
            "a@example.com https://x.example http://sap.com Note 1 (https://me.sap.com/notes/1?a=1&b=2)"
        );
        assert_eq!(
            render("x &amp;lt; y &lt;tag&gt; <unterminated", &names),
            "x &lt; y <tag> <unterminated"
        );
        assert_eq!(
            mentioned_users("<@U1> <@U2|bob> <#C1> <!here>").collect::<Vec<_>>(),
            ["U1", "U2"]
        );
    }

    #[test]
    fn titles_are_truncated_on_char_boundaries() {
        assert_eq!(truncate_title("short\n  question"), "short question");
        let long = "é".repeat(100);
        let title = truncate_title(&long);
        assert_eq!(title.chars().count(), TITLE_CHARS);
        assert!(title.ends_with('…'));
    }

    #[test]
    fn slack_timestamps_round_trip() {
        let time = ts_time("1790899200.000100").unwrap();
        assert_eq!(
            time,
            DateTime::from_timestamp(1_790_899_200, 100_000).unwrap()
        );
        assert_eq!(slack_ts(time), "1790899200.000100");
        assert_eq!(ts_time("1790899200").unwrap().timestamp(), 1_790_899_200);
        assert!(ts_time("not-a-ts").is_err());
    }
}
