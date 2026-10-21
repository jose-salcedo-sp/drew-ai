//! Socket Mode: ack envelopes, dispatch events, end the session on Slack's request or a silent link.

use std::{pin::Pin, time::Duration};

use async_trait::async_trait;
use futures::{Sink, SinkExt, Stream, StreamExt};
use serde_json::{Value, json};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{self, Message},
};

/// Covers DNS, TCP, and the TLS and WebSocket handshakes.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Slack pings a healthy connection well within this; silence this long means a half-open link.
pub const IDLE_TIMEOUT: Duration = Duration::from_secs(60);

pub trait Transport:
    Stream<Item = Result<Message, tungstenite::Error>>
    + Sink<Message, Error = tungstenite::Error>
    + Send
{
}

impl<T> Transport for T where
    T: Stream<Item = Result<Message, tungstenite::Error>>
        + Sink<Message, Error = tungstenite::Error>
        + Send
{
}

pub type Connection = Pin<Box<dyn Transport>>;

/// Opens the WebSocket at the URL `apps.connections.open` returned.
#[async_trait]
pub trait Connector: Send + Sync {
    async fn connect(&self, url: &str) -> eros::Result<Connection>;
}

pub struct WebSocketConnector;

#[async_trait]
impl Connector for WebSocketConnector {
    async fn connect(&self, url: &str) -> eros::Result<Connection> {
        connect(url).await
    }
}

/// The URL carries a one-time ticket, so it stays out of the error context.
#[eros::context("connect the Slack Socket Mode websocket")]
async fn connect(url: &str) -> eros::Result<Connection> {
    #[expect(
        clippy::disallowed_methods,
        reason = "bounded by CONNECT_TIMEOUT below"
    )]
    let connecting = connect_async(url);
    let Ok(connected) = tokio::time::timeout(CONNECT_TIMEOUT, connecting).await else {
        eros::bail!("timed out after {CONNECT_TIMEOUT:?}");
    };
    let (ws, _) = connected?;
    Ok(Box::pin(ws))
}

/// Reconnect delays: doubling from `INITIAL` up to `MAX`, and back to `INITIAL` once a session is healthy.
#[derive(Debug)]
pub struct Backoff {
    next: Duration,
}

impl Backoff {
    pub const INITIAL: Duration = Duration::from_secs(1);
    pub const MAX: Duration = Duration::from_secs(60);

    pub fn next_delay(&mut self) -> Duration {
        let delay = self.next;
        self.next = (delay * 2).min(Self::MAX);
        delay
    }

    pub fn reset(&mut self) {
        self.next = Self::INITIAL;
    }
}

impl Default for Backoff {
    fn default() -> Self {
        Self {
            next: Self::INITIAL,
        }
    }
}

/// Serves one connection until it fails or Slack ends it. `hello` resets `backoff`.
///
/// `Ok` means Slack ended a healthy session (a `disconnect` envelope, or a close after `hello`),
/// which calls for an immediate reconnect.
#[eros::context("serve a Slack Socket Mode session")]
pub async fn run_session(
    connection: Connection,
    backoff: &mut Backoff,
    on_event: impl FnMut(Value) + Send,
) -> eros::Result<()> {
    let (mut connection, mut on_event) = (connection, on_event);
    let mut greeted = false;
    loop {
        let Ok(message) = tokio::time::timeout(IDLE_TIMEOUT, connection.next()).await else {
            eros::bail!("no message from slack within {IDLE_TIMEOUT:?}");
        };
        match message {
            Some(Ok(Message::Text(text))) => {
                let envelope: Value = match serde_json::from_str(&text) {
                    Ok(envelope) => envelope,
                    Err(error) => {
                        tracing::warn!(
                            error = format!("{error:#?}"),
                            "skipped a malformed slack envelope"
                        );
                        continue;
                    }
                };
                match envelope.get("type").and_then(Value::as_str) {
                    Some("hello") => {
                        greeted = true;
                        backoff.reset();
                    }
                    Some("disconnect") => {
                        let reason = envelope.get("reason").and_then(Value::as_str);
                        tracing::info!(reason, "slack asked for a reconnect");
                        return Ok(());
                    }
                    Some("events_api") => {
                        if let Some(id) = envelope.get("envelope_id").and_then(Value::as_str) {
                            let ack = json!({ "envelope_id": id }).to_string();
                            connection.send(Message::Text(ack.into())).await?;
                        }
                        if let Some(event) = envelope.pointer("/payload/event").cloned() {
                            on_event(event);
                        }
                    }
                    Some(other) => tracing::debug!(ty = other, "ignored slack envelope"),
                    None => {}
                }
            }
            Some(Ok(Message::Close(_))) | None if greeted => return Ok(()),
            Some(Ok(Message::Close(_))) | None => {
                eros::bail!("slack closed the websocket before hello")
            }
            Some(Err(error)) => return Err(error.into()),
            Some(Ok(_)) => {}
        }
    }
}
