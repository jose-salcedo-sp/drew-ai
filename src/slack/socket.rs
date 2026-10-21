//! Socket Mode: ack envelopes, dispatch events, reconnect with backoff.

use std::{sync::Arc, time::Duration};

use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio_tungstenite::{connect_async, tungstenite::Message};

use super::api::SlackApi;

/// Covers DNS, TCP, and the TLS and WebSocket handshakes.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

pub async fn run_socket(
    api: Arc<dyn SlackApi>,
    mut on_event: impl FnMut(Value) + Send,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) -> eros::Result<()> {
    let url = api.open_connection().await?;
    #[expect(
        clippy::disallowed_methods,
        reason = "bounded by CONNECT_TIMEOUT below"
    )]
    let connecting = connect_async(&url);
    let Ok(connected) = tokio::time::timeout(CONNECT_TIMEOUT, connecting).await else {
        eros::bail!("slack websocket connect timed out after {CONNECT_TIMEOUT:?}");
    };
    let (ws, _) = connected?;
    let (mut write, mut read) = ws.split();
    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    return Ok(());
                }
            }
            message = read.next() => {
                match message {
                    Some(Ok(Message::Text(text))) => {
                        let envelope: Value = serde_json::from_str(&text)?;
                        match envelope.get("type").and_then(Value::as_str) {
                            Some("hello") => {}
                            Some("disconnect") => {
                                eros::bail!("slack sent disconnect");
                            }
                            Some("events_api") => {
                                if let Some(id) = envelope.get("envelope_id").and_then(Value::as_str) {
                                    write
                                        .send(Message::Text(json!({ "envelope_id": id }).to_string().into()))
                                        .await?;
                                }
                                if let Some(event) = envelope
                                    .pointer("/payload/event")
                                    .cloned()
                                {
                                    on_event(event);
                                }
                            }
                            Some(other) => tracing::debug!(ty = other, "ignored slack envelope"),
                            None => {}
                        }
                    }
                    Some(Ok(Message::Close(_))) => eros::bail!("slack websocket closed"),
                    Some(Err(error)) => return Err(error.into()),
                    None => eros::bail!("slack websocket ended"),
                    _ => {}
                }
            }
        }
    }
}
