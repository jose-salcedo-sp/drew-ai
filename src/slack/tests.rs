use std::{
    io::Write,
    sync::{Arc, Mutex},
};

use crate::{
    domain::{CategoryKind, Question, Score},
    pipeline::report::{Answer, Answerer, CannedAnswerer, Outcome, Verdict},
    server::{AppState, auth::AdminUsers},
    slack::{Bot, SlackArgs, fakes::FakeSlack, strip_mentions},
    store::{
        RunStore,
        fakes::{FailingRunStore, MemoryRunStore},
    },
};

fn answered() -> Outcome {
    Outcome::Answered(Answer {
        verdict: Some(Verdict::Boolean {
            yes: true,
            confidence: Score(0.9),
        }),
        why: vec![],
        citations: vec![],
    })
}

fn failed() -> Outcome {
    Outcome::Failed {
        at: "answer".into(),
        error: "POST https://system-one.internal/v1/decide: 502 body=<html>upstream</html>".into(),
    }
}

fn state(answerer: Arc<dyn Answerer>, runs: Arc<dyn RunStore>) -> AppState {
    AppState {
        answerer,
        runs,
        keys: Arc::new("k:u".parse().unwrap()),
        admins: Arc::new(AdminUsers::default()),
    }
}

fn canned(outcome: Outcome) -> Arc<dyn Answerer> {
    Arc::new(CannedAnswerer {
        category: Some(CategoryKind::Boolean),
        outcome,
    })
}

fn mention() -> serde_json::Value {
    serde_json::json!({
        "type": "app_mention",
        "user": "UJOSE",
        "text": "<@UBOT> What is AGR_USERS?",
        "channel": "C1",
        "ts": "100.001"
    })
}

/// Collects the log lines written on this thread while the guard lives.
#[derive(Clone, Default)]
pub(super) struct Logs(Arc<Mutex<Vec<u8>>>);

impl Logs {
    pub(super) fn capture() -> (Self, tracing::subscriber::DefaultGuard) {
        let logs = Self::default();
        let writer = logs.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        (logs, tracing::subscriber::set_default(subscriber))
    }

    pub(super) fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
}

impl Write for Logs {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn empty_tokens_leave_the_bot_off() {
    let args = |app: Option<&str>, bot: Option<&str>| SlackArgs {
        app_token: app.map(Into::into),
        bot_token: bot.map(Into::into),
    };
    assert!(Bot::from_args(&args(None, None)).unwrap().is_none());
    assert!(
        Bot::from_args(&args(Some(""), Some(" ")))
            .unwrap()
            .is_none()
    );
    assert!(Bot::from_args(&args(Some("xapp-1"), Some(""))).is_err());
    assert!(
        Bot::from_args(&args(Some("xapp-1"), Some("xoxb-1")))
            .unwrap()
            .is_some()
    );
}

#[test]
fn strip_mention_removes_bot_tag() {
    assert_eq!(
        strip_mentions("<@U123ABC> What is AGR_USERS?"),
        "What is AGR_USERS?"
    );
}

#[tokio::test]
async fn app_mention_posts_placeholder_then_updates_with_answer() {
    let slack = Arc::new(FakeSlack::default());
    let runs = Arc::new(MemoryRunStore::default());
    let state = state(canned(answered()), Arc::<MemoryRunStore>::clone(&runs));
    super::handle_event(&state, slack.as_ref(), mention())
        .await
        .unwrap();

    let posts = slack.posts.lock().unwrap();
    assert_eq!(posts.len(), 1);
    assert_eq!(posts[0].0, "C1");
    assert_eq!(posts[0].1.as_deref(), Some("100.001"));
    assert_eq!(posts[0].2, "Looking that up...");

    let updates = slack.updates.lock().unwrap();
    assert_eq!(updates.len(), 1);
    assert!(updates[0].2.contains("Run "));
    assert_eq!(updates[0].0, "C1");

    let stored = runs.stored();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].user.0, "slack:UJOSE");
    assert_eq!(
        stored[0].report.question,
        Question::new("What is AGR_USERS?")
    );
}

#[tokio::test]
async fn a_failed_run_posts_only_its_id_and_step_and_logs_the_error() {
    let (logs, _guard) = Logs::capture();
    let slack = Arc::new(FakeSlack::default());
    let runs = Arc::new(MemoryRunStore::default());
    let state = state(canned(failed()), Arc::<MemoryRunStore>::clone(&runs));
    super::handle_event(&state, slack.as_ref(), mention())
        .await
        .unwrap();

    let run_id = runs.stored()[0].report.id;
    let updates = slack.updates.lock().unwrap();
    assert_eq!(updates[0].2, format!("Run {run_id} failed at answer."));
    let logs = logs.text();
    assert!(logs.contains("system-one.internal"), "{logs}");
    assert_eq!(logs.matches("system-one.internal").count(), 1, "{logs}");
}

#[tokio::test]
async fn an_unsaved_run_posts_only_its_id_and_logs_the_error() {
    let (logs, _guard) = Logs::capture();
    let slack = Arc::new(FakeSlack::default());
    let state = state(canned(failed()), Arc::new(FailingRunStore));
    super::handle_event(&state, slack.as_ref(), mention())
        .await
        .unwrap();

    let updates = slack.updates.lock().unwrap();
    let text = &updates[0].2;
    assert!(text.starts_with("Run "), "{text}");
    assert!(text.ends_with(" could not be completed."), "{text}");
    assert!(!text.contains("postgres"), "{text}");
    assert!(!text.contains("system-one"), "{text}");
    let logs = logs.text();
    assert_eq!(logs.matches("db.internal").count(), 1, "{logs}");
}

mod socket {
    use std::time::Duration;

    use futures::{Sink, SinkExt, Stream, StreamExt};
    use serde_json::json;
    use tokio::{sync::watch, time::Instant};
    use tokio_tungstenite::tungstenite::{self, Message};

    use super::*;
    use crate::slack::{
        fakes::{FakeSocket, connection_pair},
        socket::{Backoff, Connector, IDLE_TIMEOUT, WebSocketConnector, run_session},
    };

    fn envelope(json: &serde_json::Value) -> Message {
        Message::text(json.to_string())
    }

    fn hello() -> Message {
        envelope(&json!({ "type": "hello" }))
    }

    fn disconnect() -> Message {
        envelope(&json!({ "type": "disconnect", "reason": "refresh_requested" }))
    }

    /// A backoff whose next delay is no longer `INITIAL`.
    fn grown() -> Backoff {
        let mut backoff = Backoff::default();
        backoff.next_delay();
        backoff.next_delay();
        backoff
    }

    async fn carries_frames_both_ways<S>(
        connector: &dyn Connector,
        url: &str,
        slack: impl Future<Output = S>,
    ) where
        S: Stream<Item = Result<Message, tungstenite::Error>>
            + Sink<Message, Error = tungstenite::Error>
            + Unpin,
    {
        let (bot, mut slack) = tokio::join!(connector.connect(url), slack);
        let mut bot = bot.unwrap();
        slack.send(hello()).await.unwrap();
        assert_eq!(bot.next().await.unwrap().unwrap(), hello());
        bot.send(Message::text("ack")).await.unwrap();
        assert_eq!(slack.next().await.unwrap().unwrap(), Message::text("ack"));
    }

    #[tokio::test]
    async fn fake_connector_carries_frames_both_ways() {
        let socket = FakeSocket::default();
        let slack = socket.accept().await;
        carries_frames_both_ways(&socket, "", async { slack }).await;
    }

    #[tokio::test]
    async fn websocket_connector_carries_frames_both_ways() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        let slack = async move {
            let (tcp, _) = listener.accept().await.unwrap();
            tokio_tungstenite::accept_async(tcp).await.unwrap()
        };
        carries_frames_both_ways(&WebSocketConnector, &url, slack).await;
    }

    #[tokio::test]
    async fn fake_connector_refusal_is_an_error() {
        let socket = FakeSocket::default();
        socket.refuse();
        assert!(socket.connect("").await.is_err());
    }

    #[tokio::test]
    async fn websocket_connector_refusal_is_an_error() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        drop(listener);
        assert!(WebSocketConnector.connect(&url).await.is_err());
    }

    #[tokio::test(start_paused = true)]
    async fn disconnect_ends_a_healthy_session_and_resets_the_backoff() {
        let (bot, mut slack) = connection_pair().await;
        slack.send(hello()).await.unwrap();
        slack.send(disconnect()).await.unwrap();
        let mut backoff = grown();
        run_session(bot, &mut backoff, |_| {}).await.unwrap();
        assert_eq!(backoff.next_delay(), Backoff::INITIAL);
    }

    #[tokio::test(start_paused = true)]
    async fn a_close_after_hello_is_routine() {
        let (bot, mut slack) = connection_pair().await;
        slack.send(hello()).await.unwrap();
        slack.send(Message::Close(None)).await.unwrap();
        run_session(bot, &mut Backoff::default(), |_| {})
            .await
            .unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn a_close_before_hello_is_a_failure() {
        let (bot, mut slack) = connection_pair().await;
        slack.send(Message::Close(None)).await.unwrap();
        let mut backoff = grown();
        assert!(run_session(bot, &mut backoff, |_| {}).await.is_err());
        assert_ne!(backoff.next_delay(), Backoff::INITIAL);
    }

    #[tokio::test(start_paused = true)]
    async fn a_malformed_envelope_is_skipped_and_the_next_event_still_arrives() {
        let (logs, _guard) = Logs::capture();
        let (bot, mut slack) = connection_pair().await;
        slack.send(hello()).await.unwrap();
        slack.send(Message::text("{not json")).await.unwrap();
        let event = json!({ "type": "app_mention" });
        slack
            .send(envelope(&json!({
                "type": "events_api",
                "envelope_id": "E1",
                "payload": { "event": event },
            })))
            .await
            .unwrap();
        slack.send(Message::Close(None)).await.unwrap();

        let mut events = Vec::new();
        run_session(bot, &mut Backoff::default(), |event| events.push(event))
            .await
            .unwrap();

        assert_eq!(events, [event]);
        assert_eq!(
            slack.next().await.unwrap().unwrap(),
            envelope(&json!({ "envelope_id": "E1" }))
        );
        assert!(logs.text().contains("malformed slack envelope"));
    }

    #[tokio::test(start_paused = true)]
    async fn a_silent_link_ends_the_session_after_the_idle_timeout() {
        let (bot, mut slack) = connection_pair().await;
        slack.send(hello()).await.unwrap();
        let started = Instant::now();
        let error = run_session(bot, &mut Backoff::default(), |_| {})
            .await
            .unwrap_err();
        let waited = started.elapsed();
        assert!(waited >= IDLE_TIMEOUT && waited < IDLE_TIMEOUT + Duration::from_secs(1));
        assert!(format!("{error:#?}").contains("no message from slack"));
    }

    #[tokio::test(start_paused = true)]
    async fn any_frame_restarts_the_idle_timeout() {
        let (bot, mut slack) = connection_pair().await;
        slack.send(hello()).await.unwrap();
        let started = Instant::now();
        let ping_at = Duration::from_secs(50);
        let mut backoff = Backoff::default();
        let (ended, ()) = tokio::join!(run_session(bot, &mut backoff, |_| {}), async {
            tokio::time::sleep(ping_at).await;
            slack.send(Message::Ping(Vec::new().into())).await.unwrap();
        });
        assert!(ended.is_err());
        let waited = started.elapsed();
        assert!(waited >= ping_at + IDLE_TIMEOUT, "{waited:?}");
    }

    #[tokio::test(start_paused = true)]
    async fn reconnects_at_once_after_disconnect_and_resets_the_backoff_after_hello() {
        let socket = Arc::new(FakeSocket::default());
        socket.refuse();
        socket.refuse();
        let mut slack = socket.accept().await;
        slack.send(hello()).await.unwrap();
        slack.send(disconnect()).await.unwrap();
        socket.refuse();
        let bot = Bot::new(
            Arc::new(FakeSlack::default()),
            Arc::<FakeSocket>::clone(&socket),
        );
        let (stop, shutdown) = watch::channel(false);
        let started = Instant::now();

        tokio::join!(
            bot.run(
                state(canned(answered()), Arc::new(MemoryRunStore::default())),
                shutdown
            ),
            async {
                tokio::time::sleep(Duration::from_secs(30)).await;
                stop.send(true).unwrap();
            }
        );

        let offsets: Vec<u64> = socket
            .connects()
            .iter()
            .map(|at| at.duration_since(started).as_secs())
            .collect();
        // Refused twice (waits 1 s, 2 s), then a healthy session Slack ends: reconnect at once.
        // The next refusal waits `INITIAL` again instead of 4 s.
        assert_eq!(offsets, [0, 1, 3, 3, 4]);
    }
}

#[tokio::test]
async fn bot_messages_are_ignored() {
    let slack = Arc::new(FakeSlack::default());
    let state = state(canned(failed()), Arc::new(MemoryRunStore::default()));
    super::handle_event(
        &state,
        slack.as_ref(),
        serde_json::json!({ "type": "message", "bot_id": "B1", "text": "hi" }),
    )
    .await
    .unwrap();
    assert!(slack.posts.lock().unwrap().is_empty());
}
