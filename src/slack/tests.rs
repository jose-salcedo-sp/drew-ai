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
