use std::sync::Arc;

use crate::{
    domain::{CategoryKind, Question, Score},
    pipeline::report::{Answer, CannedAnswerer, Outcome, Verdict},
    server::{AppState, auth::AdminUsers},
    slack::{fakes::FakeSlack, strip_mentions},
    store::fakes::MemoryRunStore,
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

#[test]
fn strip_mention_removes_bot_tag() {
    assert_eq!(
        strip_mentions("<@U123ABC> What is AGR_USERS?"),
        "What is AGR_USERS?"
    );
}

#[tokio::test]
async fn app_mention_posts_placeholder_then_updates_with_answer() {
    let slack = Arc::new(FakeSlack {
        ws_url: "ws://127.0.0.1:9".into(),
        ..Default::default()
    });
    let runs = Arc::new(MemoryRunStore::default());
    let state = AppState {
        answerer: Arc::new(CannedAnswerer {
            category: Some(CategoryKind::Boolean),
            outcome: answered(),
        }),
        runs: runs.clone(),
        keys: Arc::new("k:u".parse().unwrap()),
        admins: Arc::new(AdminUsers::default()),
    };
    let api = slack.clone();
    let event = serde_json::json!({
        "type": "app_mention",
        "user": "UJOSE",
        "text": "<@UBOT> What is AGR_USERS?",
        "channel": "C1",
        "ts": "100.001"
    });
    super::handle_event(&state, api.as_ref(), event)
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
async fn bot_messages_are_ignored() {
    let slack = Arc::new(FakeSlack::default());
    let state = AppState {
        answerer: Arc::new(CannedAnswerer {
            category: None,
            outcome: Outcome::Failed {
                at: "explain".into(),
                error: "timeout".into(),
            },
        }),
        runs: Arc::new(MemoryRunStore::default()),
        keys: Arc::new("k:u".parse().unwrap()),
        admins: Arc::new(AdminUsers::default()),
    };
    super::handle_event(
        &state,
        slack.as_ref(),
        serde_json::json!({ "type": "message", "bot_id": "B1", "text": "hi" }),
    )
    .await
    .unwrap();
    assert!(slack.posts.lock().unwrap().is_empty());
}
