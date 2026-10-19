use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use drew::{
    domain::{
        AtLeastOne, CategoryKind, ChunkId, ChunkRef, ChunkVersion, Cited, Provenance, Question,
        RunId, Score, Span, UserId,
    },
    pipeline::{
        DeclineReason,
        report::{Answer, CannedAnswerer, Citation, Outcome, Rejected, RunReport, Verdict},
        trace::Trace,
    },
    server::{AppState, router},
    store::{RunStore, runs::PgRunStore},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use sqlx::PgPool;
use tower::ServiceExt;

fn citation() -> Citation {
    Citation {
        chunk: ChunkRef {
            id: ChunkId("markdown:roles.md#roles".into()),
            version: ChunkVersion("v1".into()),
        },
        title: "Roles".into(),
        url: "https://wiki.example.test/1".into(),
        excerpt: "PFCG maintains roles.".into(),
    }
}

fn trace() -> Trace {
    serde_json::from_value(json!([{
        "step": "rerank",
        "group": null,
        "started_at": "2026-10-09T12:00:01.123456789Z",
        "duration_ms": 60,
        "model": { "id": "system-one", "version": "1" },
        "prompt_version": "rerank-v1",
        "outcome": { "status": "declined", "reason": { "kind": "low_confidence" }, "score": 0.25, "threshold": 0.5 },
        "record": { "kind": "rerank", "raw": "[1]", "chunks": [
            { "chunk": { "id": "markdown:roles.md#roles", "version": "v1" }, "score": 0.25 }
        ] }
    }]))
    .unwrap()
}

fn reports() -> Vec<(RunReport, &'static str)> {
    let report = |category, outcome| RunReport {
        id: RunId::new(),
        question: Question::new("PFCG or SU01 for roles?"),
        category,
        outcome,
        trace: trace(),
    };
    let prompt = |start, end| AtLeastOne::one(Provenance::UserPrompt(Span { start, end }));
    let answered = Outcome::Answered(Answer {
        verdict: Some(Verdict::Choice {
            selected: Cited {
                value: "PFCG".into(),
                sources: prompt(0, 4),
            },
            rejected: vec![Rejected {
                option: Cited {
                    value: "SU01".into(),
                    sources: prompt(8, 12),
                },
                reason: "edits users".into(),
            }],
        }),
        why: vec![Cited {
            value: "PFCG maintains roles.".into(),
            sources: AtLeastOne::one(Provenance::Chunk(citation().chunk.id)),
        }],
        citations: vec![citation()],
    });
    let declined = Outcome::Declined {
        at: "answer".into(),
        reason: DeclineReason::InvalidModelOutput("missing field `selected`".into()),
        score: None,
        threshold: Some(Score(0.7)),
        sources: vec![citation()],
    };
    let failed = Outcome::Failed {
        at: "classify".into(),
        error: "connection refused".into(),
    };
    vec![
        (report(Some(CategoryKind::Choice), answered), "answered"),
        (report(Some(CategoryKind::Open), declined), "declined"),
        (report(None, failed), "failed"),
    ]
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn round_trips_every_outcome(pool: PgPool) -> eros::Result<()> {
    let store = PgRunStore::new(pool.clone());
    let user = UserId("drew".into());
    for (report, status) in reports() {
        store.save(&user, &report).await?;

        let stored = store.get(report.id).await?.expect("run stored");
        assert_eq!(stored.user, user);
        assert_eq!(stored.report, report);

        let (column,): (String,) = sqlx::query_as("SELECT status FROM pipeline_runs WHERE id = $1")
            .bind(report.id.as_uuid())
            .fetch_one(&pool)
            .await?;
        assert_eq!(column, status);
    }
    assert!(store.get(RunId::new()).await?.is_none());
    Ok(())
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn a_run_id_is_saved_once(pool: PgPool) -> eros::Result<()> {
    let store = PgRunStore::new(pool);
    let (report, _) = reports().remove(0);
    store.save(&UserId("drew".into()), &report).await?;
    assert!(store.save(&UserId("jose".into()), &report).await.is_err());
    Ok(())
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn served_runs_are_persisted_in_postgres(pool: PgPool) -> eros::Result<()> {
    let (report, _) = reports().remove(1);
    let state = AppState {
        answerer: Arc::new(CannedAnswerer {
            category: report.category,
            outcome: report.outcome,
        }),
        runs: Arc::new(PgRunStore::new(pool)),
        keys: Arc::new("sk-drew:drew".parse().unwrap()),
        admins: Arc::new(drew::server::auth::AdminUsers::default()),
    };
    let app = router(state);
    let send = |request: Request<Body>| {
        let app = app.clone();
        async move {
            let response = app.oneshot(request).await.unwrap();
            let status = response.status();
            let body = response.into_body().collect().await.unwrap().to_bytes();
            (status, serde_json::from_slice::<Value>(&body).unwrap())
        }
    };

    let (status, completion) = send(
        Request::post("/v1/chat/completions")
            .header(header::AUTHORIZATION, "Bearer sk-drew")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({ "messages": [{ "role": "user", "content": "PFCG or SU01?" }] }).to_string(),
            ))?,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let id = completion["drew"]["run_id"].as_str().unwrap();

    let (status, run) = send(
        Request::get(format!("/v1/runs/{id}"))
            .header(header::AUTHORIZATION, "Bearer sk-drew")
            .body(Body::empty())?,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(run["status"], "declined");
    assert_eq!(run["question"], "PFCG or SU01?");
    assert_eq!(run["outcome"]["reason"]["kind"], "invalid_model_output");
    Ok(())
}
