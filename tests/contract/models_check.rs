//! `drew models check` against wiremock backends shaped like Workers AI: one call per port, each reported with its
//! model and outcome, and the key never printed.

use drew::{
    config::ConfigArgs,
    llm::{
        EMBEDDING_DIM,
        backend::{
            ChatArgs, ChatBackendKind, EmbedArgs, EmbedBackendKind, SystemOneArgs,
            SystemOneBackendKind,
        },
        openai::ReasoningEffort,
    },
    models::{Port, check},
    pipeline::answerer::PipelineArgs,
};
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

const TOKEN: &str = "cf-secret-token";
const ACCOUNT: &str = "/client/v4/accounts/acc-123/ai";

async fn workers_ai(system_one: ResponseTemplate) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(format!("{ACCOUNT}/v1/embeddings")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{ "index": 0, "embedding": vec![0.5; EMBEDDING_DIM] }]
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(format!("{ACCOUNT}/v1/chat/completions")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{ "message": { "content": "{\"ok\": true}", "reasoning": "a check" } }]
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path(format!("{ACCOUNT}/run/@cf/cloudflare/clef-flash")))
        .respond_with(system_one)
        .mount(&server)
        .await;
    server
}

fn clef_flash_answer() -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "result": {
            "model": "clef-flash",
            "answers": { "check": { "type": "noul", "noul": 0.97 } },
            "usage": { "input_tokens": 30, "output_tokens": 0 }
        },
        "success": true, "errors": [], "messages": []
    }))
}

fn args(server: &MockServer, system_one: SystemOneBackendKind) -> PipelineArgs {
    let openai_base = Some(format!("{}{ACCOUNT}/v1", server.uri()));
    let workers_ai = system_one == SystemOneBackendKind::WorkersAi;
    PipelineArgs {
        config: ConfigArgs::default(),
        embed: EmbedArgs {
            embed_backend: EmbedBackendKind::OpenaiCompatible,
            embed_model: "@cf/baai/bge-m3".into(),
            embed_url: openai_base.clone(),
            embed_key: Some(TOKEN.parse().unwrap()),
        },
        chat: ChatArgs {
            explain_backend: ChatBackendKind::OpenaiCompatible,
            explain_model: "@cf/qwen/qwen3.8-27b".into(),
            explain_url: openai_base,
            explain_key: Some(TOKEN.parse().unwrap()),
            explain_reasoning_effort: Some(ReasoningEffort::Medium),
            explain_schema_format: None,
        },
        system_one: SystemOneArgs {
            system_one_backend: system_one,
            system_one_model: workers_ai.then(|| "@cf/cloudflare/clef-flash".into()),
            system_one_url: workers_ai.then(|| format!("{}/client/v4", server.uri())),
            system_one_account_id: workers_ai.then(|| "acc-123".into()),
            system_one_key: workers_ai.then(|| TOKEN.parse().unwrap()),
        },
    }
}

#[tokio::test]
async fn every_backend_answers_once_and_is_reported_ok() {
    let server = workers_ai(clef_flash_answer()).await;

    let check = check(&args(&server, SystemOneBackendKind::WorkersAi)).await;

    assert_eq!(check.failed(), 0, "{check}");
    let ports: Vec<_> = check.probes.iter().map(|probe| probe.port).collect();
    assert_eq!(ports, [Port::Embeddings, Port::Explain, Port::SystemOne]);
    let report = check.to_string();
    for model in [
        "@cf/baai/bge-m3 (dimensions=1024)",
        "@cf/qwen/qwen3.8-27b (openai-compatible;reasoning_effort=medium)",
        "@cf/cloudflare/clef-flash (workers-ai-run)",
    ] {
        assert!(report.contains(model), "{report}");
    }
    assert!(!report.contains(TOKEN), "{report}");
    assert!(!report.contains("Bedrock"), "{report}");
    assert_eq!(server.received_requests().await.unwrap().len(), 3);
}

#[tokio::test]
async fn a_failing_backend_is_reported_with_its_error_and_the_others_still_pass() {
    let server = workers_ai(ResponseTemplate::new(401).set_body_json(json!({
        "result": null, "success": false,
        "errors": [{ "code": 10000, "message": "Authentication error" }], "messages": []
    })))
    .await;

    let check = check(&args(&server, SystemOneBackendKind::WorkersAi)).await;

    assert_eq!(check.failed(), 1, "{check}");
    let failed = check
        .probes
        .iter()
        .find(|probe| probe.outcome.is_err())
        .unwrap();
    assert_eq!(failed.port, Port::SystemOne);
    let report = check.to_string();
    assert!(
        report.contains("system one  @cf/cloudflare/clef-flash (workers-ai-run)")
            && report.contains("10000: Authentication error")
            && report.contains("asking one yes/no question"),
        "{report}"
    );
    assert!(!report.contains(TOKEN), "{report}");
}

#[tokio::test]
async fn a_missing_setting_fails_its_port_by_name() {
    let server = workers_ai(clef_flash_answer()).await;
    let mut args = args(&server, SystemOneBackendKind::WorkersAi);
    args.system_one.system_one_account_id = None;

    let check = check(&args).await;

    assert_eq!(check.failed(), 1, "{check}");
    let report = check.to_string();
    assert!(
        report.contains("not built")
            && report.contains(
                "DREW_SYSTEM_ONE_ACCOUNT_ID is required with DREW_SYSTEM_ONE_BACKEND=workers-ai"
            ),
        "{report}"
    );
}

#[tokio::test]
async fn no_system_one_is_a_note_not_a_failure() {
    let server = workers_ai(clef_flash_answer()).await;

    let check = check(&args(&server, SystemOneBackendKind::None)).await;

    assert_eq!(check.failed(), 0, "{check}");
    assert_eq!(check.probes.len(), 2);
    assert!(
        check
            .notes
            .iter()
            .any(|note| note.contains("DREW_SYSTEM_ONE_BACKEND=none")),
        "{check}"
    );
}
