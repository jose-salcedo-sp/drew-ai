//! System One answers are model output, so the contract is the transport: what reaches the model, and that every
//! answer comes back under its question id.

use std::{collections::BTreeMap, sync::Arc};

use drew::llm::system_one::{
    Decision, DecisionAnswer, DecisionRequest, DecisionResponse, Options, SystemOne,
};
use futures::future::BoxFuture;
use serde_json::Value;

/// An adapter set up to give one reply, and the wire body of every request it sent.
pub struct Adapter {
    pub system_one: Arc<dyn SystemOne>,
    pub sent: Box<dyn Fn() -> BoxFuture<'static, Vec<Value>> + Send + Sync>,
}

fn request() -> DecisionRequest {
    DecisionRequest {
        state: "Does AGR_USERS store role assignments?".into(),
        model: String::new(),
        questions: BTreeMap::from([
            (
                "category".into(),
                Decision::Choice {
                    instructions: "What kind of answer?".into(),
                    criteria: Options(vec![
                        ("open".into(), Some("Free-form".into())),
                        ("boolean".into(), None),
                    ]),
                },
            ),
            (
                "relevant".into(),
                Decision::Noul {
                    instructions: "Does the text name a table?".into(),
                },
            ),
        ]),
    }
}

fn reply() -> DecisionResponse {
    DecisionResponse {
        answers: BTreeMap::from([
            (
                "category".into(),
                DecisionAnswer::Choice {
                    choice: "boolean".into(),
                    confidence: 0.75,
                    probabilities: BTreeMap::from([
                        ("boolean".into(), 0.875),
                        ("open".into(), 0.125),
                    ]),
                },
            ),
            ("relevant".into(), DecisionAnswer::Noul { noul: 0.5 }),
        ]),
    }
}

pub async fn answers_come_back_by_question_id(
    make: impl AsyncFnOnce(DecisionResponse) -> Adapter,
) -> eros::Result<()> {
    let adapter = make(reply()).await;

    let response = adapter.system_one.decide(request()).await?;

    assert_eq!(response, reply());
    Ok(())
}

pub async fn the_request_is_sent_under_the_adapters_model_name(
    make: impl AsyncFnOnce(DecisionResponse) -> Adapter,
) -> eros::Result<()> {
    let adapter = make(reply()).await;
    let request = DecisionRequest {
        model: "set by the caller".into(),
        ..request()
    };

    adapter.system_one.decide(request.clone()).await?;

    let expected = DecisionRequest {
        model: adapter.system_one.model().id,
        ..request
    };
    assert_eq!((adapter.sent)().await, [serde_json::to_value(&expected)?]);
    Ok(())
}

macro_rules! system_one_contract {
    ($make:expr) => {
        #[tokio::test]
        async fn answers_come_back_by_question_id() -> eros::Result<()> {
            crate::system_one::answers_come_back_by_question_id($make).await
        }

        #[tokio::test]
        async fn the_request_is_sent_under_the_adapters_model_name() -> eros::Result<()> {
            crate::system_one::the_request_is_sent_under_the_adapters_model_name($make).await
        }
    };
}

mod scripted {
    use std::sync::Arc;

    use drew::llm::{fakes::ScriptedSystemOne, system_one::DecisionResponse};

    use super::Adapter;

    async fn scripted(reply: DecisionResponse) -> Adapter {
        let fake = Arc::new(ScriptedSystemOne::new([Ok(reply)]));
        Adapter {
            system_one: Arc::<ScriptedSystemOne>::clone(&fake),
            sent: Box::new(move || {
                let fake = Arc::clone(&fake);
                Box::pin(async move {
                    fake.requests()
                        .iter()
                        .map(|request| serde_json::to_value(request).unwrap())
                        .collect()
                })
            }),
        }
    }

    system_one_contract!(scripted);
}

mod http {
    use std::sync::Arc;

    use drew::llm::system_one::{DecisionResponse, HttpSystemOne, SystemOne, SystemOneArgs};
    use serde_json::json;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    use super::{Adapter, request};

    fn client(server: &MockServer, key: Option<&str>) -> HttpSystemOne {
        HttpSystemOne::from_args(&SystemOneArgs {
            system_one_url: Some(format!("{}/", server.uri())),
            system_one_model: "kev-latest".into(),
            system_one_key: key.map(str::to_owned),
        })
        .expect("a URL is set")
    }

    async fn responding(response: ResponseTemplate) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .respond_with(response)
            .mount(&server)
            .await;
        server
    }

    async fn kev(reply: DecisionResponse) -> Adapter {
        let mut body = serde_json::to_value(&reply).unwrap();
        body["model"] = json!("kev-latest");
        body["usage"] = json!({ "input_tokens": 40, "output_tokens": 20 });
        body["latency_ms"] = json!(120);
        let server = Arc::new(responding(ResponseTemplate::new(200).set_body_json(body)).await);
        Adapter {
            system_one: Arc::new(client(&server, None)),
            sent: Box::new(move || {
                let server = Arc::clone(&server);
                Box::pin(async move {
                    server
                        .received_requests()
                        .await
                        .unwrap()
                        .iter()
                        .map(|request| request.body_json().unwrap())
                        .collect()
                })
            }),
        }
    }

    system_one_contract!(kev);

    #[tokio::test]
    async fn an_error_status_carries_the_status_and_body() {
        let server =
            responding(ResponseTemplate::new(500).set_body_string("model not loaded")).await;

        let error = client(&server, None).decide(request()).await.unwrap_err();

        let report = format!("{error:#?}");
        assert!(
            report.contains("500") && report.contains("model not loaded"),
            "{report}"
        );
    }

    #[tokio::test]
    async fn a_body_that_is_not_json_is_an_error() {
        let server = responding(ResponseTemplate::new(200).set_body_string("<html>")).await;

        assert!(client(&server, None).decide(request()).await.is_err());
    }

    #[tokio::test]
    async fn the_key_is_sent_as_a_bearer_token_only_when_set() {
        for key in [Some("secret"), None] {
            let server =
                responding(ResponseTemplate::new(200).set_body_json(json!({ "answers": {} })))
                    .await;

            client(&server, key).decide(request()).await.unwrap();

            let sent = server.received_requests().await.unwrap();
            let authorization = sent[0]
                .headers
                .get("authorization")
                .map(|value| value.to_str().unwrap().to_owned());
            assert_eq!(authorization, key.map(|key| format!("Bearer {key}")));
        }
    }
}
