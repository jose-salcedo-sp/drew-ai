//! Vectors are model output, so the contract is their shape: one `EMBEDDING_DIM` vector per input, in input order,
//! whatever the batch size.

use drew::llm::{EMBEDDING_DIM, EmbedPurpose, Embedder};
use wiremock::MockServer;

pub struct Adapter {
    pub embedder: Box<dyn Embedder>,
    pub _server: Option<MockServer>,
}

/// More texts than one OpenAI (64) or Cohere (96) batch, and than the Titan concurrency (8).
fn texts() -> Vec<String> {
    (0..100).map(|i| format!("text {i}")).collect()
}

pub async fn every_input_gets_one_vector_in_input_order(
    make: impl AsyncFnOnce() -> Adapter,
) -> eros::Result<()> {
    let adapter = make().await;
    let texts = texts();

    let batch = adapter
        .embedder
        .embed(&texts, EmbedPurpose::Document)
        .await?;

    assert_eq!(batch.len(), texts.len());
    assert!(batch.iter().all(|vector| vector.len() == EMBEDDING_DIM));
    for (text, vector) in texts.iter().zip(&batch) {
        let single = adapter
            .embedder
            .embed(std::slice::from_ref(text), EmbedPurpose::Document)
            .await?;
        assert_eq!(&single[0], vector, "{text}");
    }
    Ok(())
}

pub async fn no_inputs_give_no_vectors(make: impl AsyncFnOnce() -> Adapter) -> eros::Result<()> {
    let adapter = make().await;

    for purpose in [EmbedPurpose::Document, EmbedPurpose::Query] {
        assert!(adapter.embedder.embed(&[], purpose).await?.is_empty());
    }
    Ok(())
}

pub async fn a_query_gets_one_vector(make: impl AsyncFnOnce() -> Adapter) -> eros::Result<()> {
    let adapter = make().await;

    let vectors = adapter
        .embedder
        .embed(
            &["Which table maps users to roles?".into()],
            EmbedPurpose::Query,
        )
        .await?;

    assert_eq!(vectors.len(), 1);
    assert_eq!(vectors[0].len(), EMBEDDING_DIM);
    Ok(())
}

macro_rules! embedder_contract {
    ($make:expr) => {
        #[tokio::test]
        async fn every_input_gets_one_vector_in_input_order() -> eros::Result<()> {
            crate::embedder::every_input_gets_one_vector_in_input_order($make).await
        }

        #[tokio::test]
        async fn no_inputs_give_no_vectors() -> eros::Result<()> {
            crate::embedder::no_inputs_give_no_vectors($make).await
        }

        #[tokio::test]
        async fn a_query_gets_one_vector() -> eros::Result<()> {
            crate::embedder::a_query_gets_one_vector($make).await
        }
    };
}

/// A vector that depends only on `text`, as a real model's does.
fn vector_of(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0; EMBEDDING_DIM];
    for (i, byte) in text.bytes().enumerate() {
        vector[i % EMBEDDING_DIM] += f32::from(byte);
    }
    vector
}

mod hash {
    use drew::llm::fakes::HashEmbedder;

    use super::Adapter;

    async fn hash() -> Adapter {
        Adapter {
            embedder: Box::new(HashEmbedder),
            _server: None,
        }
    }

    embedder_contract!(hash);
}

mod openai {
    use drew::llm::{EMBEDDING_DIM, EmbedPurpose, Embedder, openai::OpenAiEmbedder};
    use serde_json::{Value, json};
    use wiremock::{
        Mock, MockServer, Request, ResponseTemplate,
        matchers::{method, path},
    };

    use super::{Adapter, vector_of};

    /// Answers each batch with its vectors in reverse order, each tagged with its input index.
    fn embeddings(request: &Request) -> ResponseTemplate {
        let body: Value = request.body_json().unwrap();
        let mut data: Vec<_> = body["input"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, text)| {
                json!({ "index": index, "embedding": vector_of(text.as_str().unwrap()) })
            })
            .collect();
        data.reverse();
        ResponseTemplate::new(200).set_body_json(json!({ "object": "list", "data": data }))
    }

    async fn server(response: impl wiremock::Respond + 'static) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(response)
            .mount(&server)
            .await;
        server
    }

    fn embedder(server: &MockServer) -> OpenAiEmbedder {
        OpenAiEmbedder::new(&format!("{}/v1", server.uri()).parse().unwrap(), "bge-m3").unwrap()
    }

    async fn openai() -> Adapter {
        let server = server(embeddings).await;
        Adapter {
            embedder: Box::new(embedder(&server)),
            _server: Some(server),
        }
    }

    embedder_contract!(openai);

    mod workers_ai_shaped {
        use super::*;

        /// `@cf/baai/bge-m3` on Workers AI's OpenAI-compatible route, with its API token.
        async fn workers_ai() -> Adapter {
            let server = server(embeddings).await;
            Adapter {
                embedder: Box::new(embedder(&server).with_key(Some("cf-token".parse().unwrap()))),
                _server: Some(server),
            }
        }

        embedder_contract!(workers_ai);
    }

    #[tokio::test]
    async fn the_key_is_sent_as_a_bearer_token_only_when_set() {
        for key in [Some("cf-token"), None] {
            let server = server(embeddings).await;

            embedder(&server)
                .with_key(key.map(|key| key.parse().unwrap()))
                .embed(&["a".into()], EmbedPurpose::Query)
                .await
                .unwrap();

            let sent = server.received_requests().await.unwrap();
            let authorization = sent[0]
                .headers
                .get("authorization")
                .map(|value| value.to_str().unwrap().to_owned());
            assert_eq!(authorization, key.map(|key| format!("Bearer {key}")));
        }
    }

    #[tokio::test]
    async fn texts_go_in_batches_of_64() {
        let server = server(embeddings).await;
        let texts: Vec<String> = (0..65).map(|i| format!("text {i}")).collect();

        embedder(&server)
            .embed(&texts, EmbedPurpose::Document)
            .await
            .unwrap();

        let sizes: Vec<_> = server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .map(|request| {
                request.body_json::<Value>().unwrap()["input"]
                    .as_array()
                    .unwrap()
                    .len()
            })
            .collect();
        assert_eq!(sizes, [64, 1]);
    }

    #[tokio::test]
    async fn fewer_vectors_than_inputs_is_an_error() {
        let server = server(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{ "index": 0, "embedding": vec![0.0; EMBEDDING_DIM] }]
        })))
        .await;

        let error = embedder(&server)
            .embed(&["a".into(), "b".into()], EmbedPurpose::Document)
            .await
            .unwrap_err();

        assert!(
            format!("{error:#?}").contains("expected 2 embeddings, got 1"),
            "{error:#?}"
        );
    }

    #[tokio::test]
    async fn an_error_status_carries_the_status_and_body() {
        let server =
            server(ResponseTemplate::new(404).set_body_string("model \"bge-m3\" not found")).await;

        let error = embedder(&server)
            .embed(&["a".into()], EmbedPurpose::Document)
            .await
            .unwrap_err();

        let report = format!("{error:#?}");
        assert!(
            report.contains("404") && report.contains("not found"),
            "{report}"
        );
    }
}

mod bedrock {
    use std::time::Duration;

    use drew::llm::{EMBEDDING_DIM, EmbedPurpose, Embedder, bedrock_embed::BedrockEmbedder};
    use serde_json::{Value, json};
    use wiremock::{
        Mock, MockServer, Request, ResponseTemplate,
        matchers::{method, path_regex},
    };

    use super::{Adapter, vector_of};

    const TITAN: &str = "amazon.titan-embed-text-v2:0";
    const COHERE: &str = "cohere.embed-english-v3";

    /// Titan embeds one text per call. Later texts answer sooner, so out-of-order completion shows.
    fn titan(request: &Request) -> ResponseTemplate {
        let body: Value = request.body_json().unwrap();
        let text = body["inputText"].as_str().unwrap();
        let n: u64 = text
            .rsplit(' ')
            .next()
            .and_then(|n| n.parse().ok())
            .unwrap_or(0);
        ResponseTemplate::new(200)
            .set_delay(Duration::from_millis(5 * (7 - n % 8)))
            .set_body_json(json!({ "embedding": vector_of(text), "inputTextTokenCount": 2 }))
    }

    fn cohere(request: &Request) -> ResponseTemplate {
        let body: Value = request.body_json().unwrap();
        let texts = body["texts"].as_array().unwrap();
        let embeddings: Vec<_> = texts
            .iter()
            .map(|text| vector_of(text.as_str().unwrap()))
            .collect();
        ResponseTemplate::new(200).set_body_json(json!({
            "id": "test", "embeddings": embeddings, "texts": texts, "response_type": "embeddings_floats"
        }))
    }

    async fn server(response: impl wiremock::Respond + 'static) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex("^/model/[^/]+/invoke$"))
            .respond_with(response)
            .mount(&server)
            .await;
        server
    }

    fn embedder(server: &MockServer, model: &str) -> BedrockEmbedder {
        BedrockEmbedder::new(crate::bedrock::client(server), model).unwrap()
    }

    mod titan {
        use super::*;

        async fn titan_v2() -> Adapter {
            let server = server(titan).await;
            Adapter {
                embedder: Box::new(embedder(&server, TITAN)),
                _server: Some(server),
            }
        }

        embedder_contract!(titan_v2);
    }

    mod cohere {
        use super::*;

        async fn cohere_v3() -> Adapter {
            let server = server(cohere).await;
            Adapter {
                embedder: Box::new(embedder(&server, COHERE)),
                _server: Some(server),
            }
        }

        embedder_contract!(cohere_v3);
    }

    #[tokio::test]
    async fn a_rejected_call_names_the_model_and_the_reason() {
        let server = server(
            ResponseTemplate::new(400)
                .insert_header("x-amzn-errortype", "ValidationException")
                .set_body_json(json!({ "message": "The provided model identifier is invalid." })),
        )
        .await;

        let error = embedder(&server, TITAN)
            .embed(&["a".into()], EmbedPurpose::Document)
            .await
            .unwrap_err();

        let report = format!("{error:#?}");
        assert!(
            report.contains(TITAN) && report.contains("model identifier is invalid"),
            "{report}"
        );
    }

    #[tokio::test]
    async fn a_wrong_dimension_is_an_error() {
        let server = server(
            ResponseTemplate::new(200).set_body_json(json!({ "embedding": vec![0.0; 768] })),
        )
        .await;

        let error = embedder(&server, TITAN)
            .embed(&["a".into()], EmbedPurpose::Document)
            .await
            .unwrap_err();

        assert!(
            format!("{error:#?}").contains(&format!("expected {EMBEDDING_DIM}-dimensional")),
            "{error:#?}"
        );
    }

    #[tokio::test]
    async fn cohere_sends_the_purpose_as_its_input_type() {
        let server = server(cohere).await;
        let embedder = embedder(&server, COHERE);

        embedder
            .embed(&["a".into()], EmbedPurpose::Document)
            .await
            .unwrap();
        embedder
            .embed(&["a".into()], EmbedPurpose::Query)
            .await
            .unwrap();

        let types: Vec<_> = server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .map(|request| request.body_json::<Value>().unwrap()["input_type"].clone())
            .collect();
        assert_eq!(types, ["search_document", "search_query"]);
    }
}
