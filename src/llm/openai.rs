//! `ChatModel` and `Embedder` over an OpenAI-compatible HTTP API: Ollama, vLLM, or llama.cpp.

use std::time::Duration;

use async_trait::async_trait;
use eros::{Context, bail, ensure};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{
    ChatModel, ChatRequest, ChatResponse, EMBEDDING_DIM, EmbedPurpose, Embedder, ModelRef,
};

/// A local 14B model can take a minute to write a long structured answer.
const TIMEOUT: Duration = Duration::from_secs(300);
const EMBED_BATCH: usize = 64;

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(TIMEOUT)
        .build()
        .expect("a client with only a timeout set always builds")
}

fn endpoint(base_url: &str, path: &str) -> String {
    format!("{}/{path}", base_url.trim_end_matches('/'))
}

#[eros::context("POST {}", url)]
async fn post(client: &reqwest::Client, url: &str, body: &Value) -> eros::Result<Value> {
    let response = client.post(url).json(body).send().await?;
    let status = response.status();
    let text = response.text().await?;
    ensure!(status.is_success(), "returned {status}: {text}");
    Ok(serde_json::from_str(&text)?)
}

pub struct OpenAiChat {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OpenAiChat {
    /// `base_url` includes the API prefix, e.g. `http://localhost:11434/v1` for Ollama.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: client(),
            base_url: base_url.into(),
            model: model.into(),
        }
    }
}

pub fn chat_body(model: &str, request: &ChatRequest) -> Value {
    let mut body = json!({
        "model": model,
        "messages": [
            { "role": "system", "content": request.system },
            { "role": "user", "content": request.user },
        ],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    });
    if let Some(schema) = &request.json_schema {
        body["response_format"] = json!({
            "type": "json_schema",
            "json_schema": { "name": "respond", "strict": true, "schema": schema },
        });
    }
    body
}

/// The first choice's content, without the `<think>` block reasoning models such as Qwen3 may prepend.
pub fn chat_text(response: &Value) -> eros::Result<String> {
    let content = response["choices"][0]["message"]["content"]
        .as_str()
        .context("response has no choices[0].message.content")?;
    let content = match content.trim_start().strip_prefix("<think>") {
        Some(rest) => rest.split_once("</think>").map_or("", |(_, answer)| answer),
        None => content,
    };
    Ok(content.trim().to_owned())
}

#[async_trait]
impl ChatModel for OpenAiChat {
    fn model(&self) -> ModelRef {
        ModelRef::new(&self.model, "openai-compatible")
    }

    async fn complete(&self, request: ChatRequest) -> eros::Result<ChatResponse> {
        let url = endpoint(&self.base_url, "chat/completions");
        let response = post(&self.client, &url, &chat_body(&self.model, &request)).await?;
        Ok(ChatResponse {
            text: chat_text(&response)?,
            mean_logprob: None,
        })
    }
}

pub struct OpenAiEmbedder {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl OpenAiEmbedder {
    /// The model must produce `EMBEDDING_DIM` dimensions, e.g. `bge-m3` or `mxbai-embed-large` in Ollama.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: client(),
            base_url: base_url.into(),
            model: model.into(),
        }
    }
}

pub fn embeddings(response: &Value, count: usize) -> eros::Result<Vec<Vec<f32>>> {
    #[derive(Deserialize)]
    struct Response {
        data: Vec<Item>,
    }
    #[derive(Deserialize)]
    struct Item {
        index: usize,
        embedding: Vec<f32>,
    }
    let mut data = Response::deserialize(response)
        .context("parsing the embeddings response")?
        .data;
    ensure!(
        data.len() == count,
        "expected {count} embeddings, got {}",
        data.len()
    );
    data.sort_by_key(|item| item.index);
    for item in &data {
        if item.embedding.len() != EMBEDDING_DIM {
            bail!(
                "the embedding model returned {} dimensions, but chunks store {EMBEDDING_DIM}; use a \
                 {EMBEDDING_DIM}-dimension model such as bge-m3 or mxbai-embed-large",
                item.embedding.len()
            );
        }
    }
    Ok(data.into_iter().map(|item| item.embedding).collect())
}

#[async_trait]
impl Embedder for OpenAiEmbedder {
    fn model(&self) -> ModelRef {
        ModelRef::new(&self.model, format!("dimensions={EMBEDDING_DIM}"))
    }

    async fn embed(&self, texts: &[String], _purpose: EmbedPurpose) -> eros::Result<Vec<Vec<f32>>> {
        let url = endpoint(&self.base_url, "embeddings");
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(EMBED_BATCH) {
            let body = json!({ "model": self.model, "input": batch });
            let response = post(&self.client, &url, &body).await?;
            vectors.extend(embeddings(&response, batch.len())?);
        }
        Ok(vectors)
    }
}

#[cfg(test)]
mod tests {
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_partial_json, method, path},
    };

    use super::*;

    fn request(schema: Option<Value>) -> ChatRequest {
        ChatRequest {
            system: "Answer from the sources.".into(),
            user: "<question>q</question>".into(),
            max_tokens: 512,
            temperature: 0.0,
            json_schema: schema,
        }
    }

    #[test]
    fn a_schema_becomes_a_json_schema_response_format() {
        let schema = json!({ "type": "object" });
        let body = chat_body("qwen3:14b", &request(Some(schema.clone())));
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(body["response_format"]["json_schema"]["schema"], schema);
        assert_eq!(body["messages"][0]["role"], "system");
        assert!(
            chat_body("qwen3:14b", &request(None))
                .get("response_format")
                .is_none()
        );
    }

    #[test]
    fn a_think_block_is_dropped() {
        let response =
            |content: &str| json!({ "choices": [{ "message": { "content": content } }] });
        assert_eq!(
            chat_text(&response("<think>\nhmm\n</think>\n{\"claims\": []}")).unwrap(),
            "{\"claims\": []}"
        );
        assert_eq!(chat_text(&response(" {\"a\": 1} ")).unwrap(), "{\"a\": 1}");
        assert_eq!(chat_text(&response("<think>never closed")).unwrap(), "");
    }

    #[tokio::test]
    async fn chat_posts_to_chat_completions() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_partial_json(json!({ "model": "qwen3:14b" })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{ "message": { "role": "assistant", "content": "{\"claims\": []}" } }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let chat = OpenAiChat::new(format!("{}/v1/", server.uri()), "qwen3:14b");
        let response = chat.complete(request(Some(json!({})))).await.unwrap();
        assert_eq!(response.text, "{\"claims\": []}");
    }

    #[tokio::test]
    async fn errors_carry_the_status_and_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(404).set_body_string("model \"qwen3:14b\" not found"),
            )
            .mount(&server)
            .await;

        let error = OpenAiChat::new(server.uri(), "qwen3:14b")
            .complete(request(None))
            .await
            .unwrap_err();
        assert!(format!("{error:#?}").contains("not found"), "{error:#?}");
    }

    #[tokio::test]
    async fn embeddings_come_back_in_input_order_and_batched() {
        let server = MockServer::start().await;
        let vector = |x: f32| vec![x; EMBEDDING_DIM];
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    { "index": 1, "embedding": vector(2.0) },
                    { "index": 0, "embedding": vector(1.0) },
                ]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let embedder = OpenAiEmbedder::new(format!("{}/v1", server.uri()), "bge-m3");
        let vectors = embedder
            .embed(&["a".into(), "b".into()], EmbedPurpose::Document)
            .await
            .unwrap();
        assert_eq!(vectors[0][0], 1.0);
        assert_eq!(vectors[1][0], 2.0);
    }

    #[test]
    fn a_wrong_dimension_names_the_fix() {
        let response = json!({ "data": [{ "index": 0, "embedding": vec![0.0; 768] }] });
        let error = embeddings(&response, 1).unwrap_err().to_string();
        assert!(error.contains("768") && error.contains("bge-m3"), "{error}");
    }
}
