//! `ChatModel` and `Embedder` over an OpenAI-compatible HTTP API: Ollama, vLLM, llama.cpp, or Workers AI
//! (`https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1`, with a bearer key).

use std::time::Duration;

use async_trait::async_trait;
use eros::{Context, bail, ensure};
use reqwest::Url;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{
    ApiKey, ChatModel, ChatRequest, ChatResponse, EMBEDDING_DIM, EmbedPurpose, Embedder, ModelRef,
};

/// A local 14B model can take a minute to write a long structured answer, and a hosted reasoning model longer.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const EMBED_BATCH: usize = 64;

#[eros::context("building the OpenAI-compatible HTTP client")]
fn client() -> eros::Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT)
        .build()?)
}

fn endpoint(base_url: &Url, path: &str) -> String {
    format!("{}/{path}", base_url.as_str().trim_end_matches('/'))
}

#[eros::context("POST {} for model {}", url, model)]
async fn post(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    key: Option<&ApiKey>,
    body: &Value,
) -> eros::Result<Value> {
    let mut post = client.post(url).json(body);
    if let Some(key) = key {
        post = post.bearer_auth(key.expose());
    }
    let response = post.send().await?;
    let status = response.status();
    let text = response.text().await?;
    ensure!(status.is_success(), "{model} returned {status}: {text}");
    Ok(serde_json::from_str(&text)?)
}

/// How much a reasoning model thinks before answering, sent as `reasoning_effort`. The levels are those of
/// `@cf/qwen/qwen3.8-27b` on Workers AI, whose default is `xhigh`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum ReasoningEffort {
    /// Fastest. On Workers AI, `low` with a `json_schema` response format hangs until the 120 s gateway timeout
    /// (cloudflare/workers-sdk#16165), so `Explain`, which always sends a schema, should use `medium`.
    Low,
    /// The recommended level for `Explain`.
    Medium,
    /// The model's default; the slowest, and the most reasoning tokens out of `max_tokens`.
    Xhigh,
}

impl ReasoningEffort {
    /// The wire value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::Xhigh => "xhigh",
        }
    }
}

/// How a request schema is written into `response_format.json_schema`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum SchemaFormat {
    /// OpenAI's `{"name": "respond", "strict": true, "schema": ...}`.
    #[default]
    Openai,
    /// The schema itself, as Workers AI's JSON Mode documentation writes it, for a server that rejects OpenAI's form.
    Bare,
}

/// Request options that shape the output, so they are part of the [`ModelRef`] version.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChatOptions {
    /// `None`: no `reasoning_effort` is sent, and the server's default applies.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// How a request schema is sent.
    pub schema_format: SchemaFormat,
}

impl ChatOptions {
    /// `openai-compatible`, then each option that is not the default, e.g. `;reasoning_effort=medium`.
    fn version(self) -> String {
        let mut version = String::from("openai-compatible");
        if let Some(effort) = self.reasoning_effort {
            version.push_str(";reasoning_effort=");
            version.push_str(effort.as_str());
        }
        match self.schema_format {
            SchemaFormat::Openai => {}
            SchemaFormat::Bare => version.push_str(";schema=bare"),
        }
        version
    }
}

/// [`ChatModel`] over `POST {base_url}/chat/completions`. A request schema becomes a `json_schema` response format.
pub struct OpenAiChat {
    client: reqwest::Client,
    url: String,
    model: String,
    key: Option<ApiKey>,
    options: ChatOptions,
}

impl OpenAiChat {
    /// `base_url` includes the API prefix, e.g. `http://localhost:11434/v1` for Ollama. No key, default options.
    ///
    /// # Errors
    ///
    /// When the HTTP client cannot be built.
    pub fn new(base_url: &Url, model: impl Into<String>) -> eros::Result<Self> {
        Ok(Self {
            client: client()?,
            url: endpoint(base_url, "chat/completions"),
            model: model.into(),
            key: None,
            options: ChatOptions::default(),
        })
    }

    /// Sends `key` as a bearer token, when set.
    pub fn with_key(self, key: Option<ApiKey>) -> Self {
        Self { key, ..self }
    }

    /// See [`ChatOptions`].
    pub fn with_options(self, options: ChatOptions) -> Self {
        Self { options, ..self }
    }
}

/// The `chat/completions` request body.
pub fn chat_body(model: &str, request: &ChatRequest, options: ChatOptions) -> Value {
    let mut body = json!({
        "model": model,
        "messages": [
            { "role": "system", "content": request.system },
            { "role": "user", "content": request.user },
        ],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    });
    if let Some(effort) = options.reasoning_effort {
        body["reasoning_effort"] = json!(effort.as_str());
    }
    if let Some(schema) = &request.json_schema {
        let json_schema = match options.schema_format {
            SchemaFormat::Openai => json!({ "name": "respond", "strict": true, "schema": schema }),
            SchemaFormat::Bare => schema.clone(),
        };
        body["response_format"] = json!({ "type": "json_schema", "json_schema": json_schema });
    }
    body
}

/// The first choice's `content`, without any reasoning. Reasoning sent in its own field (`reasoning`,
/// `reasoning_content`) is never read; reasoning inlined as a `<think>` block, or ending at a bare `</think>` when
/// the chat template opened the block, is dropped.
///
/// # Errors
///
/// When the response has no `choices[0].message.content` string, as when a reasoning model spends all of
/// `max_tokens` reasoning (`finish_reason` `length`).
pub fn chat_text(response: &Value) -> eros::Result<String> {
    let choice = &response["choices"][0];
    let Some(content) = choice["message"]["content"].as_str() else {
        let finish = choice["finish_reason"].as_str().unwrap_or("absent");
        bail!(
            "response has no choices[0].message.content (finish_reason {finish}); on a reasoning model \
             max_tokens also bounds the reasoning, so `length` means it ran out before answering"
        );
    };
    let answer = match content.rsplit_once("</think>") {
        Some((_, answer)) => answer,
        None if content.trim_start().starts_with("<think>") => "",
        None => content,
    };
    Ok(answer.trim().to_owned())
}

#[async_trait]
impl ChatModel for OpenAiChat {
    fn model(&self) -> ModelRef {
        ModelRef::new(&self.model, self.options.version())
    }

    async fn complete(&self, request: ChatRequest) -> eros::Result<ChatResponse> {
        let body = chat_body(&self.model, &request, self.options);
        let response = post(
            &self.client,
            &self.url,
            &self.model,
            self.key.as_ref(),
            &body,
        )
        .await?;
        Ok(ChatResponse {
            text: chat_text(&response)?,
            mean_logprob: None,
        })
    }
}

/// [`Embedder`] over `POST {base_url}/embeddings`, in batches of 64. Ignores [`EmbedPurpose`].
pub struct OpenAiEmbedder {
    client: reqwest::Client,
    url: String,
    model: String,
    key: Option<ApiKey>,
}

impl OpenAiEmbedder {
    /// The model must produce `EMBEDDING_DIM` dimensions, e.g. `bge-m3` in Ollama or `@cf/baai/bge-m3` on Workers
    /// AI. No key.
    ///
    /// # Errors
    ///
    /// When the HTTP client cannot be built.
    pub fn new(base_url: &Url, model: impl Into<String>) -> eros::Result<Self> {
        Ok(Self {
            client: client()?,
            url: endpoint(base_url, "embeddings"),
            model: model.into(),
            key: None,
        })
    }

    /// Sends `key` as a bearer token, when set.
    pub fn with_key(self, key: Option<ApiKey>) -> Self {
        Self { key, ..self }
    }
}

/// The vectors of an `embeddings` response, reordered by their `index`.
///
/// # Errors
///
/// When the response does not parse, or holds other than `count` vectors of [`EMBEDDING_DIM`].
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
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(EMBED_BATCH) {
            let body = json!({ "model": self.model, "input": batch });
            let response = post(
                &self.client,
                &self.url,
                &self.model,
                self.key.as_ref(),
                &body,
            )
            .await?;
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
        let body = chat_body(
            "qwen3:14b",
            &request(Some(schema.clone())),
            ChatOptions::default(),
        );
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(body["response_format"]["json_schema"]["schema"], schema);
        assert_eq!(body["messages"][0]["role"], "system");
        assert!(
            chat_body("qwen3:14b", &request(None), ChatOptions::default())
                .get("response_format")
                .is_none()
        );
    }

    #[test]
    fn options_that_shape_the_output_are_in_the_version() {
        assert_eq!(ChatOptions::default().version(), "openai-compatible");
        let options = ChatOptions {
            reasoning_effort: Some(ReasoningEffort::Medium),
            schema_format: SchemaFormat::Bare,
        };
        assert_eq!(
            options.version(),
            "openai-compatible;reasoning_effort=medium;schema=bare"
        );
    }

    #[test]
    fn reasoning_never_reaches_the_answer() {
        let response =
            |content: &str| json!({ "choices": [{ "message": { "content": content } }] });
        assert_eq!(
            chat_text(&response("<think>\nhmm\n</think>\n{\"claims\": []}")).unwrap(),
            "{\"claims\": []}"
        );
        assert_eq!(
            chat_text(&response("the template opened it\n</think>\n{\"a\": 1}")).unwrap(),
            "{\"a\": 1}"
        );
        assert_eq!(chat_text(&response(" {\"a\": 1} ")).unwrap(), "{\"a\": 1}");
        assert_eq!(chat_text(&response("<think>never closed")).unwrap(), "");
        let separate = json!({ "choices": [{ "message": {
            "content": "{\"a\": 1}", "reasoning": "hmm", "reasoning_content": "hmm"
        } }] });
        assert_eq!(chat_text(&separate).unwrap(), "{\"a\": 1}");
    }

    #[test]
    fn reasoning_that_used_up_max_tokens_names_the_cause() {
        let response = json!({ "choices": [{
            "message": { "content": null, "reasoning": "hmm" }, "finish_reason": "length"
        }] });
        let error = format!("{:#?}", chat_text(&response).unwrap_err());
        assert!(
            error.contains("finish_reason length") && error.contains("max_tokens"),
            "{error}"
        );
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

        let chat = OpenAiChat::new(
            &format!("{}/v1/", server.uri()).parse().unwrap(),
            "qwen3:14b",
        )
        .unwrap();
        let response = chat.complete(request(Some(json!({})))).await.unwrap();
        assert_eq!(response.text, "{\"claims\": []}");
    }

    #[tokio::test]
    async fn errors_carry_the_status_and_body_but_never_the_key() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(404).set_body_string("model \"qwen3:14b\" not found"),
            )
            .mount(&server)
            .await;

        let error = OpenAiChat::new(&server.uri().parse().unwrap(), "qwen3:14b")
            .unwrap()
            .with_key(Some("sk-secret".parse().unwrap()))
            .complete(request(None))
            .await
            .unwrap_err();
        let report = format!("{error:?}");
        assert!(
            report.contains("not found") && report.contains("for model qwen3:14b"),
            "{report}"
        );
        assert!(!report.contains("sk-secret"), "{report}");
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

        let embedder =
            OpenAiEmbedder::new(&format!("{}/v1", server.uri()).parse().unwrap(), "bge-m3")
                .unwrap();
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
