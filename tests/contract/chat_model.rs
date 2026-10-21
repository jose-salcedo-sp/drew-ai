//! Replies are model output, so the contract is the transport: the request reaches the model as given, and a text
//! or structured reply comes back as text.

use std::sync::Arc;

use drew::llm::{ChatModel, ChatRequest};
use futures::future::BoxFuture;
use serde_json::{Value, json};
use wiremock::MockServer;

/// What the model is set up to answer.
#[derive(Clone)]
pub enum Reply {
    Text(String),
    /// A structured answer to a request with a `json_schema`.
    Json(Value),
}

/// An adapter set up to give one reply, and every request it sent, read back from the wire.
pub struct Adapter {
    pub chat: Arc<dyn ChatModel>,
    pub sent: Box<dyn Fn() -> BoxFuture<'static, Vec<ChatRequest>> + Send + Sync>,
}

fn request(json_schema: Option<Value>) -> ChatRequest {
    ChatRequest {
        system: "Answer from the sources.".into(),
        user: "<question>Which table maps users to roles?</question>".into(),
        max_tokens: 512,
        temperature: 0.25,
        json_schema,
    }
}

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": { "claims": { "type": "array", "items": { "type": "string" } } },
        "required": ["claims"],
    })
}

pub async fn a_text_reply_comes_back_as_its_text(
    make: impl AsyncFnOnce(Reply) -> Adapter,
) -> eros::Result<()> {
    let text = "AGR_USERS maps users to roles.";
    let adapter = make(Reply::Text(text.into())).await;

    let response = adapter.chat.complete(request(None)).await?;

    assert_eq!(response.text, text);
    Ok(())
}

pub async fn a_structured_reply_comes_back_as_json_text(
    make: impl AsyncFnOnce(Reply) -> Adapter,
) -> eros::Result<()> {
    let answer = json!({ "claims": ["AGR_USERS maps users to roles."] });
    let adapter = make(Reply::Json(answer.clone())).await;

    let response = adapter.chat.complete(request(Some(schema()))).await?;

    assert_eq!(serde_json::from_str::<Value>(&response.text)?, answer);
    Ok(())
}

pub async fn the_request_reaches_the_model_as_given(
    make: impl AsyncFn(Reply) -> Adapter,
) -> eros::Result<()> {
    for json_schema in [None, Some(schema())] {
        let reply = match json_schema {
            Some(_) => Reply::Json(json!({ "claims": [] })),
            None => Reply::Text("ok".into()),
        };
        let adapter = make(reply).await;

        adapter.chat.complete(request(json_schema.clone())).await?;

        assert_eq!((adapter.sent)().await, [request(json_schema)]);
    }
    Ok(())
}

macro_rules! chat_model_contract {
    ($make:expr) => {
        #[tokio::test]
        async fn a_text_reply_comes_back_as_its_text() -> eros::Result<()> {
            crate::chat_model::a_text_reply_comes_back_as_its_text($make).await
        }

        #[tokio::test]
        async fn a_structured_reply_comes_back_as_json_text() -> eros::Result<()> {
            crate::chat_model::a_structured_reply_comes_back_as_json_text($make).await
        }

        #[tokio::test]
        async fn the_request_reaches_the_model_as_given() -> eros::Result<()> {
            crate::chat_model::the_request_reaches_the_model_as_given($make).await
        }
    };
}

/// Reads back every request `server` received, parsed by `parse`.
fn sent_to(
    server: MockServer,
    parse: fn(&Value) -> ChatRequest,
) -> Box<dyn Fn() -> BoxFuture<'static, Vec<ChatRequest>> + Send + Sync> {
    let server = Arc::new(server);
    Box::new(move || {
        let server = Arc::clone(&server);
        Box::pin(async move {
            server
                .received_requests()
                .await
                .unwrap()
                .iter()
                .map(|request| parse(&request.body_json().unwrap()))
                .collect()
        })
    })
}

fn as_f32(value: &Value) -> f32 {
    value.as_f64().unwrap() as f32
}

mod scripted {
    use std::sync::Arc;

    use drew::llm::fakes::ScriptedChat;

    use super::{Adapter, Reply};

    async fn scripted(reply: Reply) -> Adapter {
        let text = match reply {
            Reply::Text(text) => text,
            Reply::Json(value) => value.to_string(),
        };
        let fake = Arc::new(ScriptedChat::replying([text]));
        Adapter {
            chat: Arc::<ScriptedChat>::clone(&fake),
            sent: Box::new(move || {
                let fake = Arc::clone(&fake);
                Box::pin(async move { fake.requests() })
            }),
        }
    }

    chat_model_contract!(scripted);
}

mod openai {
    use std::sync::Arc;

    use drew::llm::{ChatRequest, openai::OpenAiChat};
    use serde_json::{Value, json};
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    use super::{Adapter, Reply, as_f32, sent_to};

    fn sent(body: &Value) -> ChatRequest {
        ChatRequest {
            system: body["messages"][0]["content"].as_str().unwrap().into(),
            user: body["messages"][1]["content"].as_str().unwrap().into(),
            max_tokens: body["max_tokens"].as_u64().unwrap().try_into().unwrap(),
            temperature: as_f32(&body["temperature"]),
            json_schema: body
                .get("response_format")
                .map(|format| format["json_schema"]["schema"].clone()),
        }
    }

    async fn openai(reply: Reply) -> Adapter {
        let content = match reply {
            Reply::Text(text) => text,
            Reply::Json(value) => value.to_string(),
        };
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{ "message": { "role": "assistant", "content": content } }]
            })))
            .mount(&server)
            .await;
        Adapter {
            chat: Arc::new(OpenAiChat::new(format!("{}/v1", server.uri()), "qwen3:14b")),
            sent: sent_to(server, sent),
        }
    }

    chat_model_contract!(openai);
}

mod bedrock {
    use std::sync::Arc;

    use drew::llm::{ChatModel, ChatRequest, bedrock_chat::BedrockChat};
    use serde_json::{Value, json};
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path_regex},
    };

    use super::{Adapter, Reply, as_f32, request, sent_to};

    const CLAUDE: &str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0";

    fn sent(body: &Value) -> ChatRequest {
        ChatRequest {
            system: body["system"][0]["text"].as_str().unwrap().into(),
            user: body["messages"][0]["content"][0]["text"]
                .as_str()
                .unwrap()
                .into(),
            max_tokens: body["inferenceConfig"]["maxTokens"]
                .as_u64()
                .unwrap()
                .try_into()
                .unwrap(),
            temperature: as_f32(&body["inferenceConfig"]["temperature"]),
            json_schema: body
                .get("toolConfig")
                .map(|tools| tools["tools"][0]["toolSpec"]["inputSchema"]["json"].clone()),
        }
    }

    fn converse(reply: Reply) -> Value {
        let (content, stop) = match reply {
            Reply::Text(text) => (json!({ "text": text }), "end_turn"),
            Reply::Json(input) => (
                json!({ "toolUse": { "toolUseId": "tooluse_1", "name": "respond", "input": input } }),
                "tool_use",
            ),
        };
        json!({
            "output": { "message": { "role": "assistant", "content": [content] } },
            "stopReason": stop,
            "usage": { "inputTokens": 40, "outputTokens": 20, "totalTokens": 60 },
            "metrics": { "latencyMs": 120 },
        })
    }

    async fn server(response: ResponseTemplate) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex("^/model/[^/]+/converse$"))
            .respond_with(response)
            .mount(&server)
            .await;
        server
    }

    async fn bedrock(reply: Reply) -> Adapter {
        let server = server(ResponseTemplate::new(200).set_body_json(converse(reply))).await;
        Adapter {
            chat: Arc::new(BedrockChat::new(crate::bedrock::client(&server), CLAUDE)),
            sent: sent_to(server, sent),
        }
    }

    chat_model_contract!(bedrock);

    #[tokio::test]
    async fn a_rejected_call_names_the_model_and_the_reason() {
        let server = server(
            ResponseTemplate::new(400)
                .insert_header("x-amzn-errortype", "ValidationException")
                .set_body_json(json!({ "message": "max_tokens exceeds the model limit" })),
        )
        .await;

        let error = BedrockChat::new(crate::bedrock::client(&server), CLAUDE)
            .complete(request(None))
            .await
            .unwrap_err();

        let report = format!("{error:#?}");
        assert!(
            report.contains(CLAUDE) && report.contains("max_tokens exceeds the model limit"),
            "{report}"
        );
    }
}
