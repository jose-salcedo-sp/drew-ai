//! System One: decision questions over a text, answered with calibrated probabilities.
//!
//! The wire format is TypeSafe's `POST /v1/systemone`, served locally by Kev (github.com/jaredpalmer/kev).

use std::{collections::BTreeMap, time::Duration};

use async_trait::async_trait;
use eros::ensure;
use serde::{Deserialize, Serialize, Serializer, ser::SerializeMap};

use super::ModelRef;

#[derive(Clone, Debug, clap::Args)]
#[group(id = "system_one")]
pub struct SystemOneArgs {
    /// System One server, e.g. a local Kev at `http://127.0.0.1:8009`. Unset: every question is classified `open`.
    #[arg(long, env = "DREW_SYSTEM_ONE_URL")]
    pub system_one_url: Option<String>,
    /// Model name sent with each request.
    #[arg(long, env = "DREW_SYSTEM_ONE_MODEL", default_value = "kev-latest")]
    pub system_one_model: String,
    /// Bearer key, when the server sets `KEV_API_KEY`.
    #[arg(long, env = "DREW_SYSTEM_ONE_KEY", hide_env_values = true)]
    pub system_one_key: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DecisionRequest {
    /// The text every question is asked about.
    pub state: String,
    pub model: String,
    /// Keyed by an id the model never sees.
    pub questions: BTreeMap<String, Decision>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Decision {
    /// Yes or no.
    Noul { instructions: String },
    /// One of the options, each with an optional description.
    Choice {
        instructions: String,
        criteria: Options,
    },
    /// Levels ordered from lowest to highest.
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
}

/// Choice options in the order given: option order can change an answer, so it stays deterministic.
#[derive(Clone, Debug, PartialEq)]
pub struct Options(pub Vec<(String, Option<String>)>);

impl Serialize for Options {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (option, description) in &self.0 {
            map.serialize_entry(option, description)?;
        }
        map.end()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionResponse {
    pub answers: BTreeMap<String, DecisionAnswer>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DecisionAnswer {
    /// Probability of yes.
    Noul { noul: f32 },
    Choice {
        choice: String,
        confidence: f32,
        probabilities: BTreeMap<String, f32>,
    },
    Score {
        /// Mean level index, from 0.
        score: f32,
        confidence: f32,
        probabilities: BTreeMap<String, f32>,
    },
}

#[async_trait]
pub trait SystemOne: Send + Sync {
    fn model(&self) -> ModelRef;

    async fn decide(&self, request: DecisionRequest) -> eros::Result<DecisionResponse>;
}

pub struct HttpSystemOne {
    client: reqwest::Client,
    url: String,
    model: String,
    key: Option<String>,
}

impl HttpSystemOne {
    /// `None` when no server is configured.
    pub fn from_args(args: &SystemOneArgs) -> Option<Self> {
        let base = args.system_one_url.as_deref()?.trim_end_matches('/');
        Some(Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("a client with only a timeout set always builds"),
            url: format!("{base}/v1/systemone"),
            model: args.system_one_model.clone(),
            key: args.system_one_key.clone(),
        })
    }
}

#[async_trait]
impl SystemOne for HttpSystemOne {
    fn model(&self) -> ModelRef {
        ModelRef::new(&self.model, "system-one-v1")
    }

    async fn decide(&self, request: DecisionRequest) -> eros::Result<DecisionResponse> {
        self.post(request).await
    }
}

impl HttpSystemOne {
    #[eros::context("POST {}", self.url)]
    async fn post(&self, request: DecisionRequest) -> eros::Result<DecisionResponse> {
        let request = DecisionRequest {
            model: self.model.clone(),
            ..request
        };
        let mut post = self.client.post(&self.url).json(&request);
        if let Some(key) = &self.key {
            post = post.bearer_auth(key);
        }
        let response = post.send().await?;
        let status = response.status();
        let text = response.text().await?;
        ensure!(status.is_success(), "returned {status}: {text}");
        Ok(serde_json::from_str(&text)?)
    }
}

#[cfg(any(test, feature = "testing"))]
pub use super::fakes::ScriptedSystemOne;

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_json, header, method, path},
    };

    use super::*;

    fn request() -> DecisionRequest {
        DecisionRequest {
            state: "Does AGR_USERS store role assignments?".into(),
            model: String::new(),
            questions: BTreeMap::from([(
                "category".into(),
                Decision::Choice {
                    instructions: "What kind of answer?".into(),
                    criteria: Options(vec![
                        ("open".into(), Some("Free-form".into())),
                        ("boolean".into(), None),
                    ]),
                },
            )]),
        }
    }

    #[tokio::test]
    async fn posts_the_typesafe_wire_format_and_parses_answers() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .and(header("authorization", "Bearer secret"))
            .and(body_json(json!({
                "state": "Does AGR_USERS store role assignments?",
                "model": "kev-latest",
                "questions": { "category": {
                    "type": "choice",
                    "instructions": "What kind of answer?",
                    "criteria": { "open": "Free-form", "boolean": null }
                } }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "kev-latest",
                "answers": { "category": {
                    "type": "choice", "choice": "boolean", "confidence": 0.8,
                    "probabilities": { "boolean": 0.9, "open": 0.1 }
                } },
                "usage": { "input_tokens": 40, "output_tokens": 20 },
                "latency_ms": 120
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = HttpSystemOne::from_args(&SystemOneArgs {
            system_one_url: Some(format!("{}/", server.uri())),
            system_one_model: "kev-latest".into(),
            system_one_key: Some("secret".into()),
        })
        .unwrap();
        let response = client.decide(request()).await.unwrap();
        assert!(matches!(
            &response.answers["category"],
            DecisionAnswer::Choice { choice, .. } if choice == "boolean"
        ));
    }

    #[test]
    fn options_keep_their_order() {
        let options = Options(vec![("z".into(), None), ("a".into(), Some("first".into()))]);
        assert_eq!(
            serde_json::to_string(&options).unwrap(),
            r#"{"z":null,"a":"first"}"#
        );
    }

    #[test]
    fn unconfigured_means_none() {
        let args = SystemOneArgs {
            system_one_url: None,
            system_one_model: "kev-latest".into(),
            system_one_key: None,
        };
        assert!(HttpSystemOne::from_args(&args).is_none());
    }
}
