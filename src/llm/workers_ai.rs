//! [`SystemOne`] on Workers AI: Cloudflare's Clef models, e.g. `@cf/cloudflare/clef-flash`, behind
//! `POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{model}` with a bearer API token.
//!
//! The body is the System One [`DecisionRequest`], with `model` set to the selector the model page names (`clef`
//! or `clef-flash`, the id's last segment). Like every Workers AI REST call, the reply is wrapped as
//! `{"result": ..., "success": ..., "errors": [...], "messages": [...]}`; `result` is the System One answer set.
//! Workers AI's chat and embedding models speak the OpenAI wire format, so [`super::openai`] serves them.

use std::{fmt, str::FromStr, time::Duration};

use async_trait::async_trait;
use eros::bail;
use serde::Deserialize;

use super::{
    ApiKey, ModelRef,
    system_one::{DecisionRequest, DecisionResponse, SystemOne},
};

/// The Cloudflare API root that account routes hang off.
pub(crate) const CLOUDFLARE_API: &str = "https://api.cloudflare.com/client/v4";
/// Workers AI's own limit on a synchronous run.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Clef Flash's context window. The text `state` is truncated to fit it, without an error.
const CONTEXT_TOKENS: u64 = 24_576;
/// A request whose input came within this many tokens of [`CONTEXT_TOKENS`] may have lost the end of its state.
const TRUNCATION_MARGIN: u64 = 512;

/// A Workers AI model id, `@cf/{vendor}/{name}` (or `@hf/...`), e.g. `@cf/cloudflare/clef-flash`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct WorkersAiModel {
    id: String,
    name: String,
}

impl WorkersAiModel {
    /// The full id, used in the route and recorded as the [`ModelRef`] id.
    fn id(&self) -> &str {
        &self.id
    }

    /// The id's last segment, sent as the request's `model` selector.
    fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for WorkersAiModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.id)
    }
}

/// A model name that is not a Workers AI model id.
#[derive(Debug, thiserror::Error)]
#[error("expected a Workers AI model id such as @cf/cloudflare/clef-flash, got {0:?}")]
struct NotAWorkersAiModel(String);

impl FromStr for WorkersAiModel {
    type Err = NotAWorkersAiModel;

    fn from_str(id: &str) -> Result<Self, Self::Err> {
        let mut parts = id.split('/');
        match (parts.next(), parts.next(), parts.next(), parts.next()) {
            (Some("@cf" | "@hf"), Some(vendor), Some(name), None)
                if !vendor.is_empty() && !name.is_empty() =>
            {
                Ok(Self {
                    id: id.to_owned(),
                    name: name.to_owned(),
                })
            }
            _ => Err(NotAWorkersAiModel(id.to_owned())),
        }
    }
}

/// [`SystemOne`] over Workers AI's `ai/run` route.
pub struct WorkersAiSystemOne {
    client: reqwest::Client,
    url: String,
    model: WorkersAiModel,
    key: ApiKey,
}

impl WorkersAiSystemOne {
    /// `api_base` is the Cloudflare API root, `https://api.cloudflare.com/client/v4` outside tests; `model` a Workers
    /// AI model id such as `@cf/cloudflare/clef-flash`.
    ///
    /// # Errors
    ///
    /// When `model` is not a Workers AI model id, or the HTTP client cannot be built.
    pub fn new(api_base: &str, account_id: &str, model: &str, key: ApiKey) -> eros::Result<Self> {
        let model: WorkersAiModel = model.parse()?;
        Ok(Self {
            client: client()?,
            url: format!(
                "{}/accounts/{account_id}/ai/run/{}",
                api_base.trim_end_matches('/'),
                model.id()
            ),
            model,
            key,
        })
    }
}

#[eros::context("building the Workers AI HTTP client")]
fn client() -> eros::Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT)
        .build()?)
}

#[async_trait]
impl SystemOne for WorkersAiSystemOne {
    fn model(&self) -> ModelRef {
        ModelRef::new(self.model.id(), "workers-ai-run")
    }

    async fn decide(&self, request: DecisionRequest) -> eros::Result<DecisionResponse> {
        self.post(request).await
    }
}

/// Workers AI's REST envelope.
#[derive(Deserialize)]
struct Envelope {
    #[serde(default)]
    result: Option<RunResult>,
    #[serde(default)]
    success: bool,
    #[serde(default)]
    errors: Vec<ApiMessage>,
}

#[derive(Deserialize)]
struct RunResult {
    #[serde(flatten)]
    response: DecisionResponse,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Usage {
    input_tokens: u64,
}

#[derive(Deserialize)]
struct ApiMessage {
    code: Option<i64>,
    message: String,
}

fn messages(errors: &[ApiMessage]) -> String {
    if errors.is_empty() {
        return "no error message".to_owned();
    }
    errors
        .iter()
        .map(|error| match error.code {
            Some(code) => format!("{code}: {}", error.message),
            None => error.message.clone(),
        })
        .collect::<Vec<_>>()
        .join("; ")
}

impl WorkersAiSystemOne {
    #[eros::context("POST {}", self.url)]
    async fn post(&self, request: DecisionRequest) -> eros::Result<DecisionResponse> {
        let request = DecisionRequest {
            model: self.model.name().to_owned(),
            ..request
        };
        let response = self
            .client
            .post(&self.url)
            .bearer_auth(self.key.expose())
            .json(&request)
            .send()
            .await?;
        let status = response.status();
        let text = response.text().await?;
        let Ok(envelope) = serde_json::from_str::<Envelope>(&text) else {
            bail!("{} returned {status}: {text}", self.model);
        };
        match envelope {
            Envelope {
                success: true,
                result: Some(result),
                ..
            } if status.is_success() => {
                if let Some(usage) = result.usage
                    && usage.input_tokens + TRUNCATION_MARGIN >= CONTEXT_TOKENS
                {
                    tracing::warn!(
                        model = %self.model,
                        input_tokens = usage.input_tokens,
                        "the System One state may have been truncated to fit the {CONTEXT_TOKENS}-token context"
                    );
                }
                Ok(result.response)
            }
            Envelope { errors, .. } => {
                bail!("{} failed with {status}: {}", self.model, messages(&errors))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_id_has_a_scope_a_vendor_and_a_name() {
        let model: WorkersAiModel = "@cf/cloudflare/clef-flash".parse().unwrap();
        assert_eq!(model.id(), "@cf/cloudflare/clef-flash");
        assert_eq!(model.name(), "clef-flash");
        for bad in [
            "clef-flash",
            "kev-latest",
            "@cf/clef-flash",
            "@cf//clef",
            "@cf/cloudflare/",
            "@cf/a/b/c",
        ] {
            assert!(bad.parse::<WorkersAiModel>().is_err(), "{bad}");
        }
    }
}
