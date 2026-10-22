//! [`SystemOne`] on Workers AI: Cloudflare's Clef models, e.g. `@cf/cloudflare/clef-flash`, behind
//! `POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{model}` with a bearer API token.
//!
//! The body is the System One [`DecisionRequest`], with `model` set to the selector the model page names (`clef`
//! or `clef-flash`, the id's last segment). Like every Workers AI REST call, the reply is wrapped as
//! `{"result": ..., "success": ..., "errors": [...], "messages": [...]}`; `result` is the System One answer set.
//! Workers AI's chat and embedding models speak the OpenAI wire format, so [`super::openai`] serves them.

use std::{fmt, str::FromStr, time::Duration};

use async_trait::async_trait;
use eros::{Context, bail};
use reqwest::Url;
use serde::Deserialize;

use super::{
    ApiKey, ModelRef,
    system_one::{DecisionRequest, DecisionResponse, SystemOne},
};

/// The Cloudflare API root that account routes hang off.
const CLOUDFLARE_API: &str = "https://api.cloudflare.com/client/v4";
/// Workers AI's own limit on a synchronous run.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Clef Flash's context window; its text `state` is truncated to fit, without an error.
const CLEF_FLASH_CONTEXT_TOKENS: u64 = 24_576;
/// A request whose input came within this many tokens of its model's context may have lost the end of its state.
const TRUNCATION_MARGIN: u64 = 512;

/// A Workers AI model id, `@cf/{vendor}/{name}` (or `@hf/...`), e.g. `@cf/cloudflare/clef-flash`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkersAiModel {
    id: String,
    name: String,
}

impl WorkersAiModel {
    /// The full id, used in the route and recorded as the [`ModelRef`] id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The id's last segment, sent as the request's `model` selector.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The context window, for the models whose window is documented.
    fn context_tokens(&self) -> Option<u64> {
        match self.id.as_str() {
            "@cf/cloudflare/clef-flash" => Some(CLEF_FLASH_CONTEXT_TOKENS),
            _ => None,
        }
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
pub struct NotAWorkersAiModel(String);

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
    api_base: String,
    account_id: String,
    model: WorkersAiModel,
    key: ApiKey,
}

impl WorkersAiSystemOne {
    /// Calls the public Cloudflare API; see [`WorkersAiSystemOne::with_api_base`].
    ///
    /// # Errors
    ///
    /// When the HTTP client cannot be built.
    pub fn new(account_id: &str, model: WorkersAiModel, key: ApiKey) -> eros::Result<Self> {
        Ok(Self {
            client: client()?,
            api_base: CLOUDFLARE_API.to_owned(),
            account_id: account_id.to_owned(),
            model,
            key,
        })
    }

    /// Another API root in place of `https://api.cloudflare.com/client/v4`, such as a test server or a gateway.
    pub fn with_api_base(self, api_base: &Url) -> Self {
        Self {
            api_base: api_base.as_str().trim_end_matches('/').to_owned(),
            ..self
        }
    }

    fn route(&self) -> String {
        format!(
            "{}/accounts/{}/ai/run/{}",
            self.api_base,
            self.account_id,
            self.model.id()
        )
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
    #[eros::context("POST {}", self.route())]
    async fn post(&self, request: DecisionRequest) -> eros::Result<DecisionResponse> {
        let request = DecisionRequest {
            model: self.model.name().to_owned(),
            ..request
        };
        let response = self
            .client
            .post(self.route())
            .bearer_auth(self.key.expose())
            .json(&request)
            .send()
            .await?;
        let status = response.status();
        let text = response.text().await?;
        let envelope = serde_json::from_str::<Envelope>(&text)
            .with_context(|| format!("{} returned {status}: {text}", self.model))?;
        match envelope {
            Envelope {
                success: true,
                result: Some(result),
                ..
            } if status.is_success() => {
                if let (Some(usage), Some(context)) = (&result.usage, self.model.context_tokens())
                    && usage.input_tokens + TRUNCATION_MARGIN >= context
                {
                    tracing::warn!(
                        model = %self.model,
                        input_tokens = usage.input_tokens,
                        "the System One state may have been truncated to fit the {context}-token context"
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
        assert_eq!(model.context_tokens(), Some(CLEF_FLASH_CONTEXT_TOKENS));
        let clef: WorkersAiModel = "@cf/cloudflare/clef".parse().unwrap();
        assert_eq!(clef.context_tokens(), None);
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

    #[test]
    fn the_route_uses_the_public_api_unless_overridden() {
        let client = WorkersAiSystemOne::new(
            "acc",
            "@cf/cloudflare/clef-flash".parse().unwrap(),
            "token".parse().unwrap(),
        )
        .unwrap();
        assert_eq!(
            client.route(),
            "https://api.cloudflare.com/client/v4/accounts/acc/ai/run/@cf/cloudflare/clef-flash"
        );
        let client = client.with_api_base(&"http://127.0.0.1:9/client/v4/".parse().unwrap());
        assert_eq!(
            client.route(),
            "http://127.0.0.1:9/client/v4/accounts/acc/ai/run/@cf/cloudflare/clef-flash"
        );
    }
}
