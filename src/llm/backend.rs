//! Which backend serves each model port, read once from the command line and environment.
//!
//! Each port names its backend, with no default: `DREW_EMBED_BACKEND`, `DREW_EXPLAIN_BACKEND` and
//! `DREW_SYSTEM_ONE_BACKEND`. A missing env file is then a startup error, not a silent fall back to Bedrock. The
//! flat command-line fields are parsed into one enum per port whose variants hold exactly the settings that backend
//! uses; a missing, unparsable or unused setting is an error naming its variable. Each arm destructures every field,
//! so a new field does not compile until every backend accepts or rejects it. [`embedder`], [`explain_model`] and
//! [`system_one_client`] map those enums to the port traits, which is all the rest of the crate sees.
//!
//! Workers AI's chat and embedding models are `openai-compatible`: its `/ai/v1` routes take the same requests with a
//! bearer key, and the account id is part of the URL. Its System One is the separate `workers-ai` backend, because
//! `ai/run` differs from `/v1/systemone` in route, model selector and response envelope.

use std::sync::Arc;

use reqwest::Url;

use super::{
    ApiKey, ChatModel, Embedder,
    bedrock_chat::{BedrockChat, aws_environment, bedrock_endpoint},
    bedrock_embed::BedrockEmbedder,
    openai::{
        CHAT_PATH, ChatOptions, EMBEDDINGS_PATH, OpenAiChat, OpenAiEmbedder, ReasoningEffort,
        SchemaFormat,
    },
    system_one::{DECIDE_PATH, HttpSystemOne, SystemOne},
    workers_ai::{WorkersAiModel, WorkersAiSystemOne, cloudflare_api, run_path},
};

/// The embedding model and the backend that serves it.
#[derive(Clone, Debug, clap::Args)]
pub struct EmbedArgs {
    /// Which backend serves embeddings.
    #[arg(long, env = "DREW_EMBED_BACKEND", value_enum)]
    pub embed_backend: EmbedBackendKind,
    /// Embedding model, with 1024 dimensions: on Bedrock Titan Text Embeddings v2 or Cohere Embed v3, otherwise the
    /// server's name, e.g. `bge-m3` in Ollama or `@cf/baai/bge-m3` on Workers AI.
    #[arg(long, env = "DREW_EMBED_MODEL")]
    pub embed_model: String,
    /// `openai-compatible` only: base URL, e.g. `http://localhost:11434/v1`, or
    /// `https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1`.
    #[arg(long, env = "DREW_EMBED_URL")]
    pub embed_url: Option<Url>,
    /// `openai-compatible` only: bearer key, e.g. a Workers AI API token.
    #[arg(long, env = "DREW_EMBED_KEY", hide_env_values = true)]
    pub embed_key: Option<ApiKey>,
}

/// `DREW_EMBED_BACKEND`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum EmbedBackendKind {
    /// Amazon Bedrock; region and credentials from the default AWS chain.
    Bedrock,
    /// `POST {url}/embeddings`: Ollama, vLLM, llama.cpp, or Workers AI.
    OpenaiCompatible,
}

/// The `Explain` LLM and the backend that serves it.
#[derive(Clone, Debug, clap::Args)]
pub struct ChatArgs {
    /// Which backend serves `Explain`.
    #[arg(long, env = "DREW_EXPLAIN_BACKEND", value_enum)]
    pub explain_backend: ChatBackendKind,
    /// Model for `Explain`: a Bedrock model or inference profile id, e.g. `us.anthropic.claude-sonnet-4-5-20250929-v1:0`,
    /// or the server's model name, e.g. `qwen3:14b` or `@cf/qwen/qwen3.8-27b`.
    #[arg(long, env = "DREW_EXPLAIN_MODEL")]
    pub explain_model: String,
    /// `openai-compatible` only: base URL, e.g. `http://localhost:11434/v1`, or
    /// `https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1`.
    #[arg(long, env = "DREW_EXPLAIN_URL")]
    pub explain_url: Option<Url>,
    /// `openai-compatible` only: bearer key, e.g. a Workers AI API token.
    #[arg(long, env = "DREW_EXPLAIN_KEY", hide_env_values = true)]
    pub explain_key: Option<ApiKey>,
    /// `openai-compatible` only: `reasoning_effort` to send; unset sends none. Use `medium` on Workers AI: `low` with
    /// a `json_schema` hangs there (cloudflare/workers-sdk#16165).
    #[arg(long, env = "DREW_EXPLAIN_REASONING_EFFORT", value_enum)]
    pub explain_reasoning_effort: Option<ReasoningEffort>,
    /// `openai-compatible` only: how the response schema is sent. Unset: `openai`.
    #[arg(long, env = "DREW_EXPLAIN_SCHEMA_FORMAT", value_enum)]
    pub explain_schema_format: Option<SchemaFormat>,
}

/// `DREW_EXPLAIN_BACKEND`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum ChatBackendKind {
    /// Amazon Bedrock's Converse API; region and credentials from the default AWS chain.
    Bedrock,
    /// `POST {url}/chat/completions`: Ollama, vLLM, llama.cpp, or Workers AI.
    OpenaiCompatible,
}

/// System One and the backend that serves it.
#[derive(Clone, Debug, clap::Args)]
#[group(id = "system_one")]
pub struct SystemOneArgs {
    /// Which backend serves System One; `none` classifies every question `open`.
    #[arg(long, env = "DREW_SYSTEM_ONE_BACKEND", value_enum)]
    pub system_one_backend: SystemOneBackendKind,
    /// Model: e.g. `kev-latest`, `clef-flash` on system-one.dev, or `@cf/cloudflare/clef-flash` on Workers AI.
    #[arg(long, env = "DREW_SYSTEM_ONE_MODEL")]
    pub system_one_model: Option<String>,
    /// `system-one` only, and required there: the server root, e.g. `http://127.0.0.1:8009` or
    /// `https://system-one.dev`.
    #[arg(long, env = "DREW_SYSTEM_ONE_URL")]
    pub system_one_url: Option<Url>,
    /// `workers-ai` only: the Cloudflare account id.
    #[arg(long, env = "DREW_SYSTEM_ONE_ACCOUNT_ID")]
    pub system_one_account_id: Option<String>,
    /// Bearer key: Kev's `KEV_API_KEY` or a system-one.dev key, or the Workers AI API token (required there).
    #[arg(long, env = "DREW_SYSTEM_ONE_KEY", hide_env_values = true)]
    pub system_one_key: Option<ApiKey>,
}

/// `DREW_SYSTEM_ONE_BACKEND`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum SystemOneBackendKind {
    /// No System One: every question is classified `open`, and nothing is answered yes or no.
    None,
    /// `POST {url}/v1/systemone`: Kev, or system-one.dev.
    SystemOne,
    /// Clef on Workers AI: `POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}`.
    WorkersAi,
}

/// A backend setting that is missing, unparsable, or set for a backend that does not use it.
#[derive(Debug, thiserror::Error)]
pub(crate) enum InvalidBackend {
    #[error("{var} is required with {backend}")]
    Missing { var: &'static str, backend: String },
    #[error("{var} is set, but {backend} does not use it; unset it")]
    Unused { var: &'static str, backend: String },
    #[error("{var}: {source}")]
    Unparsable {
        var: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// Where a backend sends its requests: scheme, host, port and path. Credentials, query and fragment are dropped, so
/// it is safe to print.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoint(String);

impl Endpoint {
    /// `path` under `base`, as the adapters build their routes.
    pub(crate) fn new(base: &Url, path: &str) -> Self {
        let port = base
            .port()
            .map(|port| format!(":{port}"))
            .unwrap_or_default();
        Self(format!(
            "{}://{}{port}{}/{path}",
            base.scheme(),
            base.host_str().unwrap_or_default(),
            base.path().trim_end_matches('/'),
        ))
    }
}

impl std::fmt::Display for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The backend selection that settings are checked against.
struct Selected(String);

impl Selected {
    fn new(var: &str, value: &impl clap::ValueEnum) -> Self {
        let value = value
            .to_possible_value()
            .expect("no backend variant is skipped, so each has a value");
        Self(format!("{var}={}", value.get_name()))
    }

    fn require<T: Clone>(&self, value: Option<&T>, var: &'static str) -> Result<T, InvalidBackend> {
        value.cloned().ok_or_else(|| InvalidBackend::Missing {
            var,
            backend: self.0.clone(),
        })
    }

    fn reject<T>(&self, value: Option<&T>, var: &'static str) -> Result<(), InvalidBackend> {
        match value {
            Some(_) => Err(InvalidBackend::Unused {
                var,
                backend: self.0.clone(),
            }),
            None => Ok(()),
        }
    }
}

/// [`EmbedArgs`], checked.
pub(crate) enum EmbedBackend {
    Bedrock {
        model: String,
    },
    OpenAiCompatible {
        url: Url,
        model: String,
        key: Option<ApiKey>,
    },
}

impl TryFrom<&EmbedArgs> for EmbedBackend {
    type Error = InvalidBackend;

    fn try_from(args: &EmbedArgs) -> Result<Self, Self::Error> {
        let selected = Selected::new("DREW_EMBED_BACKEND", &args.embed_backend);
        Ok(match args.embed_backend {
            EmbedBackendKind::Bedrock => {
                let EmbedArgs {
                    embed_backend: _,
                    embed_model,
                    embed_url,
                    embed_key,
                } = args;
                selected.reject(embed_url.as_ref(), "DREW_EMBED_URL")?;
                selected.reject(embed_key.as_ref(), "DREW_EMBED_KEY")?;
                Self::Bedrock {
                    model: embed_model.clone(),
                }
            }
            EmbedBackendKind::OpenaiCompatible => {
                let EmbedArgs {
                    embed_backend: _,
                    embed_model,
                    embed_url,
                    embed_key,
                } = args;
                Self::OpenAiCompatible {
                    url: selected.require(embed_url.as_ref(), "DREW_EMBED_URL")?,
                    model: embed_model.clone(),
                    key: embed_key.clone(),
                }
            }
        })
    }
}

impl EmbedBackend {
    /// What `drew models check` prints before calling a backend that depends on ambient settings.
    pub(crate) async fn note(&self) -> Option<String> {
        match self {
            Self::Bedrock { .. } => Some(format!(
                "embeddings on Bedrock (DREW_EMBED_BACKEND=bedrock): {}",
                aws_environment().await
            )),
            Self::OpenAiCompatible { .. } => None,
        }
    }

    /// Where the embedder posts; `None` on Bedrock with no AWS region.
    pub(crate) async fn endpoint(&self) -> Option<Endpoint> {
        match self {
            Self::Bedrock { model } => bedrock_endpoint(&format!("model/{model}/invoke")).await,
            Self::OpenAiCompatible { url, .. } => Some(Endpoint::new(url, EMBEDDINGS_PATH)),
        }
    }

    #[eros::context("setting up the embeddings backend")]
    pub(crate) async fn build(self) -> eros::Result<Arc<dyn Embedder>> {
        Ok(match self {
            Self::Bedrock { model } => Arc::new(BedrockEmbedder::load(&model).await?),
            Self::OpenAiCompatible { url, model, key } => {
                Arc::new(OpenAiEmbedder::new(&url, model)?.with_key(key))
            }
        })
    }
}

/// [`ChatArgs`], checked.
pub(crate) enum ChatBackend {
    Bedrock {
        model: String,
    },
    OpenAiCompatible {
        url: Url,
        model: String,
        key: Option<ApiKey>,
        options: ChatOptions,
    },
}

impl TryFrom<&ChatArgs> for ChatBackend {
    type Error = InvalidBackend;

    fn try_from(args: &ChatArgs) -> Result<Self, Self::Error> {
        let selected = Selected::new("DREW_EXPLAIN_BACKEND", &args.explain_backend);
        Ok(match args.explain_backend {
            ChatBackendKind::Bedrock => {
                let ChatArgs {
                    explain_backend: _,
                    explain_model,
                    explain_url,
                    explain_key,
                    explain_reasoning_effort,
                    explain_schema_format,
                } = args;
                selected.reject(explain_url.as_ref(), "DREW_EXPLAIN_URL")?;
                selected.reject(explain_key.as_ref(), "DREW_EXPLAIN_KEY")?;
                selected.reject(
                    explain_reasoning_effort.as_ref(),
                    "DREW_EXPLAIN_REASONING_EFFORT",
                )?;
                selected.reject(explain_schema_format.as_ref(), "DREW_EXPLAIN_SCHEMA_FORMAT")?;
                Self::Bedrock {
                    model: explain_model.clone(),
                }
            }
            ChatBackendKind::OpenaiCompatible => {
                let ChatArgs {
                    explain_backend: _,
                    explain_model,
                    explain_url,
                    explain_key,
                    explain_reasoning_effort,
                    explain_schema_format,
                } = args;
                Self::OpenAiCompatible {
                    url: selected.require(explain_url.as_ref(), "DREW_EXPLAIN_URL")?,
                    model: explain_model.clone(),
                    key: explain_key.clone(),
                    options: ChatOptions {
                        reasoning_effort: *explain_reasoning_effort,
                        schema_format: explain_schema_format.unwrap_or_default(),
                    },
                }
            }
        })
    }
}

impl ChatBackend {
    /// What `drew models check` prints before calling a backend that depends on ambient settings.
    pub(crate) async fn note(&self) -> Option<String> {
        match self {
            Self::Bedrock { .. } => Some(format!(
                "explain on Bedrock (DREW_EXPLAIN_BACKEND=bedrock): {}",
                aws_environment().await
            )),
            Self::OpenAiCompatible { .. } => None,
        }
    }

    /// Where `Explain` posts; `None` on Bedrock with no AWS region.
    pub(crate) async fn endpoint(&self) -> Option<Endpoint> {
        match self {
            Self::Bedrock { model } => bedrock_endpoint(&format!("model/{model}/converse")).await,
            Self::OpenAiCompatible { url, .. } => Some(Endpoint::new(url, CHAT_PATH)),
        }
    }

    #[eros::context("setting up the Explain backend")]
    pub(crate) async fn build(self) -> eros::Result<Arc<dyn ChatModel>> {
        Ok(match self {
            Self::Bedrock { model } => Arc::new(BedrockChat::load(&model).await),
            Self::OpenAiCompatible {
                url,
                model,
                key,
                options,
            } => Arc::new(
                OpenAiChat::new(&url, model)?
                    .with_key(key)
                    .with_options(options),
            ),
        })
    }
}

/// [`SystemOneArgs`], checked.
pub(crate) enum SystemOneBackend {
    None,
    SystemOne {
        url: Url,
        model: String,
        key: Option<ApiKey>,
    },
    WorkersAi {
        account_id: String,
        model: WorkersAiModel,
        key: ApiKey,
    },
}

impl TryFrom<&SystemOneArgs> for SystemOneBackend {
    type Error = InvalidBackend;

    fn try_from(args: &SystemOneArgs) -> Result<Self, Self::Error> {
        let selected = Selected::new("DREW_SYSTEM_ONE_BACKEND", &args.system_one_backend);
        Ok(match args.system_one_backend {
            SystemOneBackendKind::None => {
                let SystemOneArgs {
                    system_one_backend: _,
                    system_one_model,
                    system_one_url,
                    system_one_account_id,
                    system_one_key,
                } = args;
                selected.reject(system_one_model.as_ref(), "DREW_SYSTEM_ONE_MODEL")?;
                selected.reject(system_one_url.as_ref(), "DREW_SYSTEM_ONE_URL")?;
                selected.reject(system_one_account_id.as_ref(), "DREW_SYSTEM_ONE_ACCOUNT_ID")?;
                selected.reject(system_one_key.as_ref(), "DREW_SYSTEM_ONE_KEY")?;
                Self::None
            }
            SystemOneBackendKind::SystemOne => {
                let SystemOneArgs {
                    system_one_backend: _,
                    system_one_model,
                    system_one_url,
                    system_one_account_id,
                    system_one_key,
                } = args;
                selected.reject(system_one_account_id.as_ref(), "DREW_SYSTEM_ONE_ACCOUNT_ID")?;
                Self::SystemOne {
                    url: selected.require(system_one_url.as_ref(), "DREW_SYSTEM_ONE_URL")?,
                    model: selected.require(system_one_model.as_ref(), "DREW_SYSTEM_ONE_MODEL")?,
                    key: system_one_key.clone(),
                }
            }
            SystemOneBackendKind::WorkersAi => {
                let SystemOneArgs {
                    system_one_backend: _,
                    system_one_model,
                    system_one_url,
                    system_one_account_id,
                    system_one_key,
                } = args;
                selected.reject(system_one_url.as_ref(), "DREW_SYSTEM_ONE_URL")?;
                let model = selected
                    .require(system_one_model.as_ref(), "DREW_SYSTEM_ONE_MODEL")?
                    .parse()
                    .map_err(|error: crate::llm::workers_ai::NotAWorkersAiModel| {
                        InvalidBackend::Unparsable {
                            var: "DREW_SYSTEM_ONE_MODEL",
                            source: Box::new(error),
                        }
                    })?;
                Self::WorkersAi {
                    account_id: selected
                        .require(system_one_account_id.as_ref(), "DREW_SYSTEM_ONE_ACCOUNT_ID")?,
                    model,
                    key: selected.require(system_one_key.as_ref(), "DREW_SYSTEM_ONE_KEY")?,
                }
            }
        })
    }
}

impl SystemOneBackend {
    /// What `drew models check` prints before calling a backend that depends on ambient settings, or instead of
    /// calling none.
    pub(crate) fn note(&self) -> Option<String> {
        match self {
            Self::None => Some(
                "system one: none (DREW_SYSTEM_ONE_BACKEND=none); every question is classified open"
                    .to_owned(),
            ),
            Self::SystemOne { .. } | Self::WorkersAi { .. } => None,
        }
    }

    /// Where System One posts; `None` for [`SystemOneBackend::None`].
    pub(crate) fn endpoint(&self) -> Option<Endpoint> {
        match self {
            Self::None => None,
            Self::SystemOne { url, .. } => Some(Endpoint::new(url, DECIDE_PATH)),
            Self::WorkersAi {
                account_id, model, ..
            } => Some(Endpoint::new(
                &cloudflare_api(),
                &run_path(account_id, model),
            )),
        }
    }

    /// `None` for [`SystemOneBackend::None`].
    #[eros::context("setting up the System One backend")]
    pub(crate) fn build(self) -> eros::Result<Option<Arc<dyn SystemOne>>> {
        Ok(match self {
            Self::None => None,
            Self::SystemOne { url, model, key } => {
                Some(Arc::new(HttpSystemOne::new(&url, model, key)?))
            }
            Self::WorkersAi {
                account_id,
                model,
                key,
            } => Some(Arc::new(WorkersAiSystemOne::new(&account_id, model, key)?)),
        })
    }
}

/// The embedder `DREW_EMBED_BACKEND` selects.
///
/// # Errors
///
/// When a backend setting is missing, unparsable or unused, the HTTP client cannot be built, or on Bedrock when the
/// model is not Titan Text Embeddings v2 or Cohere Embed v3.
pub async fn embedder(args: &EmbedArgs) -> eros::Result<Arc<dyn Embedder>> {
    EmbedBackend::try_from(args)?.build().await
}

/// The `Explain` LLM `DREW_EXPLAIN_BACKEND` selects.
///
/// # Errors
///
/// When a backend setting is missing, unparsable or unused, or the HTTP client cannot be built.
pub async fn explain_model(args: &ChatArgs) -> eros::Result<Arc<dyn ChatModel>> {
    ChatBackend::try_from(args)?.build().await
}

/// The System One `DREW_SYSTEM_ONE_BACKEND` selects; `None` for `none`.
///
/// # Errors
///
/// When a backend setting is missing, unparsable or unused, or the HTTP client cannot be built.
pub fn system_one_client(args: &SystemOneArgs) -> eros::Result<Option<Arc<dyn SystemOne>>> {
    SystemOneBackend::try_from(args)?.build()
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    use super::*;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        embed: EmbedArgs,
        #[command(flatten)]
        chat: ChatArgs,
        #[command(flatten)]
        system_one: SystemOneArgs,
    }

    #[test]
    fn every_backend_is_required_with_no_default() {
        let command = Cli::command();
        for (id, env) in [
            ("embed_backend", "DREW_EMBED_BACKEND"),
            ("explain_backend", "DREW_EXPLAIN_BACKEND"),
            ("system_one_backend", "DREW_SYSTEM_ONE_BACKEND"),
            ("explain_model", "DREW_EXPLAIN_MODEL"),
            ("embed_model", "DREW_EMBED_MODEL"),
        ] {
            let arg = command
                .get_arguments()
                .find(|arg| arg.get_id() == id)
                .unwrap();
            assert_eq!(arg.get_env().and_then(|env| env.to_str()), Some(env));
            assert!(arg.is_required_set(), "{id}");
            assert!(arg.get_default_values().is_empty(), "{id}");
        }
    }

    #[test]
    fn keys_are_hidden_from_help() {
        let command = Cli::command();
        for id in ["embed_key", "explain_key", "system_one_key"] {
            let arg = command
                .get_arguments()
                .find(|arg| arg.get_id() == id)
                .unwrap();
            assert!(arg.is_hide_env_values_set(), "{id}");
        }
    }

    fn try_parse(args: &[&str]) -> Result<Cli, clap::Error> {
        let base = [
            "drew",
            "--embed-backend",
            "openai-compatible",
            "--embed-model",
            "bge-m3",
            "--explain-model",
            "qwen3:14b",
        ];
        Cli::try_parse_from(base.iter().chain(args))
    }

    fn parse(args: &[&str]) -> Cli {
        try_parse(args).unwrap()
    }

    fn missing(error: InvalidBackend) -> &'static str {
        match error {
            InvalidBackend::Missing { var, .. } => var,
            other => panic!("expected a missing setting, got {other}"),
        }
    }

    #[test]
    fn a_backend_names_the_setting_it_lacks() {
        let cli = parse(&[
            "--explain-backend",
            "openai-compatible",
            "--system-one-backend",
            "workers-ai",
            "--system-one-model",
            "@cf/cloudflare/clef-flash",
            "--system-one-key",
            "token",
        ]);
        assert_eq!(
            missing(EmbedBackend::try_from(&cli.embed).err().unwrap()),
            "DREW_EMBED_URL"
        );
        assert_eq!(
            missing(ChatBackend::try_from(&cli.chat).err().unwrap()),
            "DREW_EXPLAIN_URL"
        );
        let error = SystemOneBackend::try_from(&cli.system_one).err().unwrap();
        assert_eq!(
            error.to_string(),
            "DREW_SYSTEM_ONE_ACCOUNT_ID is required with DREW_SYSTEM_ONE_BACKEND=workers-ai"
        );
        assert_eq!(missing(error), "DREW_SYSTEM_ONE_ACCOUNT_ID");
    }

    #[test]
    fn a_setting_for_another_backend_is_rejected() {
        let cli = parse(&[
            "--explain-backend",
            "bedrock",
            "--explain-reasoning-effort",
            "medium",
            "--system-one-backend",
            "none",
            "--system-one-url",
            "http://127.0.0.1:8009",
        ]);
        assert_eq!(
            ChatBackend::try_from(&cli.chat).err().unwrap().to_string(),
            "DREW_EXPLAIN_REASONING_EFFORT is set, but DREW_EXPLAIN_BACKEND=bedrock does not use it; unset it"
        );
        assert!(matches!(
            SystemOneBackend::try_from(&cli.system_one),
            Err(InvalidBackend::Unused {
                var: "DREW_SYSTEM_ONE_URL",
                ..
            })
        ));
    }

    #[test]
    fn workers_ai_needs_a_workers_ai_model_id_named_by_its_variable() {
        let workers_ai = |model: &str| {
            parse(&[
                "--explain-backend",
                "openai-compatible",
                "--system-one-backend",
                "workers-ai",
                "--system-one-model",
                model,
                "--system-one-account-id",
                "acc",
                "--system-one-key",
                "token",
            ])
        };
        let cli = workers_ai("@cf/cloudflare/clef-flash");
        let backend = SystemOneBackend::try_from(&cli.system_one).unwrap();
        assert!(matches!(backend, SystemOneBackend::WorkersAi { .. }));
        assert_eq!(
            backend.endpoint().unwrap().to_string(),
            "https://api.cloudflare.com/client/v4/accounts/acc/ai/run/@cf/cloudflare/clef-flash"
        );
        assert!(system_one_client(&cli.system_one).unwrap().is_some());

        let error = SystemOneBackend::try_from(&workers_ai("kev-latest").system_one)
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.starts_with("DREW_SYSTEM_ONE_MODEL: ")
                && error.contains("@cf/cloudflare/clef-flash"),
            "{error}"
        );
    }

    #[test]
    fn workers_ai_rejects_a_system_one_url_left_over_from_kev() {
        let cli = parse(&[
            "--explain-backend",
            "openai-compatible",
            "--system-one-backend",
            "workers-ai",
            "--system-one-model",
            "@cf/cloudflare/clef-flash",
            "--system-one-account-id",
            "acc",
            "--system-one-key",
            "token",
            "--system-one-url",
            "http://127.0.0.1:8009",
        ]);
        assert_eq!(
            SystemOneBackend::try_from(&cli.system_one)
                .err()
                .unwrap()
                .to_string(),
            "DREW_SYSTEM_ONE_URL is set, but DREW_SYSTEM_ONE_BACKEND=workers-ai does not use it; unset it"
        );
        assert!(system_one_client(&cli.system_one).is_err());
    }

    #[test]
    fn an_endpoint_keeps_scheme_host_port_and_path_only() {
        let base: Url = "http://user:secret@127.0.0.1:8009/root/?key=secret#frag"
            .parse()
            .unwrap();
        assert_eq!(
            Endpoint::new(&base, DECIDE_PATH).to_string(),
            "http://127.0.0.1:8009/root/v1/systemone"
        );
        let base: Url = "https://system-one.dev".parse().unwrap();
        assert_eq!(
            Endpoint::new(&base, DECIDE_PATH).to_string(),
            "https://system-one.dev/v1/systemone"
        );
    }

    #[test]
    fn a_bad_url_fails_at_parse_time() {
        assert!(
            try_parse(&[
                "--explain-backend",
                "openai-compatible",
                "--explain-url",
                "localhost 11434",
                "--system-one-backend",
                "none",
            ])
            .is_err()
        );
    }

    #[test]
    fn an_empty_key_is_rejected_at_parse_time() {
        assert!(
            try_parse(&[
                "--explain-backend",
                "bedrock",
                "--system-one-backend",
                "system-one",
                "--system-one-key",
                " ",
            ])
            .is_err()
        );
    }
}
