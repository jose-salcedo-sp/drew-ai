//! Which backend serves each model port, read once from the command line and environment.
//!
//! Each port names its backend, with no default: `DREW_EMBED_BACKEND`, `DREW_EXPLAIN_BACKEND` and
//! `DREW_SYSTEM_ONE_BACKEND`. A missing env file is then a startup error, not a silent fall back to Bedrock. The
//! flat command-line fields are parsed into one enum per port whose variants hold exactly the settings that backend
//! uses; a missing setting, or one the backend does not use, is an error naming its variable. [`embedder`],
//! [`explain_model`] and [`system_one_client`] map those enums to the port traits, which is all the rest of the crate
//! sees.
//!
//! Workers AI's chat and embedding models are `openai-compatible`: its `/ai/v1` routes take the same requests with a
//! bearer key, and the account id is part of the URL. Its System One is the separate `workers-ai` backend, because
//! `ai/run` differs from `/v1/systemone` in route, model selector and response envelope.

use std::sync::Arc;

use super::{
    ApiKey, ChatModel, Embedder,
    bedrock_chat::BedrockChat,
    bedrock_embed::BedrockEmbedder,
    openai::{ChatOptions, OpenAiChat, OpenAiEmbedder, ReasoningEffort, SchemaFormat},
    system_one::{HttpSystemOne, SystemOne},
    workers_ai::{CLOUDFLARE_API, WorkersAiSystemOne},
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
    pub embed_url: Option<String>,
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
    pub explain_url: Option<String>,
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
    /// Server root: required for `system-one`, e.g. `http://127.0.0.1:8009` or `https://system-one.dev`. For
    /// `workers-ai`, the Cloudflare API root, `https://api.cloudflare.com/client/v4` when unset.
    #[arg(long, env = "DREW_SYSTEM_ONE_URL")]
    pub system_one_url: Option<String>,
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
    /// Clef on Workers AI: `POST {url}/accounts/{account_id}/ai/run/{model}`.
    WorkersAi,
}

/// A backend setting that is missing, or set for a backend that does not use it.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum InvalidBackend {
    /// The selected backend needs this setting.
    #[error("{var} is required with {backend}")]
    Missing {
        /// The setting's environment variable.
        var: &'static str,
        /// `VAR=value` of the backend selection.
        backend: String,
    },
    /// The selected backend ignores this setting, so it was probably meant for another.
    #[error("{var} is set, but {backend} does not use it; unset it")]
    Unused {
        /// The setting's environment variable.
        var: &'static str,
        /// `VAR=value` of the backend selection.
        backend: String,
    },
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

enum EmbedBackend {
    Bedrock {
        model: String,
    },
    OpenAiCompatible {
        url: String,
        model: String,
        key: Option<ApiKey>,
    },
}

impl TryFrom<&EmbedArgs> for EmbedBackend {
    type Error = InvalidBackend;

    fn try_from(args: &EmbedArgs) -> Result<Self, Self::Error> {
        let selected = Selected::new("DREW_EMBED_BACKEND", &args.embed_backend);
        let model = args.embed_model.clone();
        Ok(match args.embed_backend {
            EmbedBackendKind::Bedrock => {
                selected.reject(args.embed_url.as_ref(), "DREW_EMBED_URL")?;
                selected.reject(args.embed_key.as_ref(), "DREW_EMBED_KEY")?;
                Self::Bedrock { model }
            }
            EmbedBackendKind::OpenaiCompatible => Self::OpenAiCompatible {
                url: selected.require(args.embed_url.as_ref(), "DREW_EMBED_URL")?,
                model,
                key: args.embed_key.clone(),
            },
        })
    }
}

enum ChatBackend {
    Bedrock {
        model: String,
    },
    OpenAiCompatible {
        url: String,
        model: String,
        key: Option<ApiKey>,
        options: ChatOptions,
    },
}

impl TryFrom<&ChatArgs> for ChatBackend {
    type Error = InvalidBackend;

    fn try_from(args: &ChatArgs) -> Result<Self, Self::Error> {
        let selected = Selected::new("DREW_EXPLAIN_BACKEND", &args.explain_backend);
        let model = args.explain_model.clone();
        Ok(match args.explain_backend {
            ChatBackendKind::Bedrock => {
                selected.reject(args.explain_url.as_ref(), "DREW_EXPLAIN_URL")?;
                selected.reject(args.explain_key.as_ref(), "DREW_EXPLAIN_KEY")?;
                selected.reject(
                    args.explain_reasoning_effort.as_ref(),
                    "DREW_EXPLAIN_REASONING_EFFORT",
                )?;
                selected.reject(
                    args.explain_schema_format.as_ref(),
                    "DREW_EXPLAIN_SCHEMA_FORMAT",
                )?;
                Self::Bedrock { model }
            }
            ChatBackendKind::OpenaiCompatible => Self::OpenAiCompatible {
                url: selected.require(args.explain_url.as_ref(), "DREW_EXPLAIN_URL")?,
                model,
                key: args.explain_key.clone(),
                options: ChatOptions {
                    reasoning_effort: args.explain_reasoning_effort,
                    schema_format: args.explain_schema_format.unwrap_or_default(),
                },
            },
        })
    }
}

enum SystemOneBackend {
    None,
    SystemOne {
        url: String,
        model: String,
        key: Option<ApiKey>,
    },
    WorkersAi {
        api_base: String,
        account_id: String,
        model: String,
        key: ApiKey,
    },
}

impl TryFrom<&SystemOneArgs> for SystemOneBackend {
    type Error = InvalidBackend;

    fn try_from(args: &SystemOneArgs) -> Result<Self, Self::Error> {
        let selected = Selected::new("DREW_SYSTEM_ONE_BACKEND", &args.system_one_backend);
        let model = || selected.require(args.system_one_model.as_ref(), "DREW_SYSTEM_ONE_MODEL");
        Ok(match args.system_one_backend {
            SystemOneBackendKind::None => {
                selected.reject(args.system_one_model.as_ref(), "DREW_SYSTEM_ONE_MODEL")?;
                selected.reject(args.system_one_url.as_ref(), "DREW_SYSTEM_ONE_URL")?;
                selected.reject(
                    args.system_one_account_id.as_ref(),
                    "DREW_SYSTEM_ONE_ACCOUNT_ID",
                )?;
                selected.reject(args.system_one_key.as_ref(), "DREW_SYSTEM_ONE_KEY")?;
                Self::None
            }
            SystemOneBackendKind::SystemOne => {
                selected.reject(
                    args.system_one_account_id.as_ref(),
                    "DREW_SYSTEM_ONE_ACCOUNT_ID",
                )?;
                Self::SystemOne {
                    url: selected.require(args.system_one_url.as_ref(), "DREW_SYSTEM_ONE_URL")?,
                    model: model()?,
                    key: args.system_one_key.clone(),
                }
            }
            SystemOneBackendKind::WorkersAi => Self::WorkersAi {
                api_base: args
                    .system_one_url
                    .clone()
                    .unwrap_or_else(|| CLOUDFLARE_API.to_owned()),
                account_id: selected.require(
                    args.system_one_account_id.as_ref(),
                    "DREW_SYSTEM_ONE_ACCOUNT_ID",
                )?,
                model: model()?,
                key: selected.require(args.system_one_key.as_ref(), "DREW_SYSTEM_ONE_KEY")?,
            },
        })
    }
}

/// The embedder `DREW_EMBED_BACKEND` selects.
///
/// # Errors
///
/// As [`InvalidBackend`], when the HTTP client cannot be built, or on Bedrock when the model is not Titan Text
/// Embeddings v2 or Cohere Embed v3.
#[eros::context("setting up the embeddings backend")]
pub async fn embedder(args: &EmbedArgs) -> eros::Result<Arc<dyn Embedder>> {
    Ok(match EmbedBackend::try_from(args)? {
        EmbedBackend::Bedrock { model } => Arc::new(BedrockEmbedder::load(&model).await?),
        EmbedBackend::OpenAiCompatible { url, model, key } => {
            Arc::new(OpenAiEmbedder::new(&url, model)?.with_key(key))
        }
    })
}

/// The `Explain` LLM `DREW_EXPLAIN_BACKEND` selects.
///
/// # Errors
///
/// As [`InvalidBackend`], or when the HTTP client cannot be built.
#[eros::context("setting up the Explain backend")]
pub async fn explain_model(args: &ChatArgs) -> eros::Result<Arc<dyn ChatModel>> {
    Ok(match ChatBackend::try_from(args)? {
        ChatBackend::Bedrock { model } => Arc::new(BedrockChat::load(&model).await),
        ChatBackend::OpenAiCompatible {
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

/// The System One `DREW_SYSTEM_ONE_BACKEND` selects; `None` for `none`.
///
/// # Errors
///
/// As [`InvalidBackend`], when the HTTP client cannot be built, or on Workers AI when the model is not a Workers AI
/// model id.
#[eros::context("setting up the System One backend")]
pub fn system_one_client(args: &SystemOneArgs) -> eros::Result<Option<Arc<dyn SystemOne>>> {
    Ok(match SystemOneBackend::try_from(args)? {
        SystemOneBackend::None => None,
        SystemOneBackend::SystemOne { url, model, key } => {
            Some(Arc::new(HttpSystemOne::new(&url, model, key)?))
        }
        SystemOneBackend::WorkersAi {
            api_base,
            account_id,
            model,
            key,
        } => Some(Arc::new(WorkersAiSystemOne::new(
            &api_base,
            &account_id,
            &model,
            key,
        )?)),
    })
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

    fn parse(args: &[&str]) -> Cli {
        let base = [
            "drew",
            "--embed-backend",
            "openai-compatible",
            "--embed-model",
            "bge-m3",
            "--explain-model",
            "qwen3:14b",
        ];
        Cli::try_parse_from(base.iter().chain(args)).unwrap()
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
        let missing = |error| match error {
            InvalidBackend::Missing { var, .. } => var,
            InvalidBackend::Unused { var, .. } => panic!("{var} reported unused"),
        };
        assert_eq!(
            missing(EmbedBackend::try_from(&cli.embed).err().unwrap()),
            "DREW_EMBED_URL"
        );
        assert_eq!(
            missing(ChatBackend::try_from(&cli.chat).err().unwrap()),
            "DREW_EXPLAIN_URL"
        );
        assert_eq!(
            missing(SystemOneBackend::try_from(&cli.system_one).err().unwrap()),
            "DREW_SYSTEM_ONE_ACCOUNT_ID"
        );
        let error = SystemOneBackend::try_from(&cli.system_one).err().unwrap();
        assert_eq!(
            error.to_string(),
            "DREW_SYSTEM_ONE_ACCOUNT_ID is required with DREW_SYSTEM_ONE_BACKEND=workers-ai"
        );
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
            ChatBackend::try_from(&cli.chat).err().unwrap(),
            InvalidBackend::Unused {
                var: "DREW_EXPLAIN_REASONING_EFFORT",
                backend: "DREW_EXPLAIN_BACKEND=bedrock".into(),
            }
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
    fn workers_ai_defaults_to_the_cloudflare_api_and_needs_a_workers_ai_model() {
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
        ]);
        let Ok(SystemOneBackend::WorkersAi { api_base, .. }) =
            SystemOneBackend::try_from(&cli.system_one)
        else {
            panic!("expected Workers AI");
        };
        assert_eq!(api_base, CLOUDFLARE_API);
        assert!(system_one_client(&cli.system_one).unwrap().is_some());

        let cli = parse(&[
            "--explain-backend",
            "openai-compatible",
            "--system-one-backend",
            "workers-ai",
            "--system-one-model",
            "kev-latest",
            "--system-one-account-id",
            "acc",
            "--system-one-key",
            "token",
        ]);
        let error = format!("{:#?}", system_one_client(&cli.system_one).err().unwrap());
        assert!(error.contains("@cf/cloudflare/clef-flash"), "{error}");
        assert!(!error.contains("token"), "{error}");
    }

    #[test]
    fn an_empty_key_is_rejected_at_parse_time() {
        let base = [
            "drew",
            "--embed-backend",
            "bedrock",
            "--embed-model",
            "m",
            "--explain-backend",
            "bedrock",
            "--explain-model",
            "m",
            "--system-one-backend",
            "system-one",
            "--system-one-key",
            " ",
        ];
        assert!(Cli::try_parse_from(base).is_err());
    }
}
