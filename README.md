# Drew AI

Drew AI answers questions about SAP and ARM internals, using Drew Steinfatt's Confluence pages, Jira comments, and Slack threads as its knowledge source. Every answer cites its sources; when the sources don't support an answer, it declines instead of guessing.

It exposes an OpenAI-compatible API, so any OpenAI client can talk to it.

- [Design](docs/design.md): scope, pipeline model, evaluation, and milestones.
- [Architecture diagram](docs/archify/architecture.html): open it in a browser.

## Status

v1 is a demo for four internal users. What works today:

- Batch ingestion from Confluence, Jira, and Slack into Postgres with pgvector.
- Hybrid retrieval (full text + vectors, reciprocal rank fusion).
- Classify and Rerank on a System One model ([Kev](https://github.com/jaredpalmer/kev)), with calibrated confidence scores.
- The `open` path end to end: retrieve, rerank, explain with an LLM, verify every citation.
- The HTTP API, run persistence with full traces, and `drew eval`.
- Models on Bedrock, or fully local: Ollama for embeddings and Explain, Kev for System One.

Not built yet: the Answer step (milestone 6), so `boolean`, `choice`, and `ranking` questions are classified but answered through the `open` path; confidence thresholds; and permission enforcement.

## How a question is answered

```
Classify ∥ Retrieve → Rerank → Explain → Verify
```

Each step is typed, so a pipeline with steps in the wrong order does not compile. Verify rejects any claim without a citation, or citing a chunk the run did not retrieve. Every run, including declines and failures, is stored with a trace of each step's model, prompt version, score, and raw output.

## Running locally

Requires Docker and Rust (the toolchain is pinned in `rust-toolchain.toml`; `rustup` installs it).

```sh
docker compose up -d --wait db     # Postgres + pgvector on localhost:5433
cp .env.example .env               # then fill it in, see Configuration
set -a; source .env; set +a

cargo run -- ingest                # pull changed content from the configured sources
cargo run -- serve                 # API on DREW_BIND
```

Ask a question:

```sh
curl -s localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-local-drew" \
  -H "Content-Type: application/json" \
  -d '{"model": "drew-ai", "messages": [{"role": "user", "content": "What does AGR_USERS store?"}]}'
```

For an API client such as Yaak or Postman, import `clients/drew-ai.postman_collection.json`.

The answer is in `choices[0].message.content`. The top-level `drew` object carries the run id, outcome, citations, and trace; `GET /v1/runs/{id}` returns a stored run.

### Running fully locally

`.env.example` is set up for local models; nothing calls AWS.

- **Embeddings and Explain: [Ollama](https://ollama.com).** Embeddings must have 1024 dimensions, the size of `chunks.embedding`.

  ```sh
  brew install ollama && brew services start ollama
  ollama pull bge-m3
  ollama pull qwen3:14b
  ```

  Any OpenAI-compatible server works instead, such as `llama-server` from llama.cpp in a Podman container with GPU passthrough: point `DREW_EMBED_URL` and `DREW_EXPLAIN_URL` at it.

- **System One (Classify): [Kev](https://github.com/jaredpalmer/kev),** natively rather than in a container, because it runs on Apple Silicon through MLX. Needs Python 3.12 or 3.13 and `uv`; Kev-4B wants a 32 GB Mac, and Kev-0.8B is the lighter option.

  ```sh
  git clone https://github.com/jaredpalmer/kev.git && cd kev
  uv sync --extra serve
  uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b@v1.0 --port 8009
  ```

  Without `DREW_SYSTEM_ONE_URL`, every question is classified `open` and Rerank keeps the top 5 matches by retrieval score.

Switching embedding models (for example from Bedrock to `bge-m3`) requires `drew ingest --reembed`, because vectors from different models are never mixed.

## Commands

| Command | What it does |
|---|---|
| `drew serve` | OpenAI-compatible API: `/v1/models`, `/v1/chat/completions` (JSON or `stream: true`), `/v1/runs/{id}` |
| `drew ingest` | Fetches content changed since the last successful run per source, embeds new chunk versions, supersedes old ones. `--reembed` after changing the embedding model |
| `drew eval run --golden <jsonl> --out <json>` | Answers a golden set and reports per-step metrics; `--baseline <json>` fails on regressions |
| `drew eval tune --results <json> --candidates <json>` | Re-applies candidate thresholds to recorded scores, without calling any model |

`drew <command> --help` lists every flag and its environment variable.

## Configuration

All configuration is environment variables; `.env.example` lists them all.

- **Database:** `DATABASE_URL`.
- **API:** `DREW_BIND`, and `DREW_API_KEYS` as comma-separated `key:user_id` pairs.
- **Models:** `DREW_EMBED_MODEL` and `DREW_EXPLAIN_MODEL`. With `DREW_EMBED_URL` / `DREW_EXPLAIN_URL` set, they are served by that OpenAI-compatible server; otherwise by Bedrock, with credentials and region from the default AWS chain.
- **System One:** `DREW_SYSTEM_ONE_URL`, plus `DREW_SYSTEM_ONE_MODEL` (default `kev-latest`) and `DREW_SYSTEM_ONE_KEY` when the server requires a key.
- **Sources:** a source is ingested only when all of its variables are set.
  - Confluence: base URL, space key, email, API token, and Drew's account id.
  - Jira: base URL, email, API token, a scope JQL without `ORDER BY`, and Drew's account id.
  - Slack: a bot token with `channels:read`, `channels:history`, and `users:read`, channel ids, the workspace URL, and Drew's user id. Invite the bot to each channel; private channels and DMs are rejected.

## Docker

```sh
docker build -t drew-ai .

# Inside the container, bind on all interfaces and reach the host's Postgres via host.docker.internal.
db=postgres://drew:drew@host.docker.internal:5433/drew
docker run --env-file .env -e DREW_BIND=0.0.0.0:8080 -e DATABASE_URL=$db -p 8080:8080 drew-ai   # drew serve
docker run --env-file .env -e DATABASE_URL=$db drew-ai ingest
```

## Code layout

| Path | Contents |
|---|---|
| `src/domain.rs` | Shared types: chunks and versions, labels, provenance, `Cited<T>` |
| `src/pipeline/` | The typed pipeline (`Run`, `Step`, `Pipeline`, `Seq`), its steps, the trace, and the `Answerer` |
| `src/ingest/` | The ingest runner and the Confluence, Jira, and Slack connectors |
| `src/retrieval/` | Hybrid retrieval over Postgres |
| `src/llm/` | Embedding and chat clients (Bedrock, OpenAI-compatible) and the System One client |
| `src/store/` | Chunk store and run store |
| `src/server/` | The HTTP API and answer rendering |
| `src/eval/` | Golden-set metrics and threshold tuning |
| `migrations/` | Postgres schema |
| `eval/golden/` | Golden sets; `example.jsonl` is synthetic |

## Development

With the database running:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Tests run offline: model clients, retrieval, and sources have fakes, and each database test gets its own database. Conventions are in [AGENTS.md](AGENTS.md).
