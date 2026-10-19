# Drew AI

Drew AI answers questions about SAP and ARM internals from **Drew Steinfatt's pre-audited markdown documentation**. Every answer cites its sources; when the sources don't support an answer, it declines instead of guessing.

It ships as a **developer-facing Slack bot** (Socket Mode, in `drew serve`) and an **OpenAI-compatible HTTP API**.

- [Design](docs/design.md): scope, pipeline model, evaluation, and milestones.
- [Architecture diagram](docs/archify/architecture.html): open it in a browser.

## Status

What works today:

- Batch ingestion of a markdown directory into Postgres with pgvector.
- Hybrid retrieval (full text + vectors, reciprocal rank fusion).
- Classify and Rerank on a System One model ([Kev](https://github.com/jaredpalmer/kev)), with calibrated confidence scores.
- The `open` path end to end: retrieve, rerank, explain with an LLM, verify every citation.
- OpenAI-compatible HTTP API, run persistence with full traces (trace on `GET /v1/runs/{id}`, not in chat responses).
- Optional Slack bot when `SLACK_APP_TOKEN` and `SLACK_BOT_TOKEN` are set.
- `drew eval` and local or Bedrock models.

Yes/no questions get a verdict: Kev answers yes or no over the reranked sources, the LLM explains why, and the run declines when the sources contradict the verdict or Kev is not confident enough. A classification below `classify.min_confidence` falls back to `open`.

Not built yet: Answer for `choice` and `ranking` (they take the `open` path); permission enforcement.

## How a question is answered

```
Classify ∥ Retrieve → Rerank → Explain → Verify
```

```mermaid
sequenceDiagram
    actor C as Client (API or Slack)
    participant S as drew serve
    participant P as Pipeline
    participant K as System One (Kev)
    participant E as Embeddings
    participant DB as Postgres + pgvector
    participant L as Explain LLM

    C->>S: POST /v1/chat/completions (bearer key)
    S->>P: last user message as the question
    par Classify
        P->>K: one choice over boolean, choice, ranking, open
        K-->>P: category and its probability
    and Retrieve
        P->>E: embed the question
        E-->>P: query vector
        P->>DB: full text top 50 and vector top 50, fused by RRF
        DB-->>P: 50 chunks, with neighbouring sections as context
    end
    Note over P: Below classify.min_confidence, or choice and ranking: the open path
    loop Rerank: each of the top 25 matches
        P->>K: does this text help answer the question?
        K-->>P: probability of yes
    end
    Note over P: Rerank fuses that with retrieval rank and keeps 5
    opt boolean
        P->>K: Answer: going by these sources, is the answer yes?
        K-->>P: verdict and its confidence, checked against the threshold
    end
    P->>L: Explain: question, 5 sources and any verdict, delimited as data
    L-->>P: claims as JSON citing chunk ids, a contradiction declines
    Note over P: Verify: an uncited claim, or a citation outside<br/>this run's chunks, declines the run
    P-->>S: run report (answered, declined or failed) and trace
    S->>DB: save the run and its trace
    S-->>C: answer with numbered citations and the run id
```

Each step is typed, so a pipeline with steps in the wrong order does not compile. Verify rejects any claim without a citation, or citing a chunk the run did not retrieve. Every run, including declines and failures, is stored with a trace of each step's model, prompt version, score, and raw output.

## Latest evaluation

DaaS golden set (`eval/golden/daas.jsonl`, 58 questions), 2026-10-19, local models: `bge-m3` embeddings, Kev 4B for System One, `qwen3:14b` for Explain. Full runs and traces: [`eval/results/daas-2026-10-19.json`](eval/results/daas-2026-10-19.json).

| Scope | Runs | Classify | Recall@50 | Hit@5 | Declined | Failed | Citations valid |
|---|---|---|---|---|---|---|---|
| Overall | 58 | 82.8% | 78.4% | 36.2% | 3.4% | 0.0% | 100.0% |
| Boolean | 15 | 93.3% | 56.7% | 20.0% | 6.7% | 0.0% | 100.0% |
| Choice | 1 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| Open | 42 | 78.6% | 85.7% | 40.5% | 2.4% | 0.0% | 100.0% |

Retrieval usually finds the section that answers the question, but Rerank keeps it in the top 5 only about a third of the time; that is the step to improve next. Answer precision is left out because the Answer step does not exist yet, so decisive questions have no verdict to grade.

## Running locally

Requires Docker and Rust (the toolchain is pinned in `rust-toolchain.toml`; `rustup` installs it).

```sh
docker compose up -d --wait db     # Postgres + pgvector on localhost:5433
cp .env.example .env               # then fill it in, see Configuration
set -a; source .env; set +a

export DREW_DOCS_DIR=/path/to/drew-markdown   # your audited corpus
cargo run -- ingest
cargo run -- serve                 # API on DREW_BIND; Slack bot if tokens are set
```

Ask a question:

```sh
curl -s localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-local-drew" \
  -H "Content-Type: application/json" \
  -d '{"model": "drew-ai", "messages": [{"role": "user", "content": "What does AGR_USERS store?"}]}'
```

The answer is in `choices[0].message.content`. The top-level `drew` object carries the run id, outcome, and citations. Use `GET /v1/runs/{id}` for the full trace (admins listed in `DREW_ADMIN_USERS` can read any run).

### Slack app setup

1. Create a Slack app with **Socket Mode** enabled.
2. App-level token with `connections:write` → `SLACK_APP_TOKEN` (`xapp-…`).
3. Bot token scopes: `app_mentions:read`, `chat:write`, `im:history`.
4. Subscribe to bot events: `app_mention`, `message.im`.
5. Enable the App Home **Messages** tab for DMs.
6. Set `SLACK_BOT_TOKEN` (`xoxb-…`) and restart `drew serve`.

The bot replies in threads on `@mentions` and in DMs. Runs are stored as user `slack:{slack_user_id}`.

### Running fully locally

`.env.example` is set up for local models; nothing calls AWS.

- **Embeddings and Explain: [Ollama](https://ollama.com).** Embeddings must have 1024 dimensions.

  ```sh
  brew install ollama && brew services start ollama
  ollama pull bge-m3
  ollama pull qwen3:14b
  ```

- **System One (Classify): [Kev](https://github.com/jaredpalmer/kev)** on Apple Silicon via MLX.

Switching embedding models requires `drew ingest --reembed`.

## Commands

| Command | What it does |
|---|---|
| `drew serve` | OpenAI API plus optional Slack bot |
| `drew ingest` | Snapshot ingest from `DREW_DOCS_DIR`; `--reembed` after changing the embedding model |
| `drew eval run …` / `drew eval tune …` | Golden-set metrics and threshold tuning. `eval/golden/daas.jsonl` holds DaaS's own golden questions (`_ops/EVAL.md`, held out of the corpus): `drew eval run --golden eval/golden/daas.jsonl --out eval/results/daas.json --concurrency 2` |

## Configuration

Tunables (retrieval, the Classify fallback, Rerank, step thresholds) are in [`config/drew.json`](config/drew.json), or the file `DREW_CONFIG` names; unknown keys are rejected. `drew eval run` prints the config and models at the start (through `jq` when installed) and records them in the results file.

See `.env.example`. Key variables: `DATABASE_URL`, `DREW_API_KEYS`, `DREW_ADMIN_USERS`, `DREW_DOCS_DIR`, `DREW_DOCS_BASE_URL`, `DREW_DOCS_AUTHORITATIVE` / `DREW_DOCS_EVIDENCE` / `DREW_DOCS_SKIP` (which paths are Drew's answers, which are supporting evidence, and which are left out), model URLs, `SLACK_APP_TOKEN`, `SLACK_BOT_TOKEN`.

## Code layout

| Path | Contents |
|---|---|
| `src/ingest/markdown.rs` | Markdown directory source and heading-section chunking |
| `src/slack/` | Socket Mode bot, Slack API client, mrkdwn rendering |
| `src/pipeline/` | Typed pipeline and steps |
| `src/server/` | OpenAI-compatible HTTP API |
| `migrations/` | Postgres schema |

## Development

With the database running:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo nextest run        # brew install cargo-nextest, or cargo install cargo-nextest --locked
cargo test --doc         # nextest does not run doctests; the compile_fail ones guard the pipeline types
```

Conventions are in [AGENTS.md](AGENTS.md).
