//! The real `Answerer`: classify and retrieve in parallel, join on the category, then drive its pipeline.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use sqlx::PgPool;

use crate::{
    domain::{CategoryKind, Chunk, Provenance, Question},
    llm::{
        ChatModel,
        bedrock_chat::{BedrockChat, ChatArgs},
        bedrock_embed::{BedrockEmbedder, EmbedArgs},
    },
    retrieval::{
        Retrieved, Retriever,
        hybrid::{HybridConfig, HybridRetriever},
    },
};

use super::{
    Boolean, Category, Choice, Halt, Halted, Open, Pipeline, Ranking, Run, Seq, Unclassified,
    report::{Answer, Answerer, Citation, Outcome, RunReport},
    steps::{
        Candidates, Classify, Explain, OpenEnded, Rerank, Reranked, Retrieve, Verified, Verify,
    },
};

const EXCERPT_CHARS: usize = 200;

/// The one place where runtime data becomes a type. See `docs/design.md` section 5.4.
pub enum Classified {
    Boolean(Run<Boolean, Reranked>),
    Choice(Run<Choice, Reranked>),
    Ranking(Run<Ranking, Reranked>),
    Open(Run<Open, Reranked>),
}

#[derive(Clone, Debug, clap::Args)]
pub struct PipelineArgs {
    #[command(flatten)]
    pub embed: EmbedArgs,
    #[command(flatten)]
    pub chat: ChatArgs,
}

pub struct PipelineAnswerer {
    classify: Classify,
    retrieve: Retrieve,
    rerank_boolean: Rerank<Boolean>,
    rerank_choice: Rerank<Choice>,
    rerank_ranking: Rerank<Ranking>,
    rerank_open: Rerank<Open>,
    open: OpenEnded,
}

impl PipelineAnswerer {
    pub fn new(retriever: Arc<dyn Retriever>, explain: Arc<dyn ChatModel>) -> Self {
        Self {
            classify: Classify,
            retrieve: Retrieve::new(retriever),
            rerank_boolean: Rerank::default(),
            rerank_choice: Rerank::default(),
            rerank_ranking: Rerank::default(),
            rerank_open: Rerank::default(),
            open: Seq(Explain::new(explain), Verify::new()),
        }
    }

    /// Hybrid retrieval with Bedrock embeddings, and Bedrock for `Explain`.
    pub async fn from_args(pool: PgPool, args: &PipelineArgs) -> anyhow::Result<Self> {
        let embedder = Arc::new(BedrockEmbedder::from_args(&args.embed).await?);
        let retriever = HybridRetriever::new(pool, embedder, HybridConfig::default());
        let explain = BedrockChat::from_args(&args.chat).await;
        Ok(Self::new(Arc::new(retriever), Arc::new(explain)))
    }

    async fn classify(&self, question: Question) -> Result<Classified, Halted> {
        let run = Run::new(question)
            .both("classify_retrieve", &self.classify, &self.retrieve)
            .await?;
        let category = run.state().0;
        let run: Run<Unclassified, Candidates> = run.map(|(_, candidates)| candidates);
        Ok(match category {
            CategoryKind::Boolean => {
                Classified::Boolean(run.classified().then(&self.rerank_boolean).await?)
            }
            CategoryKind::Choice => {
                Classified::Choice(run.classified().then(&self.rerank_choice).await?)
            }
            CategoryKind::Ranking => {
                Classified::Ranking(run.classified().then(&self.rerank_ranking).await?)
            }
            CategoryKind::Open => Classified::Open(run.classified().then(&self.rerank_open).await?),
        })
    }
}

#[async_trait]
impl Answerer for PipelineAnswerer {
    async fn answer(&self, question: Question) -> RunReport {
        let classified = match self.classify(question).await {
            Ok(classified) => classified,
            Err(halted) => return halted_report(halted, &[]),
        };
        match classified {
            Classified::Boolean(run) => halted_report(answer_not_implemented(run), &[]),
            Classified::Choice(run) => halted_report(answer_not_implemented(run), &[]),
            Classified::Ranking(run) => halted_report(answer_not_implemented(run), &[]),
            Classified::Open(run) => {
                // `Halted` carries no state, so a later decline lists the sources kept here.
                let sources = run.state().0.clone();
                match self.open.drive(run).await {
                    Ok(run) => answered_report(run),
                    Err(halted) => halted_report(halted, &sources),
                }
            }
        }
    }
}

fn answer_not_implemented<TCategory: Category>(run: Run<TCategory, Reranked>) -> Halted {
    let (id, question, _, trace) = run.into_parts();
    Halted {
        id,
        question,
        trace,
        category: TCategory::KIND,
        at: "answer",
        halt: Halt::Failed {
            error: "Answer is not implemented until milestone 6".into(),
        },
    }
}

fn answered_report(run: Run<Open, Verified>) -> RunReport {
    let (id, question, Verified { answer, sources }, trace) = run.into_parts();
    let why = answer.into_claims();
    let mut cited = HashSet::new();
    let citations = why
        .iter()
        .flat_map(|claim| claim.sources.as_slice())
        .filter_map(|source| match source {
            Provenance::Chunk(id) => Some(id),
            Provenance::UserPrompt(_) => None,
        })
        .filter(|id| cited.insert(*id))
        .filter_map(|id| sources.0.iter().find(|hit| &hit.chunk.id == id))
        .map(|hit| citation(&hit.chunk))
        .collect();
    RunReport {
        id,
        question,
        category: Open::KIND,
        outcome: Outcome::Answered(Answer {
            verdict: None,
            why,
            citations,
        }),
        trace,
    }
}

fn halted_report(halted: Halted, sources: &[Retrieved]) -> RunReport {
    let at = halted.at.to_owned();
    let outcome = match halted.halt {
        Halt::Declined {
            reason,
            score,
            threshold,
        } => Outcome::Declined {
            at,
            reason,
            score,
            threshold,
            sources: sources.iter().map(|hit| citation(&hit.chunk)).collect(),
        },
        Halt::Failed { error } => Outcome::Failed { at, error },
    };
    RunReport {
        id: halted.id,
        question: halted.question,
        category: halted.category,
        outcome,
        trace: halted.trace,
    }
}

fn citation(chunk: &Chunk) -> Citation {
    Citation {
        chunk: chunk.reference(),
        title: chunk.title.clone(),
        url: chunk.url.clone(),
        excerpt: excerpt(&chunk.text),
    }
}

fn excerpt(text: &str) -> String {
    let text = text.trim();
    match text.char_indices().nth(EXCERPT_CHARS) {
        Some((end, _)) => format!("{}…", text[..end].trim_end()),
        None => text.to_owned(),
    }
}

#[cfg(test)]
mod tests;
