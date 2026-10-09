//! The real `Answerer`: classify and retrieve in parallel, join on the category, then drive its pipeline.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use sqlx::PgPool;

use crate::{
    domain::{CategoryKind, Chunk, Provenance, Question},
    llm::{
        self, ChatModel,
        bedrock_chat::ChatArgs,
        bedrock_embed::EmbedArgs,
        system_one::{HttpSystemOne, SystemOne, SystemOneArgs},
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
    #[command(flatten)]
    pub system_one: SystemOneArgs,
}

pub struct PipelineAnswerer {
    classify: Classify,
    retrieve: Retrieve,
    rerank_open: Rerank<Open>,
    open: OpenEnded,
}

impl PipelineAnswerer {
    /// Classifies every question as `open`; see `with_classifier`.
    pub fn new(retriever: Arc<dyn Retriever>, explain: Arc<dyn ChatModel>) -> Self {
        Self {
            classify: Classify::new(None),
            retrieve: Retrieve::new(retriever),
            rerank_open: Rerank::default(),
            open: Seq(Explain::new(explain), Verify::new()),
        }
    }

    pub fn with_classifier(self, system_one: Arc<dyn SystemOne>) -> Self {
        Self {
            classify: Classify::new(Some(system_one)),
            ..self
        }
    }

    /// Hybrid retrieval and `Explain` on Bedrock or OpenAI-compatible servers; `Classify` on System One when configured.
    pub async fn from_args(pool: PgPool, args: &PipelineArgs) -> anyhow::Result<Self> {
        let embedder = llm::embedder(&args.embed).await?;
        let retriever = HybridRetriever::new(pool, embedder, HybridConfig::default());
        let explain = llm::explain_model(&args.chat).await;
        let answerer = Self::new(Arc::new(retriever), explain);
        Ok(match HttpSystemOne::from_args(&args.system_one) {
            Some(system_one) => answerer.with_classifier(Arc::new(system_one)),
            None => answerer,
        })
    }

    /// Until `Answer` lands (milestone 6), `boolean`, `choice`, and `ranking` questions take the open path; the
    /// returned category is still the one `Classify` chose.
    async fn classify(&self, question: Question) -> Result<(CategoryKind, Classified), Halted> {
        let run = Run::new(question)
            .both("classify_retrieve", &self.classify, &self.retrieve)
            .await?;
        let category = run.state().0;
        let run: Run<Unclassified, Candidates> = run.map(|(_, candidates)| candidates);
        let run = run.classified::<Open>().then(&self.rerank_open).await?;
        Ok((category, Classified::Open(run)))
    }
}

#[async_trait]
impl Answerer for PipelineAnswerer {
    async fn answer(&self, question: Question) -> RunReport {
        let (category, classified) = match self.classify(question).await {
            Ok(classified) => classified,
            Err(halted) => return halted_report(halted, &[]),
        };
        let mut report = match classified {
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
        };
        report.category = Some(category);
        report
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
