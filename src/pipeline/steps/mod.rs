//! The steps every category's pipeline is built from. See `docs/design.md` sections 5.3 to 5.5.

mod classify;
mod explain;
mod rerank;
mod retrieve;
mod verify;

pub use self::{
    classify::{CLASSIFY_PROMPT_VERSION, Classify},
    explain::{Claim, Explain, Explained, Explanation, InvalidExplanation, PROMPT_VERSION},
    rerank::Rerank,
    retrieve::Retrieve,
    verify::{Verified, VerifiedAnswer, Verify},
};

use crate::retrieval::Retrieved;

use super::{Open, Seq, trace::RankedChunk};

pub type OpenEnded = Seq<Explain<Open>, Verify<Open>>;

/// Output of `Retrieve`: hybrid matches, best first. Never empty.
#[derive(Clone, Debug)]
pub struct Candidates(pub Vec<Retrieved>);

/// Output of `Rerank`: the sources `Answer` and `Explain` see, best first.
#[derive(Clone, Debug)]
pub struct Reranked(pub Vec<Retrieved>);

fn ranked(hits: &[Retrieved]) -> Vec<RankedChunk> {
    hits.iter()
        .map(|hit| RankedChunk {
            chunk: hit.chunk.reference(),
            score: hit.score,
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod fixtures {
    use crate::domain::{
        Chunk, ChunkId, ChunkVersion, DocumentId, Labels, Sensitivity, SourceKind,
    };

    pub fn chunk(locator: &str, title: &str, text: &str) -> Chunk {
        let document_id = DocumentId::new(SourceKind::Confluence, title);
        Chunk {
            id: ChunkId::new(&document_id, locator),
            version: ChunkVersion(format!("v-{locator}")),
            document_id,
            source: SourceKind::Confluence,
            title: title.into(),
            heading: Some(locator.into()),
            url: format!("https://confluence.example.test/{title}#{locator}"),
            text: text.into(),
            authoritative: false,
            labels: Labels {
                acl: vec![],
                customer: None,
                sensitivity: Sensitivity::Internal,
            },
        }
    }
}
