//! Types shared by every module: questions, chunks, labels, provenance, and probabilities.

use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::llm::ModelRef;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Question(String);

impl Question {
    pub fn new(text: impl Into<String>) -> Self {
        Self(text.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The text a `Span` points at, if the span is in bounds and on char boundaries.
    pub fn slice(&self, span: Span) -> Option<&str> {
        self.0.get(span.start..span.end)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunId(Uuid);

impl RunId {
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Uuid> for RunId {
    fn from(id: Uuid) -> Self {
        Self(id)
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct UserId(pub String);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CategoryKind {
    Boolean,
    Choice,
    Ranking,
    Open,
}

impl CategoryKind {
    pub const ALL: [Self; 4] = [Self::Boolean, Self::Choice, Self::Ranking, Self::Open];

    /// The serde name, also stored in `pipeline_runs.category`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Boolean => "boolean",
            Self::Choice => "choice",
            Self::Ranking => "ranking",
            Self::Open => "open",
        }
    }
}

impl TryFrom<&str> for CategoryKind {
    type Error = UnknownVariant;

    fn try_from(name: &str) -> Result<Self, UnknownVariant> {
        match name {
            "boolean" => Ok(Self::Boolean),
            "choice" => Ok(Self::Choice),
            "ranking" => Ok(Self::Ranking),
            "open" => Ok(Self::Open),
            other => Err(UnknownVariant::new("category", other)),
        }
    }
}

/// A string that names no variant of a text-backed domain enum.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("unknown {kind} {name:?}")]
pub struct UnknownVariant {
    pub kind: &'static str,
    pub name: String,
}

impl UnknownVariant {
    fn new(kind: &'static str, name: &str) -> Self {
        Self {
            kind,
            name: name.to_owned(),
        }
    }
}

/// A probability or a confidence: finite and in `0..=1`. Step scores and thresholds are compared as these, so a
/// model's out-of-range or `NaN` output cannot pass a threshold. Serialized as a bare number.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct Probability(f32);

#[derive(Debug, PartialEq, thiserror::Error)]
#[error("expected a probability between 0 and 1, got {0}")]
pub struct NotAProbability(pub f32);

impl Probability {
    pub fn new(value: f32) -> Result<Self, NotAProbability> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(NotAProbability(value))
        }
    }

    pub fn get(self) -> f32 {
        self.0
    }

    /// `1 - p`: the probability of the other answer to a yes/no question.
    pub fn complement(self) -> Self {
        Self(1.0 - self.0)
    }
}

impl TryFrom<f32> for Probability {
    type Error = NotAProbability;

    fn try_from(value: f32) -> Result<Self, NotAProbability> {
        Self::new(value)
    }
}

impl From<Probability> for f32 {
    fn from(probability: Probability) -> f32 {
        probability.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Markdown,
}

impl SourceKind {
    pub const ALL: [Self; 1] = [Self::Markdown];

    /// The serde name, also stored in `documents.source` and used as the `DocumentId` prefix.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Markdown => "markdown",
        }
    }
}

impl TryFrom<&str> for SourceKind {
    type Error = UnknownVariant;

    fn try_from(name: &str) -> Result<Self, UnknownVariant> {
        match name {
            "markdown" => Ok(Self::Markdown),
            other => Err(UnknownVariant::new("source", other)),
        }
    }
}

/// A source unit: one markdown file in the docs corpus. Format: `{source}:{external_id}`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DocumentId(pub String);

impl DocumentId {
    pub fn new(source: SourceKind, external_id: &str) -> Self {
        Self(format!("{}:{external_id}", source.as_str()))
    }
}

/// Stable identity of a chunk across re-ingestion. Format: `{document_id}#{locator}`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ChunkId(pub String);

impl ChunkId {
    pub fn new(document: &DocumentId, locator: &str) -> Self {
        Self(format!("{}#{locator}", document.0))
    }
}

/// Content address of one immutable chunk version: hex SHA-256 over its content, labels, and embedding model.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ChunkVersion(pub String);

impl ChunkVersion {
    pub fn compute(document: &Document, draft: &ChunkDraft, embedding_model: &ModelRef) -> Self {
        #[derive(Serialize)]
        struct Content<'a> {
            title: &'a str,
            heading: &'a Option<String>,
            url: &'a str,
            text: &'a str,
            authoritative: bool,
            labels: &'a Labels,
            embedding_model: &'a ModelRef,
        }
        let content = Content {
            title: &document.title,
            heading: &draft.heading,
            url: &draft.url,
            text: &draft.text,
            authoritative: draft.authoritative,
            labels: &document.labels,
            embedding_model,
        };
        let bytes = serde_json::to_vec(&content)
            .expect("chunk content is plain data, which always serializes");
        Self(hex::encode(Sha256::digest(bytes)))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkRef {
    pub id: ChunkId,
    pub version: ChunkVersion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sensitivity {
    Public,
    Internal,
    Confidential,
}

impl Sensitivity {
    pub const ALL: [Self; 3] = [Self::Public, Self::Internal, Self::Confidential];

    /// The serde name, also stored in `chunks.sensitivity`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Internal => "internal",
            Self::Confidential => "confidential",
        }
    }
}

impl TryFrom<&str> for Sensitivity {
    type Error = UnknownVariant;

    fn try_from(name: &str) -> Result<Self, UnknownVariant> {
        match name {
            "public" => Ok(Self::Public),
            "internal" => Ok(Self::Internal),
            "confidential" => Ok(Self::Confidential),
            other => Err(UnknownVariant::new("sensitivity", other)),
        }
    }
}

/// Recorded on every chunk; enforced only in the security phase.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Labels {
    /// Principals inherited from the source ACL (groups, users, or channel membership).
    pub acl: Vec<String>,
    pub customer: Option<String>,
    pub sensitivity: Sensitivity,
}

/// What a source connector produces for one source unit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub source: SourceKind,
    /// Path relative to the docs root, with `/` separators.
    pub external_id: String,
    pub title: String,
    pub url: String,
    pub updated_at: DateTime<Utc>,
    pub labels: Labels,
    pub chunks: Vec<ChunkDraft>,
}

impl Document {
    pub fn id(&self) -> DocumentId {
        DocumentId::new(self.source, &self.external_id)
    }

    pub fn chunk_id(&self, draft: &ChunkDraft) -> ChunkId {
        ChunkId::new(&self.id(), &draft.locator)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChunkDraft {
    /// Unique within the document and stable across re-ingestion (heading anchor, comment id, message ts).
    pub locator: String,
    pub heading: Option<String>,
    /// Deep link to the section, comment, or message.
    pub url: String,
    pub text: String,
    /// Written by Drew.
    pub authoritative: bool,
}

/// A current chunk version as read back from the store.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub id: ChunkId,
    pub version: ChunkVersion,
    pub document_id: DocumentId,
    pub source: SourceKind,
    /// The document's title.
    pub title: String,
    pub heading: Option<String>,
    pub url: String,
    pub text: String,
    pub authoritative: bool,
    pub labels: Labels,
}

impl Chunk {
    pub fn reference(&self) -> ChunkRef {
        ChunkRef {
            id: self.id.clone(),
            version: self.version.clone(),
        }
    }
}

/// Byte range into the question text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    UserPrompt(Span),
    Chunk(ChunkId),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cited<T> {
    pub value: T,
    pub sources: AtLeastOne<Provenance>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<T>", into = "Vec<T>")]
pub struct AtLeastOne<T: Clone>(Vec<T>);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<T>", into = "Vec<T>")]
pub struct AtLeastTwo<T: Clone>(Vec<T>);

#[derive(Debug, thiserror::Error)]
#[error("expected at least {expected} items, got {actual}")]
pub struct TooFew {
    pub expected: usize,
    pub actual: usize,
}

macro_rules! at_least {
    ($name:ident, $n:literal) => {
        impl<T: Clone> $name<T> {
            pub fn as_slice(&self) -> &[T] {
                &self.0
            }

            pub fn into_vec(self) -> Vec<T> {
                self.0
            }
        }

        impl<T: Clone> TryFrom<Vec<T>> for $name<T> {
            type Error = TooFew;

            fn try_from(items: Vec<T>) -> Result<Self, TooFew> {
                if items.len() < $n {
                    return Err(TooFew {
                        expected: $n,
                        actual: items.len(),
                    });
                }
                Ok(Self(items))
            }
        }

        impl<T: Clone> From<$name<T>> for Vec<T> {
            fn from(items: $name<T>) -> Vec<T> {
                items.0
            }
        }
    };
}

at_least!(AtLeastOne, 1);
at_least!(AtLeastTwo, 2);

impl<T: Clone> AtLeastOne<T> {
    pub fn one(item: T) -> Self {
        Self(vec![item])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probability_rejects_out_of_range_and_non_finite_values() {
        for bad in [
            -0.3,
            1.7,
            -f32::EPSILON,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            assert!(Probability::new(bad).is_err(), "{bad}");
        }
        for good in [0.0, 0.5, 1.0] {
            assert_eq!(Probability::new(good).unwrap().get(), good);
        }
        assert_eq!(Probability::new(0.25).unwrap().complement().get(), 0.75);
    }

    #[test]
    fn probability_is_a_bare_number_and_deserialization_checks_it() {
        let p = Probability::new(0.875).unwrap();
        assert_eq!(serde_json::to_string(&p).unwrap(), "0.875");
        assert_eq!(serde_json::from_str::<Probability>("0.875").unwrap(), p);
        for bad in ["1.5", "-0.1", "1e39"] {
            assert!(serde_json::from_str::<Probability>(bad).is_err(), "{bad}");
        }
    }

    fn assert_text_matches_serde<T>(variants: &[T], as_str: fn(&T) -> &'static str)
    where
        T: Copy
            + fmt::Debug
            + PartialEq
            + Serialize
            + for<'a> TryFrom<&'a str, Error = UnknownVariant>,
    {
        for &variant in variants {
            let name = as_str(&variant);
            assert_eq!(serde_json::to_value(variant).unwrap(), name, "{variant:?}");
            assert_eq!(T::try_from(name).unwrap(), variant);
        }
        assert!(T::try_from("Unknown").is_err());
    }

    #[test]
    fn category_text_matches_serde_and_round_trips() {
        assert_text_matches_serde(&CategoryKind::ALL, CategoryKind::as_str);
    }

    #[test]
    fn sensitivity_text_matches_serde_and_round_trips() {
        assert_text_matches_serde(&Sensitivity::ALL, Sensitivity::as_str);
    }

    #[test]
    fn source_text_matches_serde_and_round_trips() {
        assert_text_matches_serde(&SourceKind::ALL, SourceKind::as_str);
    }

    #[test]
    fn at_least_two_rejects_one() {
        assert!(AtLeastTwo::try_from(vec![1]).is_err());
        assert!(AtLeastTwo::try_from(vec![1, 2]).is_ok());
    }

    #[test]
    fn at_least_one_deserialization_rejects_empty() {
        assert!(serde_json::from_str::<AtLeastOne<u8>>("[]").is_err());
        assert!(serde_json::from_str::<AtLeastOne<u8>>("[1]").is_ok());
    }

    #[test]
    fn chunk_version_changes_with_embedding_model() {
        let document = Document {
            source: SourceKind::Markdown,
            external_id: "roles.md".into(),
            title: "Roles".into(),
            url: "https://example.test/roles.md".into(),
            updated_at: Utc::now(),
            labels: Labels {
                acl: vec![],
                customer: None,
                sensitivity: Sensitivity::Internal,
            },
            chunks: vec![],
        };
        let draft = ChunkDraft {
            locator: "intro".into(),
            heading: None,
            url: document.url.clone(),
            text: "AGR_USERS maps users to roles.".into(),
            authoritative: true,
        };
        let titan = ModelRef::new("amazon.titan-embed-text-v2", "1");
        let cohere = ModelRef::new("cohere.embed-english-v3", "1");
        assert_ne!(
            ChunkVersion::compute(&document, &draft, &titan),
            ChunkVersion::compute(&document, &draft, &cohere),
        );
        assert_eq!(document.chunk_id(&draft).0, "markdown:roles.md#intro");
    }
}
