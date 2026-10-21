use std::sync::Arc;

use crate::{
    domain::{AtLeastOne, Question},
    pipeline::{
        Asked, DeclineReason, Scored, Step, StepError, Unclassified, trace::RetrieveRecord,
    },
    retrieval::Retriever,
};

use super::{Candidates, ranked};

/// Runs the [`Retriever`] on the question, in parallel with `Classify`. Declines with
/// [`DeclineReason::NoSources`] when it returns nothing.
pub struct Retrieve {
    retriever: Arc<dyn Retriever>,
}

impl Retrieve {
    /// Wraps the retriever.
    pub fn new(retriever: Arc<dyn Retriever>) -> Self {
        Self { retriever }
    }
}

impl Step for Retrieve {
    const NAME: &'static str = "retrieve";
    type Category = Unclassified;
    type Input = Asked;
    type Output = Candidates;
    type Record = RetrieveRecord;

    async fn run(
        &self,
        question: &Question,
        _input: Asked,
    ) -> Result<Scored<Candidates, RetrieveRecord>, StepError> {
        let hits = self.retriever.retrieve(question.as_str()).await?;
        let record = RetrieveRecord {
            chunks: ranked(&hits),
        };
        let Ok(hits) = AtLeastOne::try_from(hits) else {
            return Err(StepError::decline(DeclineReason::NoSources).with_record(record));
        };
        Ok(Scored::new(Candidates(hits), record))
    }
}
