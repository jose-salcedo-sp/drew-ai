use std::sync::Arc;

use crate::{
    domain::Question,
    pipeline::{
        Asked, DeclineReason, Scored, Step, StepError, Unclassified, trace::RetrieveRecord,
    },
    retrieval::Retriever,
};

use super::{Candidates, ranked};

pub struct Retrieve {
    retriever: Arc<dyn Retriever>,
}

impl Retrieve {
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
        if hits.is_empty() {
            return Err(StepError::decline(DeclineReason::NoSources).with_record(record));
        }
        Ok(Scored::new(Candidates(hits), record))
    }
}
