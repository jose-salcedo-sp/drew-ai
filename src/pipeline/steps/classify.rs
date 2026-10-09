use crate::{
    domain::{CategoryKind, Question},
    pipeline::{Asked, Scored, Step, StepError, Unclassified, trace::ClassifyRecord},
};

/// Milestone 3 stub: every question is `open`. The System One classifier replaces it in milestone 4.
pub struct Classify;

impl Step for Classify {
    const NAME: &'static str = "classify";
    type Category = Unclassified;
    type Input = Asked;
    type Output = CategoryKind;
    type Record = ClassifyRecord;

    async fn run(
        &self,
        _question: &Question,
        _input: Asked,
    ) -> Result<Scored<CategoryKind, ClassifyRecord>, StepError> {
        let category = CategoryKind::Open;
        let record = ClassifyRecord {
            raw: None,
            category: Some(category),
        };
        Ok(Scored::new(category, record))
    }
}
