use std::{collections::BTreeMap, sync::Arc};

use crate::{
    domain::{CategoryKind, Question, Score},
    llm::system_one::{Decision, DecisionAnswer, DecisionRequest, Options, SystemOne},
    pipeline::{
        Asked, DeclineReason, Scored, Step, StepError, Unclassified, trace::ClassifyRecord,
    },
};

pub const CLASSIFY_PROMPT_VERSION: &str = "classify-v1";

const QUESTION_ID: &str = "category";

const OPTIONS: [(CategoryKind, &str, &str); 4] = [
    (CategoryKind::Boolean, "boolean", "A yes or no answer"),
    (
        CategoryKind::Choice,
        "choice",
        "Picking one of two or more options named in the question",
    ),
    (
        CategoryKind::Ranking,
        "ranking",
        "Ordering several items named in the question by a stated criterion",
    ),
    (
        CategoryKind::Open,
        "open",
        "An explanation, a how-to, or any other free-form answer",
    ),
];

/// Asks System One which category the question is. Without a System One every question is `open`.
pub struct Classify {
    system_one: Option<Arc<dyn SystemOne>>,
}

impl Classify {
    pub fn new(system_one: Option<Arc<dyn SystemOne>>) -> Self {
        Self { system_one }
    }

    fn request(question: &Question) -> DecisionRequest {
        let options = OPTIONS
            .iter()
            .map(|(_, name, description)| ((*name).to_owned(), Some((*description).to_owned())))
            .collect();
        DecisionRequest {
            state: question.as_str().to_owned(),
            model: String::new(),
            questions: BTreeMap::from([(
                QUESTION_ID.to_owned(),
                Decision::Choice {
                    instructions: "What kind of answer does this question ask for?".to_owned(),
                    criteria: Options(options),
                },
            )]),
        }
    }
}

/// The chosen category and the calibrated probability System One gave it.
fn parse(answer: Option<&DecisionAnswer>) -> Result<(CategoryKind, Score), DeclineReason> {
    let invalid = |detail: String| DeclineReason::InvalidModelOutput(detail);
    let Some(DecisionAnswer::Choice {
        choice,
        probabilities,
        ..
    }) = answer
    else {
        return Err(invalid(format!(
            "expected a choice answer for {QUESTION_ID:?}"
        )));
    };
    let category = OPTIONS
        .iter()
        .find(|(_, name, _)| name == choice)
        .map(|(category, _, _)| *category)
        .ok_or_else(|| invalid(format!("unknown category {choice:?}")))?;
    let probability = probabilities
        .get(choice)
        .copied()
        .ok_or_else(|| invalid(format!("no probability for {choice:?}")))?;
    Ok((category, Score(probability)))
}

impl Step for Classify {
    const NAME: &'static str = "classify";
    type Category = Unclassified;
    type Input = Asked;
    type Output = CategoryKind;
    type Record = ClassifyRecord;

    async fn run(
        &self,
        question: &Question,
        _input: Asked,
    ) -> Result<Scored<CategoryKind, ClassifyRecord>, StepError> {
        let Some(system_one) = &self.system_one else {
            let category = CategoryKind::Open;
            let record = ClassifyRecord {
                raw: None,
                category: Some(category),
            };
            return Ok(Scored::new(category, record));
        };

        let model = system_one.model();
        let response = system_one
            .decide(Self::request(question))
            .await
            .map_err(|error| {
                StepError::from(error).with_model(model.clone(), CLASSIFY_PROMPT_VERSION)
            })?;
        let answer = response.answers.get(QUESTION_ID);
        let raw = answer.and_then(|answer| serde_json::to_string(answer).ok());
        let (category, score) = parse(answer).map_err(|reason| {
            StepError::decline(reason)
                .with_record(ClassifyRecord {
                    raw: raw.clone(),
                    category: None,
                })
                .with_model(model.clone(), CLASSIFY_PROMPT_VERSION)
        })?;
        let record = ClassifyRecord {
            raw,
            category: Some(category),
        };
        Ok(Scored::new(category, record)
            .with_score(score)
            .with_model(model, CLASSIFY_PROMPT_VERSION))
    }
}

#[cfg(test)]
mod tests {
    use crate::llm::system_one::{DecisionResponse, ScriptedSystemOne};

    use super::*;

    fn choice(choice: &str, probability: f32) -> DecisionResponse {
        DecisionResponse {
            answers: BTreeMap::from([(
                QUESTION_ID.to_owned(),
                DecisionAnswer::Choice {
                    choice: choice.into(),
                    confidence: 0.5,
                    probabilities: BTreeMap::from([(choice.into(), probability)]),
                },
            )]),
        }
    }

    async fn classify(
        system_one: &Arc<ScriptedSystemOne>,
    ) -> Result<Scored<CategoryKind, ClassifyRecord>, StepError> {
        Classify::new(Some(system_one.clone()))
            .run(
                &Question::new("Does AGR_USERS hold role assignments?"),
                Asked,
            )
            .await
    }

    #[tokio::test]
    async fn the_choice_and_its_probability_become_the_category_and_score() {
        let system_one = Arc::new(ScriptedSystemOne::new([Ok(choice("boolean", 0.87))]));

        let scored = classify(&system_one).await.unwrap();

        assert_eq!(scored.output, CategoryKind::Boolean);
        assert_eq!(scored.score, Some(Score(0.87)));
        assert_eq!(scored.prompt_version, Some(CLASSIFY_PROMPT_VERSION));
        let request = &system_one.requests()[0];
        assert_eq!(request.state, "Does AGR_USERS hold role assignments?");
        let Decision::Choice { criteria, .. } = &request.questions[QUESTION_ID] else {
            panic!("expected a choice question");
        };
        let names: Vec<_> = criteria.0.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["boolean", "choice", "ranking", "open"]);
    }

    #[tokio::test]
    async fn an_unknown_category_declines_with_the_raw_answer() {
        let system_one = Arc::new(ScriptedSystemOne::new([Ok(choice("haiku", 0.9))]));

        let error = classify(&system_one).await.err().unwrap();

        assert!(matches!(
            error.kind,
            crate::pipeline::StepErrorKind::Decline(DeclineReason::InvalidModelOutput(_))
        ));
        assert!(error.record.is_some());
        assert_eq!(error.prompt_version, Some(CLASSIFY_PROMPT_VERSION));
    }

    #[tokio::test]
    async fn without_system_one_every_question_is_open_and_unscored() {
        let scored = Classify::new(None)
            .run(&Question::new("anything"), Asked)
            .await
            .unwrap();
        assert_eq!(scored.output, CategoryKind::Open);
        assert_eq!(scored.score, None);
    }
}
