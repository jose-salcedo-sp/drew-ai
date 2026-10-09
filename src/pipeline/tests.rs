use super::{
    trace::{StepOutcome, TraceRecord, VerifyRecord},
    *,
};

struct Fetched(u32);
struct Doubled(u32);

struct Fetch<TCategory> {
    value: u32,
    score: f32,
    threshold: Option<f32>,
    category: PhantomData<TCategory>,
}

impl<TCategory: Category> Step for Fetch<TCategory> {
    const NAME: &'static str = "fetch";
    type Category = TCategory;
    type Input = Asked;
    type Output = Fetched;
    type Record = VerifyRecord;

    fn threshold(&self) -> Option<Score> {
        self.threshold.map(Score)
    }

    async fn run(
        &self,
        _question: &Question,
        _input: Asked,
    ) -> Result<Scored<Fetched, VerifyRecord>, StepError> {
        Ok(Scored::new(Fetched(self.value), empty_record()).with_score(Score(self.score)))
    }
}

struct Double;

impl Step for Double {
    const NAME: &'static str = "double";
    type Category = Open;
    type Input = Fetched;
    type Output = Doubled;
    type Record = VerifyRecord;

    async fn run(
        &self,
        _question: &Question,
        input: Fetched,
    ) -> Result<Scored<Doubled, VerifyRecord>, StepError> {
        Ok(Scored::new(Doubled(input.0 * 2), empty_record()))
    }
}

struct Refuse;

impl Step for Refuse {
    const NAME: &'static str = "refuse";
    type Category = Unclassified;
    type Input = Asked;
    type Output = ();
    type Record = VerifyRecord;

    async fn run(
        &self,
        _question: &Question,
        _input: Asked,
    ) -> Result<Scored<(), VerifyRecord>, StepError> {
        Err(
            StepError::decline(DeclineReason::InvalidModelOutput("not json".into())).with_record(
                VerifyRecord {
                    uncited_claims: vec!["raw".into()],
                    unknown_citations: vec![],
                },
            ),
        )
    }
}

fn empty_record() -> VerifyRecord {
    VerifyRecord {
        uncited_claims: vec![],
        unknown_citations: vec![],
    }
}

fn open_run() -> Run<Open, Asked> {
    Run::new(Question::new("does it work?")).classified()
}

fn fetch<TCategory>(score: f32, threshold: Option<f32>) -> Fetch<TCategory> {
    Fetch {
        value: 21,
        score,
        threshold,
        category: PhantomData,
    }
}

#[tokio::test]
async fn seq_drives_every_step_and_traces_each() {
    let pipeline = Seq(fetch(0.9, Some(0.5)), Double);

    let run = pipeline.drive(open_run()).await.unwrap();

    assert_eq!(run.state().0, 42);
    let steps: Vec<_> = run
        .trace()
        .entries()
        .iter()
        .map(|e| e.step.as_str())
        .collect();
    assert_eq!(steps, ["fetch", "double"]);
}

#[tokio::test]
async fn score_below_threshold_halts_with_the_trace_so_far() {
    let pipeline = Seq(fetch(0.2, Some(0.5)), Double);

    let halted = pipeline.drive(open_run()).await.err().unwrap();

    assert_eq!(halted.at, "fetch");
    assert_eq!(halted.category, Some(CategoryKind::Open));
    assert!(matches!(
        halted.halt,
        Halt::Declined {
            reason: DeclineReason::LowConfidence,
            ..
        }
    ));
    assert_eq!(halted.trace.entries().len(), 1);
}

#[tokio::test]
async fn declined_step_keeps_its_record() {
    let halted = Run::new(Question::new("q"))
        .then(&Refuse)
        .await
        .err()
        .unwrap();

    assert_eq!(halted.category, None);
    let entry = &halted.trace.entries()[0];
    assert!(matches!(entry.outcome, StepOutcome::Declined { .. }));
    assert!(matches!(entry.record, Some(TraceRecord::Verify(_))));
}

#[tokio::test]
async fn both_traces_the_other_branch_when_one_halts() {
    let halted = Run::new(Question::new("q"))
        .both("classify_retrieve", &Refuse, &fetch(0.9, None))
        .await
        .err()
        .unwrap();

    assert_eq!(halted.at, "refuse");
    let entries = halted.trace.entries();
    assert_eq!(entries.len(), 2);
    assert!(
        entries
            .iter()
            .all(|e| e.group.as_deref() == Some("classify_retrieve"))
    );
}

#[test]
fn trace_round_trips_through_json() {
    let entry = TraceEntry {
        step: "verify".into(),
        group: None,
        started_at: Utc::now(),
        duration_ms: 3,
        model: None,
        prompt_version: None,
        outcome: StepOutcome::Declined {
            reason: DeclineReason::Unverified("claim 2 has no citation".into()),
            score: None,
            threshold: None,
        },
        record: Some(empty_record().into()),
    };
    let json = serde_json::to_string(&entry).unwrap();
    assert_eq!(serde_json::from_str::<TraceEntry>(&json).unwrap(), entry);
}
