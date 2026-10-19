use std::{collections::HashMap, sync::Mutex};

use async_trait::async_trait;
use chrono::Utc;

use crate::{
    domain::{RunId, UserId},
    pipeline::report::RunReport,
};

use super::{RunStore, StoredRun};

#[derive(Default)]
pub struct MemoryRunStore {
    runs: Mutex<HashMap<RunId, StoredRun>>,
}

#[async_trait]
impl RunStore for MemoryRunStore {
    async fn save(&self, user: &UserId, report: &RunReport) -> eros::Result<()> {
        let run = StoredRun {
            user: user.clone(),
            created_at: Utc::now(),
            report: report.clone(),
        };
        self.runs
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .insert(report.id, run);
        Ok(())
    }

    async fn get(&self, id: RunId) -> eros::Result<Option<StoredRun>> {
        Ok(self
            .runs
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .get(&id)
            .cloned())
    }
}

impl MemoryRunStore {
    pub fn stored(&self) -> Vec<StoredRun> {
        self.runs
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .values()
            .cloned()
            .collect()
    }
}
