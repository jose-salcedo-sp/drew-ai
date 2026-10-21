use std::{
    collections::{HashMap, hash_map::Entry},
    sync::Mutex,
};

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
        let mut runs = self
            .runs
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it");
        let Entry::Vacant(slot) = runs.entry(report.id) else {
            eros::bail!("run {} already saved", report.id);
        };
        slot.insert(StoredRun {
            user: user.clone(),
            created_at: Utc::now(),
            report: report.clone(),
        });
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
