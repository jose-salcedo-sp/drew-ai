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

/// Fails every call, with a message like a database error's, for error-path tests.
pub struct FailingRunStore;

#[async_trait]
impl RunStore for FailingRunStore {
    async fn save(&self, _: &UserId, _: &RunReport) -> eros::Result<()> {
        eros::bail!("pool timed out connecting to postgres://drew@db.internal:5432/drew")
    }

    async fn get(&self, _: RunId) -> eros::Result<Option<StoredRun>> {
        eros::bail!("pool timed out connecting to postgres://drew@db.internal:5432/drew")
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
