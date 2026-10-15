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
    async fn save(&self, user: &UserId, report: &RunReport) -> anyhow::Result<()> {
        let run = StoredRun {
            user: user.clone(),
            created_at: Utc::now(),
            report: report.clone(),
        };
        self.runs.lock().unwrap().insert(report.id, run);
        Ok(())
    }

    async fn get(&self, id: RunId) -> anyhow::Result<Option<StoredRun>> {
        Ok(self.runs.lock().unwrap().get(&id).cloned())
    }
}

impl MemoryRunStore {
    pub fn stored(&self) -> Vec<StoredRun> {
        self.runs.lock().unwrap().values().cloned().collect()
    }
}
