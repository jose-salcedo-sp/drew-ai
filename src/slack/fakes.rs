use std::sync::Mutex;

use async_trait::async_trait;

use super::api::SlackApi;

#[derive(Default)]
pub struct FakeSlack {
    pub posts: Mutex<Vec<(String, Option<String>, String)>>,
    pub updates: Mutex<Vec<(String, String, String)>>,
    pub ws_url: String,
}

#[async_trait]
impl SlackApi for FakeSlack {
    async fn open_connection(&self) -> anyhow::Result<String> {
        Ok(self.ws_url.clone())
    }

    async fn post_message(
        &self,
        channel: &str,
        thread_ts: Option<&str>,
        text: &str,
    ) -> anyhow::Result<String> {
        self.posts.lock().unwrap().push((
            channel.to_owned(),
            thread_ts.map(str::to_owned),
            text.to_owned(),
        ));
        Ok(format!("{}.0001", self.posts.lock().unwrap().len()))
    }

    async fn update_message(&self, channel: &str, ts: &str, text: &str) -> anyhow::Result<()> {
        self.updates
            .lock()
            .unwrap()
            .push((channel.to_owned(), ts.to_owned(), text.to_owned()));
        Ok(())
    }
}
