//! Jira: an issue plus Drew's comments is one unit.

use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use chrono::{DateTime, TimeDelta, Utc};
use reqwest::{StatusCode, header::RETRY_AFTER};
use serde::{Deserialize, Deserializer, de::DeserializeOwned};

use crate::domain::{ChunkDraft, Document, Labels, Sensitivity, SourceKind};

use super::Source;

/// A unit longer than this is split on comment boundaries.
const MAX_WORDS: usize = 6_000;
const PAGE_SIZE: &str = "100";
const ISSUE_FIELDS: &str = "summary,description,updated,labels,security,project";
const MAX_ATTEMPTS: u32 = 5;
const MAX_RETRY_DELAY: Duration = Duration::from_secs(60);
const SETTINGS: [&str; 5] = [
    "JIRA_BASE_URL",
    "JIRA_EMAIL",
    "JIRA_API_TOKEN",
    "JIRA_JQL",
    "DREW_JIRA_ACCOUNT_ID",
];

#[derive(Debug, clap::Args)]
#[group(id = "jira")]
pub struct Args {
    /// Jira Cloud site, e.g. `https://example.atlassian.net`.
    #[arg(long, env = "JIRA_BASE_URL")]
    pub jira_base_url: Option<String>,
    /// Account email that owns the API token.
    #[arg(long, env = "JIRA_EMAIL")]
    pub jira_email: Option<String>,
    #[arg(long, env = "JIRA_API_TOKEN", hide_env_values = true)]
    pub jira_api_token: Option<String>,
    /// JQL selecting the issues to ingest, e.g. `project = ARM`. Must not contain `ORDER BY`.
    #[arg(long, env = "JIRA_JQL")]
    pub jira_jql: Option<String>,
    /// Drew's Atlassian account id; only issues he commented on are ingested.
    #[arg(long, env = "DREW_JIRA_ACCOUNT_ID")]
    pub drew_jira_account_id: Option<String>,
}

pub struct Connector {
    http: reqwest::Client,
    base_url: String,
    email: String,
    api_token: String,
    jql: String,
    drew_account_id: String,
}

impl Connector {
    /// `None` when this source is not configured.
    pub fn from_args(args: &Args) -> anyhow::Result<Option<Self>> {
        let values = [
            &args.jira_base_url,
            &args.jira_email,
            &args.jira_api_token,
            &args.jira_jql,
            &args.drew_jira_account_id,
        ]
        .map(|value| value.as_deref().map(str::trim).filter(|v| !v.is_empty()));
        match values {
            [None, None, None, None, None] => Ok(None),
            [
                Some(base_url),
                Some(email),
                Some(api_token),
                Some(jql),
                Some(drew_account_id),
            ] => Ok(Some(Self {
                http: reqwest::Client::builder()
                    .timeout(Duration::from_secs(60))
                    .build()?,
                base_url: base_url.trim_end_matches('/').to_owned(),
                email: email.to_owned(),
                api_token: api_token.to_owned(),
                jql: jql.to_owned(),
                drew_account_id: drew_account_id.to_owned(),
            })),
            _ => {
                let missing: Vec<&str> = SETTINGS
                    .iter()
                    .zip(&values)
                    .filter(|(_, value)| value.is_none())
                    .map(|(name, _)| *name)
                    .collect();
                anyhow::bail!(
                    "Jira is partially configured; also set {}",
                    missing.join(", ")
                )
            }
        }
    }

    async fn search(&self, jql: &str) -> anyhow::Result<Vec<Issue>> {
        let url = format!("{}/rest/api/3/search/jql", self.base_url);
        let mut issues = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let mut query = vec![
                ("jql", jql),
                ("fields", ISSUE_FIELDS),
                ("maxResults", PAGE_SIZE),
            ];
            if let Some(token) = &token {
                query.push(("nextPageToken", token));
            }
            let page: SearchPage = self.get(&url, &query).await?;
            issues.extend(page.issues);
            match page.next_page_token {
                Some(next) if !page.is_last => token = Some(next),
                _ => return Ok(issues),
            }
        }
    }

    async fn comments(&self, key: &str) -> anyhow::Result<Vec<Comment>> {
        let url = format!("{}/rest/api/3/issue/{key}/comment", self.base_url);
        let mut comments = Vec::new();
        loop {
            let start_at = comments.len().to_string();
            let query = [
                ("startAt", start_at.as_str()),
                ("maxResults", PAGE_SIZE),
                ("orderBy", "created"),
            ];
            let page: CommentPage = self.get(&url, &query).await?;
            let done =
                page.comments.is_empty() || page.start_at + page.comments.len() >= page.total;
            comments.extend(page.comments);
            if done {
                return Ok(comments);
            }
        }
    }

    async fn get<T: DeserializeOwned>(
        &self,
        url: &str,
        query: &[(&str, &str)],
    ) -> anyhow::Result<T> {
        let mut attempt = 1;
        loop {
            let response = self
                .http
                .get(url)
                .basic_auth(&self.email, Some(&self.api_token))
                .query(query)
                .send()
                .await
                .with_context(|| format!("GET {url}"))?;
            let status = response.status();
            if status == StatusCode::TOO_MANY_REQUESTS && attempt < MAX_ATTEMPTS {
                let delay = retry_after(&response)
                    .unwrap_or(Duration::from_secs(1 << attempt))
                    .min(MAX_RETRY_DELAY);
                tracing::warn!(url, attempt, ?delay, "Jira rate limit hit; retrying");
                tokio::time::sleep(delay).await;
                attempt += 1;
                continue;
            }
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("GET {url} returned {status}: {body}");
            }
            return response
                .json()
                .await
                .with_context(|| format!("decoding GET {url}"));
        }
    }

    /// `None` when Drew has not commented on the issue.
    fn document(&self, issue: Issue, comments: Vec<Comment>) -> Option<Document> {
        let Issue { key, fields } = issue;
        let url = format!("{}/browse/{key}", self.base_url);
        let drew_comments: Vec<(String, String)> = comments
            .into_iter()
            .filter_map(|comment| {
                let author = comment
                    .author
                    .filter(|author| author.account_id == self.drew_account_id)?;
                let body = comment.body.as_ref().map(adf::to_text).unwrap_or_default();
                (!body.is_empty()).then(|| {
                    let date = comment.created.format("%Y-%m-%d");
                    let text = format!("Comment from {} on {date}:\n{body}", author.display_name);
                    (comment.id, text)
                })
            })
            .collect();
        if drew_comments.is_empty() {
            return None;
        }
        let title = format!("{key}: {}", fields.summary.trim());
        let description = fields
            .description
            .as_ref()
            .map(adf::to_text)
            .unwrap_or_default();
        let header = if description.is_empty() {
            title.clone()
        } else {
            format!("{title}\n\n{description}")
        };
        Some(Document {
            source: SourceKind::Jira,
            chunks: chunks(&url, header, drew_comments),
            labels: labels(&fields),
            external_id: key,
            title,
            url,
            updated_at: fields.updated,
        })
    }
}

#[async_trait]
impl Source for Connector {
    fn kind(&self) -> SourceKind {
        SourceKind::Jira
    }

    async fn fetch_changed(&self, since: Option<DateTime<Utc>>) -> anyhow::Result<Vec<Document>> {
        let issues = self.search(&scope_jql(&self.jql, since)).await?;
        let mut documents = Vec::new();
        for issue in issues {
            if since.is_some_and(|since| issue.fields.updated <= since) {
                continue;
            }
            let comments = self
                .comments(&issue.key)
                .await
                .with_context(|| format!("fetching comments of {}", issue.key))?;
            documents.extend(self.document(issue, comments));
        }
        Ok(documents)
    }
}

fn scope_jql(scope: &str, since: Option<DateTime<Utc>>) -> String {
    match since {
        None => scope.to_owned(),
        // JQL dates have minute precision and are read in the API user's time zone, so the
        // window is widened by a day and `fetch_changed` filters on the exact timestamp.
        Some(since) => format!(
            "({scope}) AND updated > \"{}\"",
            (since - TimeDelta::days(1)).format("%Y-%m-%d %H:%M")
        ),
    }
}

fn retry_after(response: &reqwest::Response) -> Option<Duration> {
    let seconds = response
        .headers()
        .get(RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some(Duration::from_secs(seconds))
}

struct Part {
    url: String,
    sections: Vec<String>,
    words: usize,
}

/// One `issue` chunk, or `issue-1`, `issue-2`, ... split between comments when the unit is too long.
fn chunks(issue_url: &str, header: String, comments: Vec<(String, String)>) -> Vec<ChunkDraft> {
    let words = |text: &str| text.split_whitespace().count();
    let draft = |locator: String, url: String, sections: Vec<String>| ChunkDraft {
        locator,
        heading: None,
        url,
        text: sections.join("\n\n"),
        authoritative: true,
    };
    let total = words(&header) + comments.iter().map(|(_, text)| words(text)).sum::<usize>();
    if total <= MAX_WORDS {
        let sections = std::iter::once(header)
            .chain(comments.into_iter().map(|(_, text)| text))
            .collect();
        return vec![draft("issue".into(), issue_url.into(), sections)];
    }
    let mut parts = vec![Part {
        url: issue_url.to_owned(),
        words: words(&header),
        sections: vec![header],
    }];
    for (id, text) in comments {
        let count = words(&text);
        let part = parts.last_mut().expect("parts start with the header");
        if part.words + count > MAX_WORDS {
            parts.push(Part {
                url: format!("{issue_url}?focusedCommentId={id}"),
                sections: vec![text],
                words: count,
            });
        } else {
            part.sections.push(text);
            part.words += count;
        }
    }
    parts
        .into_iter()
        .enumerate()
        .map(|(i, part)| draft(format!("issue-{}", i + 1), part.url, part.sections))
        .collect()
}

fn labels(fields: &IssueFields) -> Labels {
    let mut acl = vec![format!("project:{}", fields.project.key)];
    acl.extend(
        fields
            .security
            .as_ref()
            .map(|level| format!("security-level:{}", level.name)),
    );
    let jira_labels: Vec<String> = fields.labels.iter().map(|l| l.to_lowercase()).collect();
    let has = |label: &str| jira_labels.iter().any(|l| l == label);
    let sensitivity = if has("drew-confidential") {
        Sensitivity::Confidential
    } else if has("drew-public") {
        Sensitivity::Public
    } else {
        Sensitivity::Internal
    };
    let customer = jira_labels
        .iter()
        .find_map(|l| l.strip_prefix("customer-").filter(|name| !name.is_empty()))
        .map(str::to_owned);
    Labels {
        acl,
        customer,
        sensitivity,
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchPage {
    #[serde(default)]
    issues: Vec<Issue>,
    next_page_token: Option<String>,
    #[serde(default)]
    is_last: bool,
}

#[derive(Deserialize)]
struct Issue {
    key: String,
    fields: IssueFields,
}

#[derive(Deserialize)]
struct IssueFields {
    #[serde(default)]
    summary: String,
    description: Option<adf::Node>,
    #[serde(deserialize_with = "jira_time")]
    updated: DateTime<Utc>,
    #[serde(default)]
    labels: Vec<String>,
    security: Option<SecurityLevel>,
    project: Project,
}

#[derive(Deserialize)]
struct SecurityLevel {
    name: String,
}

#[derive(Deserialize)]
struct Project {
    key: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CommentPage {
    #[serde(default)]
    start_at: usize,
    #[serde(default)]
    total: usize,
    #[serde(default)]
    comments: Vec<Comment>,
}

#[derive(Deserialize)]
struct Comment {
    id: String,
    author: Option<Author>,
    body: Option<adf::Node>,
    #[serde(deserialize_with = "jira_time")]
    created: DateTime<Utc>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Author {
    account_id: String,
    #[serde(default)]
    display_name: String,
}

/// Jira timestamps look like `2026-10-01T12:34:56.789+0000`, which is not RFC 3339.
fn jira_time<'de, D: Deserializer<'de>>(deserializer: D) -> Result<DateTime<Utc>, D::Error> {
    let text = String::deserialize(deserializer)?;
    DateTime::parse_from_str(&text, "%Y-%m-%dT%H:%M:%S%.f%z")
        .map(|time| time.to_utc())
        .map_err(serde::de::Error::custom)
}

/// Atlassian Document Format, the rich-text JSON of Jira Cloud's v3 API, rendered as plain text.
mod adf {
    use chrono::{DateTime, Utc};
    use serde::Deserialize;
    use serde_json::{Map, Value};

    #[derive(Debug, Default, Deserialize)]
    pub struct Node {
        #[serde(rename = "type", default)]
        kind: String,
        #[serde(default)]
        text: String,
        #[serde(default)]
        attrs: Map<String, Value>,
        #[serde(default)]
        marks: Vec<Mark>,
        #[serde(default)]
        content: Vec<Node>,
    }

    #[derive(Debug, Deserialize)]
    struct Mark {
        #[serde(rename = "type")]
        kind: String,
        #[serde(default)]
        attrs: Map<String, Value>,
    }

    impl Node {
        fn attr(&self, name: &str) -> Option<&str> {
            self.attrs.get(name).and_then(Value::as_str)
        }
    }

    /// Blank lines between blocks, `- ` / `1. ` list items, fenced code, and `|` tables.
    pub fn to_text(doc: &Node) -> String {
        block(doc).trim().to_owned()
    }

    fn blocks(nodes: &[Node], separator: &str) -> String {
        nodes
            .iter()
            .map(block)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(separator)
    }

    fn block(node: &Node) -> String {
        match node.kind.as_str() {
            "paragraph" | "heading" => inline(&node.content).trim().to_owned(),
            "bulletList" | "orderedList" | "taskList" | "decisionList" => list(node),
            "codeBlock" => {
                let code: String = node.content.iter().map(|n| n.text.as_str()).collect();
                let language = node.attr("language").unwrap_or_default();
                format!("```{language}\n{}\n```", code.trim_end())
            }
            "blockquote" => blocks(&node.content, "\n\n")
                .lines()
                .map(|line| format!("> {line}").trim_end().to_owned())
                .collect::<Vec<_>>()
                .join("\n"),
            "rule" => "---".to_owned(),
            "table" => table(node),
            "media" | "mediaSingle" | "mediaGroup" => String::new(),
            _ if node.content.iter().any(is_inline) => inline(&node.content).trim().to_owned(),
            _ if node.content.is_empty() => inline(std::slice::from_ref(node)).trim().to_owned(),
            _ => blocks(&node.content, "\n\n"),
        }
    }

    fn is_inline(node: &Node) -> bool {
        matches!(
            node.kind.as_str(),
            "text"
                | "hardBreak"
                | "mention"
                | "emoji"
                | "inlineCard"
                | "date"
                | "status"
                | "mediaInline"
                | "placeholder"
                | "inlineExtension"
        )
    }

    fn list(node: &Node) -> String {
        let mut number = node.attrs.get("order").and_then(Value::as_u64).unwrap_or(1);
        node.content
            .iter()
            .map(|item| {
                let marker = match (node.kind.as_str(), item.kind.as_str()) {
                    ("orderedList", _) => {
                        number += 1;
                        format!("{}. ", number - 1)
                    }
                    (_, "taskItem") if item.attr("state") == Some("DONE") => "[x] ".to_owned(),
                    (_, "taskItem") => "[ ] ".to_owned(),
                    (_, "listItem" | "decisionItem") => "- ".to_owned(),
                    _ => "  ".to_owned(),
                };
                let body = if item.kind == "listItem" {
                    blocks(&item.content, "\n")
                } else {
                    block(item)
                };
                hang(&marker, &body)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Prefixes the first line with `marker` and indents the rest to line up under it.
    fn hang(marker: &str, body: &str) -> String {
        let indent = " ".repeat(marker.chars().count());
        body.lines()
            .enumerate()
            .map(|(i, line)| match i {
                0 => format!("{marker}{line}"),
                _ if line.is_empty() => String::new(),
                _ => format!("{indent}{line}"),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn table(node: &Node) -> String {
        let mut lines = Vec::new();
        for row in &node.content {
            let cells: Vec<String> = row
                .content
                .iter()
                .map(|cell| blocks(&cell.content, " ").replace('\n', " "))
                .collect();
            lines.push(format!("| {} |", cells.join(" | ")));
            if !row.content.is_empty() && row.content.iter().all(|c| c.kind == "tableHeader") {
                lines.push(format!("| {} |", vec!["---"; cells.len()].join(" | ")));
            }
        }
        lines.join("\n")
    }

    fn inline(nodes: &[Node]) -> String {
        let mut out = String::new();
        for node in nodes {
            match node.kind.as_str() {
                "text" => out.push_str(&marked_text(node)),
                "hardBreak" => out.push('\n'),
                "mention" => {
                    let name = node
                        .attr("text")
                        .filter(|text| !text.is_empty())
                        .or(node.attr("id"))
                        .unwrap_or_default();
                    if !name.starts_with('@') {
                        out.push('@');
                    }
                    out.push_str(name);
                }
                "emoji" => out.push_str(
                    node.attr("text")
                        .or(node.attr("shortName"))
                        .unwrap_or_default(),
                ),
                "inlineCard" | "blockCard" | "embedCard" => {
                    out.push_str(node.attr("url").unwrap_or_default())
                }
                "date" => out.push_str(&date(node)),
                "status" => out.push_str(node.attr("text").unwrap_or_default()),
                _ => out.push_str(&inline(&node.content)),
            }
        }
        out
    }

    fn marked_text(node: &Node) -> String {
        let mut text = node.text.clone();
        for mark in &node.marks {
            match mark.kind.as_str() {
                "code" => text = format!("`{text}`"),
                "link" => {
                    if let Some(href) = mark.attrs.get("href").and_then(Value::as_str)
                        && href != node.text
                    {
                        text = format!("{text} ({href})");
                    }
                }
                _ => {}
            }
        }
        text
    }

    /// ADF dates are UTC midnight as a string of epoch milliseconds.
    fn date(node: &Node) -> String {
        node.attr("timestamp")
            .and_then(|millis| millis.parse().ok())
            .and_then(DateTime::<Utc>::from_timestamp_millis)
            .map(|time| time.format("%Y-%m-%d").to_string())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{basic_auth, method, path, query_param, query_param_is_missing},
    };

    use super::*;

    const SEARCH_PAGE_1: &str = include_str!("../../tests/fixtures/jira/search-page-1.json");
    const SEARCH_PAGE_2: &str = include_str!("../../tests/fixtures/jira/search-page-2.json");
    const SEARCH_SINCE: &str = include_str!("../../tests/fixtures/jira/search-since.json");
    const COMMENTS_ARM_1_PAGE_1: &str =
        include_str!("../../tests/fixtures/jira/comments-ARM-1-page-1.json");
    const COMMENTS_ARM_1_PAGE_2: &str =
        include_str!("../../tests/fixtures/jira/comments-ARM-1-page-2.json");
    const COMMENTS_ARM_2: &str = include_str!("../../tests/fixtures/jira/comments-ARM-2.json");
    const COMMENTS_ARM_3: &str = include_str!("../../tests/fixtures/jira/comments-ARM-3.json");
    const ADF: &str = include_str!("../../tests/fixtures/jira/adf.json");
    const ADF_TEXT: &str = include_str!("../../tests/fixtures/jira/adf.txt");

    fn args(base_url: &str) -> Args {
        Args {
            jira_base_url: Some(base_url.into()),
            jira_email: Some("drew-bot@example.test".into()),
            jira_api_token: Some("secret".into()),
            jira_jql: Some("project = ARM".into()),
            drew_jira_account_id: Some("drew-account".into()),
        }
    }

    fn connector(base_url: &str) -> Connector {
        Connector::from_args(&args(base_url))
            .unwrap()
            .expect("configured")
    }

    fn fixture(body: &str) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_raw(body, "application/json")
    }

    async fn mount_comments(server: &MockServer, key: &str, body: &str) {
        Mock::given(method("GET"))
            .and(path(format!("/rest/api/3/issue/{key}/comment")))
            .respond_with(fixture(body))
            .mount(server)
            .await;
    }

    fn paragraph(text: &str) -> serde_json::Value {
        json!({"type": "doc", "version": 1, "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": text}]}
        ]})
    }

    fn issue(key: &str, description: &str, labels: &[&str]) -> Issue {
        serde_json::from_value(json!({
            "key": key,
            "fields": {
                "summary": "Long thread",
                "description": paragraph(description),
                "updated": "2026-10-01T10:00:00.000+0000",
                "labels": labels,
                "project": {"key": "ARM"},
            }
        }))
        .unwrap()
    }

    fn comment(id: &str, account_id: &str, text: &str) -> Comment {
        serde_json::from_value(json!({
            "id": id,
            "author": {"accountId": account_id, "displayName": "Drew Steinfatt"},
            "body": paragraph(text),
            "created": "2026-09-01T09:00:00.000+0000",
        }))
        .unwrap()
    }

    fn words(n: usize) -> String {
        vec!["word"; n].join(" ")
    }

    #[tokio::test]
    async fn fetches_every_page_and_keeps_issues_drew_commented_on() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/rest/api/3/search/jql"))
            .and(basic_auth("drew-bot@example.test", "secret"))
            .and(query_param("jql", "project = ARM"))
            .and(query_param("fields", ISSUE_FIELDS))
            .and(query_param_is_missing("nextPageToken"))
            .respond_with(fixture(SEARCH_PAGE_1))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/rest/api/3/search/jql"))
            .and(query_param("nextPageToken", "page-2"))
            .respond_with(fixture(SEARCH_PAGE_2))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/rest/api/3/issue/ARM-1/comment"))
            .and(query_param("startAt", "0"))
            .respond_with(fixture(COMMENTS_ARM_1_PAGE_1))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/rest/api/3/issue/ARM-1/comment"))
            .and(query_param("startAt", "2"))
            .respond_with(fixture(COMMENTS_ARM_1_PAGE_2))
            .expect(1)
            .mount(&server)
            .await;
        mount_comments(&server, "ARM-2", COMMENTS_ARM_2).await;
        mount_comments(&server, "ARM-3", COMMENTS_ARM_3).await;

        let base = server.uri();
        let documents = connector(&base).fetch_changed(None).await.unwrap();

        let keys: Vec<&str> = documents.iter().map(|d| d.external_id.as_str()).collect();
        assert_eq!(keys, ["ARM-1", "ARM-3"]);

        let arm1 = &documents[0];
        assert_eq!(arm1.source, SourceKind::Jira);
        assert_eq!(arm1.title, "ARM-1: Role sync drops AGR_USERS rows");
        assert_eq!(arm1.url, format!("{base}/browse/ARM-1"));
        assert_eq!(arm1.updated_at.to_rfc3339(), "2026-10-01T10:00:00+00:00");
        assert_eq!(
            arm1.chunks,
            [ChunkDraft {
                locator: "issue".into(),
                heading: None,
                url: format!("{base}/browse/ARM-1"),
                text: "ARM-1: Role sync drops AGR_USERS rows\n\n\
                       Nightly sync loses rows for composite roles.\n\n\
                       Comment from Drew Steinfatt on 2026-09-02:\nCheck `S_TCODE` first.\n\n\
                       Comment from Drew Steinfatt on 2026-09-05:\n\
                       Fixed in the runbook (https://wiki.example.test/runbook)."
                    .into(),
                authoritative: true,
            }]
        );
        assert_eq!(
            arm1.labels,
            Labels {
                acl: vec!["project:ARM".into(), "security-level:Internal Only".into()],
                customer: Some("acme".into()),
                sensitivity: Sensitivity::Confidential,
            }
        );

        let arm3 = &documents[1];
        assert_eq!(
            arm3.chunks[0].text,
            "ARM-3: Can SU01 reset locked users?\n\n\
             Comment from Drew Steinfatt on 2026-10-09:\nYes, use SU01."
        );
        assert_eq!(
            arm3.labels,
            Labels {
                acl: vec!["project:ARM".into()],
                customer: None,
                sensitivity: Sensitivity::Public,
            }
        );
    }

    #[tokio::test]
    async fn since_narrows_the_jql_and_drops_issues_not_updated_after_it() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/rest/api/3/search/jql"))
            .and(query_param(
                "jql",
                "(project = ARM) AND updated > \"2026-10-07 12:30\"",
            ))
            .respond_with(fixture(SEARCH_SINCE))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/rest/api/3/issue/ARM-2/comment"))
            .respond_with(fixture(COMMENTS_ARM_2))
            .expect(0)
            .mount(&server)
            .await;
        mount_comments(&server, "ARM-3", COMMENTS_ARM_3).await;

        let since = "2026-10-08T12:30:45Z".parse().unwrap();
        let documents = connector(&server.uri())
            .fetch_changed(Some(since))
            .await
            .unwrap();

        let keys: Vec<&str> = documents.iter().map(|d| d.external_id.as_str()).collect();
        assert_eq!(keys, ["ARM-3"]);
    }

    #[tokio::test]
    async fn retries_after_rate_limiting() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/rest/api/3/search/jql"))
            .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "0"))
            .up_to_n_times(1)
            .with_priority(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/rest/api/3/search/jql"))
            .respond_with(fixture(SEARCH_PAGE_2))
            .mount(&server)
            .await;
        mount_comments(&server, "ARM-3", COMMENTS_ARM_3).await;

        let documents = connector(&server.uri()).fetch_changed(None).await.unwrap();

        assert_eq!(documents.len(), 1);
        let searches = server
            .received_requests()
            .await
            .unwrap()
            .into_iter()
            .filter(|r| r.url.path() == "/rest/api/3/search/jql")
            .count();
        assert_eq!(searches, 2);
    }

    #[tokio::test]
    async fn reports_errors_with_the_response_body() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/rest/api/3/search/jql"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad JQL"))
            .mount(&server)
            .await;

        let error = connector(&server.uri())
            .fetch_changed(None)
            .await
            .unwrap_err();

        assert!(
            error.to_string().contains("400 Bad Request: bad JQL"),
            "{error}"
        );
    }

    #[test]
    fn converts_adf_to_plain_text() {
        let doc: adf::Node = serde_json::from_str(ADF).unwrap();
        assert_eq!(adf::to_text(&doc), ADF_TEXT.trim_end());
    }

    #[test]
    fn skips_issues_without_comments_from_drew() {
        let connector = connector("https://example.atlassian.net");
        let issue = issue("ARM-9", "Question", &[]);
        let comments = vec![comment("1", "someone-else", "I think so")];
        assert!(connector.document(issue, comments).is_none());
    }

    #[test]
    fn splits_long_units_between_comments() {
        let connector = connector("https://example.atlassian.net/");
        let issue = issue("ARM-9", &words(1_000), &[]);
        let comments = vec![
            comment("101", "drew-account", &words(2_500)),
            comment("102", "drew-account", &words(2_500)),
            comment("103", "drew-account", &words(2_500)),
        ];

        let document = connector.document(issue, comments).unwrap();

        let parts: Vec<(&str, &str)> = document
            .chunks
            .iter()
            .map(|c| (c.locator.as_str(), c.url.as_str()))
            .collect();
        assert_eq!(
            parts,
            [
                ("issue-1", "https://example.atlassian.net/browse/ARM-9"),
                (
                    "issue-2",
                    "https://example.atlassian.net/browse/ARM-9?focusedCommentId=102"
                ),
            ]
        );
        assert!(
            document.chunks[0]
                .text
                .starts_with("ARM-9: Long thread\n\nword")
        );
        assert_eq!(document.chunks[0].text.matches("Comment from").count(), 1);
        assert!(document.chunks[1].text.starts_with("Comment from"));
        assert_eq!(document.chunks[1].text.matches("Comment from").count(), 2);
        assert!(document.chunks.iter().all(|c| c.authoritative));
    }

    #[test]
    fn short_units_are_one_chunk() {
        let connector = connector("https://example.atlassian.net");
        let issue = issue("ARM-9", &words(10), &[]);
        let comments = vec![comment("101", "drew-account", "Yes.")];
        let document = connector.document(issue, comments).unwrap();
        assert_eq!(document.chunks.len(), 1);
        assert_eq!(document.chunks[0].locator, "issue");
    }

    #[test]
    fn labels_default_to_internal_and_confidential_wins() {
        let plain = issue("ARM-9", "", &[]);
        assert_eq!(
            labels(&plain.fields),
            Labels {
                acl: vec!["project:ARM".into()],
                customer: None,
                sensitivity: Sensitivity::Internal,
            }
        );
        let both = issue(
            "ARM-9",
            "",
            &[
                "Drew-Public",
                "drew-confidential",
                "customer-",
                "customer-Globex",
            ],
        );
        let labels = labels(&both.fields);
        assert_eq!(labels.sensitivity, Sensitivity::Confidential);
        assert_eq!(labels.customer.as_deref(), Some("globex"));
    }

    #[test]
    fn from_args_is_none_when_unconfigured_and_errors_when_partial() {
        let none = Args {
            jira_base_url: None,
            jira_email: None,
            jira_api_token: None,
            jira_jql: None,
            drew_jira_account_id: Some("  ".into()),
        };
        assert!(Connector::from_args(&none).unwrap().is_none());

        let partial = Args {
            jira_base_url: Some("https://example.atlassian.net".into()),
            jira_jql: Some("project = ARM".into()),
            ..none
        };
        let error = Connector::from_args(&partial)
            .err()
            .expect("partial config");
        assert_eq!(
            error.to_string(),
            "Jira is partially configured; also set JIRA_EMAIL, JIRA_API_TOKEN, DREW_JIRA_ACCOUNT_ID"
        );
    }
}
