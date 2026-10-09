//! ARM Confluence space: one chunk per heading section, linked to its parent page.

use std::{collections::HashSet, fmt, mem, time::Duration};

use anyhow::{Context, bail};
use async_trait::async_trait;
use chrono::{DateTime, TimeDelta, Utc};
use futures::{StreamExt, TryStreamExt, stream};
use quick_xml::{
    Reader, XmlVersion,
    escape::{resolve_html5_entity, resolve_predefined_entity},
    events::{BytesRef, BytesStart, Event},
};
use reqwest::{StatusCode, Url, header::RETRY_AFTER};
use serde::{Deserialize, de::DeserializeOwned};

use crate::domain::{ChunkDraft, Document, Labels, Sensitivity, SourceKind};

use super::Source;

/// The v2 API's maximum page size, and the maximum number of ids per `GET /pages?id=`.
const V2_LIMIT: usize = 250;
const CQL_LIMIT: usize = 100;
const RESTRICTION_LIMIT: usize = 100;
const MAX_RETRIES: u32 = 5;
/// Pages whose labels and restrictions are fetched concurrently.
const CONCURRENCY: usize = 8;
const PUBLIC_LABEL: &str = "drew-public";
const CONFIDENTIAL_LABEL: &str = "drew-confidential";
const CUSTOMER_LABEL_PREFIX: &str = "customer-";
const INTRO: &str = "intro";

#[derive(Clone, Default, clap::Args)]
#[group(id = "confluence")]
pub struct Args {
    /// Confluence Cloud site, e.g. `https://example.atlassian.net`.
    #[arg(long, env = "CONFLUENCE_BASE_URL")]
    pub confluence_base_url: Option<String>,
    /// Key of the space to ingest.
    #[arg(long, env = "CONFLUENCE_SPACE_KEY")]
    pub confluence_space_key: Option<String>,
    /// Email of the account that owns the API token.
    #[arg(long, env = "CONFLUENCE_EMAIL")]
    pub confluence_email: Option<String>,
    #[arg(long, env = "CONFLUENCE_API_TOKEN", hide_env_values = true)]
    pub confluence_api_token: Option<String>,
    /// Drew's Atlassian account id; pages he created are authoritative.
    #[arg(long, env = "DREW_CONFLUENCE_ACCOUNT_ID")]
    pub drew_confluence_account_id: Option<String>,
}

impl fmt::Debug for Args {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Args")
            .field("confluence_base_url", &self.confluence_base_url)
            .field("confluence_space_key", &self.confluence_space_key)
            .field("confluence_email", &self.confluence_email)
            .field(
                "confluence_api_token",
                &self.confluence_api_token.as_ref().map(|_| "<redacted>"),
            )
            .field(
                "drew_confluence_account_id",
                &self.drew_confluence_account_id,
            )
            .finish()
    }
}

pub struct Connector {
    http: reqwest::Client,
    /// Site root without a trailing slash or `/wiki`.
    site: String,
    space_key: String,
    email: String,
    api_token: String,
    drew_account_id: String,
}

impl Connector {
    /// `None` when this source is not configured.
    pub fn from_args(args: &Args) -> anyhow::Result<Option<Self>> {
        let settings = [
            ("CONFLUENCE_BASE_URL", &args.confluence_base_url),
            ("CONFLUENCE_SPACE_KEY", &args.confluence_space_key),
            ("CONFLUENCE_EMAIL", &args.confluence_email),
            ("CONFLUENCE_API_TOKEN", &args.confluence_api_token),
            (
                "DREW_CONFLUENCE_ACCOUNT_ID",
                &args.drew_confluence_account_id,
            ),
        ];
        let value = |v: &Option<String>| {
            v.as_deref()
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(str::to_owned)
        };
        let missing: Vec<_> = settings
            .iter()
            .filter(|(_, v)| value(v).is_none())
            .map(|(name, _)| *name)
            .collect();
        if missing.len() == settings.len() {
            return Ok(None);
        }
        if !missing.is_empty() {
            bail!(
                "Confluence is partially configured; missing {}",
                missing.join(", ")
            );
        }
        let [base_url, space_key, email, api_token, drew_account_id] =
            settings.map(|(_, v)| value(v).unwrap_or_default());

        let site = base_url.trim_end_matches('/');
        let site = site.strip_suffix("/wiki").unwrap_or(site).to_owned();
        Url::parse(&site).with_context(|| format!("invalid CONFLUENCE_BASE_URL {site:?}"))?;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()?;
        Ok(Some(Self {
            http,
            site,
            space_key,
            email,
            api_token,
            drew_account_id,
        }))
    }

    fn url(&self, path: &str, query: &[(&str, &str)]) -> anyhow::Result<Url> {
        Url::parse_with_params(&format!("{}{path}", self.site), query)
            .with_context(|| format!("building Confluence URL for {path}"))
    }

    /// v2 `next` links include `/wiki`; v1 links are relative to it.
    fn next_url(&self, next: &str) -> anyhow::Result<Url> {
        let url = if next.starts_with("http://") || next.starts_with("https://") {
            next.to_owned()
        } else if next.starts_with("/wiki/") {
            format!("{}{next}", self.site)
        } else {
            format!("{}/wiki{next}", self.site)
        };
        Url::parse(&url).with_context(|| format!("invalid Confluence next link {next:?}"))
    }

    async fn get_json<T: DeserializeOwned>(&self, url: Url) -> anyhow::Result<T> {
        let mut attempt = 0;
        loop {
            let response = self
                .http
                .get(url.clone())
                .basic_auth(&self.email, Some(&self.api_token))
                .header(reqwest::header::ACCEPT, "application/json")
                .send()
                .await
                .with_context(|| format!("GET {url}"))?;
            let status = response.status();
            if status == StatusCode::TOO_MANY_REQUESTS && attempt < MAX_RETRIES {
                let wait = retry_after(&response).unwrap_or(Duration::from_secs(1 << attempt));
                tracing::warn!(%url, ?wait, attempt, "Confluence rate limited; retrying");
                tokio::time::sleep(wait).await;
                attempt += 1;
                continue;
            }
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                let body: String = body.chars().take(500).collect();
                bail!("GET {url} returned {status}: {body}");
            }
            return response
                .json()
                .await
                .with_context(|| format!("decoding response of GET {url}"));
        }
    }

    /// Follows `_links.next` until the last page.
    async fn get_all<T: DeserializeOwned>(&self, mut url: Url) -> anyhow::Result<Vec<T>> {
        let mut all = Vec::new();
        loop {
            let page: Paged<T> = self.get_json(url).await?;
            all.extend(page.results);
            match page.links.next {
                Some(next) => url = self.next_url(&next)?,
                None => return Ok(all),
            }
        }
    }

    async fn space_id(&self) -> anyhow::Result<String> {
        let url = self.url("/wiki/api/v2/spaces", &[("keys", &self.space_key)])?;
        let spaces: Vec<Space> = self.get_all(url).await?;
        spaces
            .into_iter()
            .find(|s| s.key == self.space_key)
            .map(|s| s.id)
            .with_context(|| format!("Confluence space {} not found", self.space_key))
    }

    async fn all_pages(&self) -> anyhow::Result<Vec<Page>> {
        let space_id = self.space_id().await?;
        let url = self.url(
            &format!("/wiki/api/v2/spaces/{space_id}/pages"),
            &[
                ("body-format", "storage"),
                ("status", "current"),
                ("limit", &V2_LIMIT.to_string()),
            ],
        )?;
        self.get_all(url).await
    }

    /// A superset of the pages changed after `since`; the caller filters on the exact timestamp.
    async fn pages_modified_around(&self, since: DateTime<Utc>) -> anyhow::Result<Vec<Page>> {
        // CQL dates have minute precision and are read in the API user's profile timezone,
        // so the query starts a day early.
        let from = (since - TimeDelta::days(1)).format("%Y-%m-%d %H:%M");
        let cql = format!(
            r#"space = "{}" and type = page and lastmodified >= "{from}" order by lastmodified asc"#,
            self.space_key.replace('\\', "\\\\").replace('"', "\\\"")
        );
        let url = self.url(
            "/wiki/rest/api/content/search",
            &[("cql", &cql), ("limit", &CQL_LIMIT.to_string())],
        )?;
        let hits: Vec<SearchHit> = self.get_all(url).await?;
        let mut seen = HashSet::new();
        let ids: Vec<String> = hits
            .into_iter()
            .map(|h| h.id)
            .filter(|id| seen.insert(id.clone()))
            .collect();

        let mut pages = Vec::with_capacity(ids.len());
        for batch in ids.chunks(V2_LIMIT) {
            let url = self.url(
                "/wiki/api/v2/pages",
                &[
                    ("id", &batch.join(",")),
                    ("body-format", "storage"),
                    ("status", "current"),
                    ("limit", &V2_LIMIT.to_string()),
                ],
            )?;
            pages.extend(self.get_all::<Page>(url).await?);
        }
        Ok(pages)
    }

    async fn label_names(&self, page_id: &str) -> anyhow::Result<Vec<String>> {
        let url = self.url(
            &format!("/wiki/api/v2/pages/{page_id}/labels"),
            &[("limit", &V2_LIMIT.to_string())],
        )?;
        let labels: Vec<Label> = self.get_all(url).await?;
        Ok(labels
            .into_iter()
            .filter(|l| l.prefix.as_deref().is_none_or(|p| p == "global"))
            .map(|l| l.name)
            .collect())
    }

    /// `user:{accountId}` and `group:{name}` principals of the page's own read restrictions.
    async fn read_restrictions(&self, page_id: &str) -> anyhow::Result<Vec<String>> {
        let mut principals = Vec::new();
        let mut start = 0;
        loop {
            let url = self.url(
                &format!("/wiki/rest/api/content/{page_id}/restriction/byOperation/read"),
                &[
                    ("expand", "restrictions.user,restrictions.group"),
                    ("start", &start.to_string()),
                    ("limit", &RESTRICTION_LIMIT.to_string()),
                ],
            )?;
            let Restrictions { user, group } =
                self.get_json::<ReadRestrictions>(url).await?.restrictions;
            let more = user.is_full() || group.is_full();
            let step = user.limit.max(group.limit);
            principals.extend(
                user.results
                    .into_iter()
                    .filter_map(|u| u.account_id)
                    .map(|id| format!("user:{id}")),
            );
            principals.extend(
                group
                    .results
                    .into_iter()
                    .map(|g| format!("group:{}", g.name)),
            );
            if !more {
                break;
            }
            start += step;
        }
        principals.sort();
        principals.dedup();
        Ok(principals)
    }

    async fn document(&self, page: Page) -> anyhow::Result<Document> {
        let (label_names, acl) =
            tokio::try_join!(self.label_names(&page.id), self.read_restrictions(&page.id))?;
        let url = format!(
            "{}/wiki/spaces/{}/pages/{}",
            self.site, self.space_key, page.id
        );
        let authoritative = page.author_id.as_deref() == Some(self.drew_account_id.as_str());
        let storage = page
            .body
            .as_ref()
            .and_then(|b| b.storage.as_ref())
            .map_or("", |s| s.value.as_str());
        let chunks = chunks(&url, storage, authoritative)
            .with_context(|| format!("parsing Confluence page {}", page.id))?;
        let acl = if acl.is_empty() {
            vec![format!("space:{}", self.space_key)]
        } else {
            acl
        };
        let (sensitivity, customer) = classify(&label_names);
        Ok(Document {
            source: SourceKind::Confluence,
            external_id: page.id,
            title: page.title,
            url,
            updated_at: page.version.created_at,
            labels: Labels {
                acl,
                customer,
                sensitivity,
            },
            chunks,
        })
    }
}

#[async_trait]
impl Source for Connector {
    fn kind(&self) -> SourceKind {
        SourceKind::Confluence
    }

    async fn fetch_changed(&self, since: Option<DateTime<Utc>>) -> anyhow::Result<Vec<Document>> {
        let pages = match since {
            None => self.all_pages().await?,
            Some(since) => self.pages_modified_around(since).await?,
        };
        let changed: Vec<Page> = pages
            .into_iter()
            .filter(|p| since.is_none_or(|since| p.version.created_at > since))
            .collect();
        tracing::debug!(pages = changed.len(), "fetched changed Confluence pages");
        stream::iter(changed)
            .map(|page| self.document(page))
            .buffered(CONCURRENCY)
            .try_collect()
            .await
    }
}

fn retry_after(response: &reqwest::Response) -> Option<Duration> {
    let value = response.headers().get(RETRY_AFTER)?.to_str().ok()?.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let at = DateTime::parse_from_rfc2822(value).ok()?;
    Some(
        (at.with_timezone(&Utc) - Utc::now())
            .to_std()
            .unwrap_or_default(),
    )
}

/// Sensitivity and customer from page labels; `drew-confidential` wins over `drew-public`.
fn classify(labels: &[String]) -> (Sensitivity, Option<String>) {
    let has = |name: &str| labels.iter().any(|l| l == name);
    let sensitivity = if has(CONFIDENTIAL_LABEL) {
        Sensitivity::Confidential
    } else if has(PUBLIC_LABEL) {
        Sensitivity::Public
    } else {
        Sensitivity::Internal
    };
    let mut customers: Vec<&str> = labels
        .iter()
        .filter_map(|l| l.strip_prefix(CUSTOMER_LABEL_PREFIX))
        .filter(|c| !c.is_empty())
        .collect();
    customers.sort_unstable();
    if customers.len() > 1 {
        tracing::warn!(
            ?customers,
            "page has several customer labels; using the first"
        );
    }
    (sensitivity, customers.first().map(|c| (*c).to_owned()))
}

#[derive(Deserialize)]
struct Paged<T> {
    results: Vec<T>,
    #[serde(rename = "_links", default)]
    links: Links,
}

#[derive(Default, Deserialize)]
struct Links {
    next: Option<String>,
}

#[derive(Deserialize)]
struct Space {
    id: String,
    key: String,
}

#[derive(Deserialize)]
struct SearchHit {
    id: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Page {
    id: String,
    title: String,
    /// The account that created the page.
    author_id: Option<String>,
    version: PageVersion,
    body: Option<PageBody>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PageVersion {
    created_at: DateTime<Utc>,
}

#[derive(Deserialize)]
struct PageBody {
    storage: Option<StorageBody>,
}

#[derive(Deserialize)]
struct StorageBody {
    value: String,
}

#[derive(Deserialize)]
struct Label {
    name: String,
    prefix: Option<String>,
}

#[derive(Deserialize)]
struct ReadRestrictions {
    restrictions: Restrictions,
}

#[derive(Deserialize)]
struct Restrictions {
    #[serde(default)]
    user: RestrictionList<RestrictedUser>,
    #[serde(default)]
    group: RestrictionList<RestrictedGroup>,
}

#[derive(Deserialize)]
struct RestrictionList<T> {
    results: Vec<T>,
    /// The page size the server applied, which may be below the requested one.
    #[serde(default)]
    limit: usize,
}

impl<T> RestrictionList<T> {
    fn is_full(&self) -> bool {
        self.limit > 0 && self.results.len() >= self.limit
    }
}

impl<T> Default for RestrictionList<T> {
    fn default() -> Self {
        Self {
            results: Vec::new(),
            limit: 0,
        }
    }
}

#[derive(Deserialize)]
struct RestrictedUser {
    #[serde(rename = "accountId")]
    account_id: Option<String>,
}

#[derive(Deserialize)]
struct RestrictedGroup {
    name: String,
}

/// One chunk per non-empty heading section of a page body in storage format.
fn chunks(page_url: &str, storage: &str, authoritative: bool) -> anyhow::Result<Vec<ChunkDraft>> {
    // Locators are assigned before empty sections are dropped so they only depend on headings.
    let mut used = HashSet::from([INTRO.to_owned()]);
    let mut drafts = Vec::new();
    for section in sections(storage)? {
        let (locator, heading, url) = match section.path.last() {
            None => (INTRO.to_owned(), None, page_url.to_owned()),
            Some(title) => {
                let mut url = Url::parse(page_url)?;
                url.set_fragment(Some(&anchor(title)));
                (
                    unique(slug(title), &mut used),
                    Some(section.path.join(" > ")),
                    url.to_string(),
                )
            }
        };
        if section.text.is_empty() {
            continue;
        }
        drafts.push(ChunkDraft {
            locator,
            heading,
            url,
            text: section.text,
            authoritative,
        });
    }
    Ok(drafts)
}

/// Lowercase alphanumeric words joined by `-`.
fn slug(heading: &str) -> String {
    let mut slug = String::new();
    for c in heading.chars().flat_map(char::to_lowercase) {
        if c.is_alphanumeric() {
            slug.push(c);
        } else if !slug.is_empty() && !slug.ends_with('-') {
            slug.push('-');
        }
    }
    let slug = slug.trim_end_matches('-');
    if slug.is_empty() {
        "section".to_owned()
    } else {
        slug.to_owned()
    }
}

fn unique(slug: String, used: &mut HashSet<String>) -> String {
    let mut candidate = slug.clone();
    let mut n = 2;
    while !used.insert(candidate.clone()) {
        candidate = format!("{slug}-{n}");
        n += 1;
    }
    candidate
}

/// Confluence Cloud's heading anchor: the heading text with whitespace runs replaced by `-`.
fn anchor(heading: &str) -> String {
    heading.split_whitespace().collect::<Vec<_>>().join("-")
}

#[derive(Debug, PartialEq)]
struct Section {
    /// Headings from the outermost to this section's own; empty for the intro.
    path: Vec<String>,
    text: String,
}

/// Every section of the page, including empty ones; the first is always the intro.
fn sections(storage: &str) -> anyhow::Result<Vec<Section>> {
    let mut reader = Reader::from_str(storage);
    reader.config_mut().check_end_names = false;
    let mut converter = Converter::default();
    loop {
        let event = reader.read_event().with_context(|| {
            format!(
                "invalid storage format near byte {}",
                reader.buffer_position()
            )
        })?;
        match event {
            Event::Start(e) => converter.start(&e, false),
            Event::Empty(e) => {
                converter.start(&e, true);
                converter.end(e.name().as_ref());
            }
            Event::End(e) => converter.end(e.name().as_ref()),
            Event::Text(t) => converter.text(&t.xml10_content()),
            Event::CData(t) => converter.cdata(&t.xml10_content()),
            Event::GeneralRef(r) => converter.text(&resolve_reference(&r)),
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(converter.finish())
}

fn resolve_entity(name: &str) -> Option<&'static str> {
    resolve_predefined_entity(name).or_else(|| resolve_html5_entity(name))
}

fn resolve_reference(reference: &BytesRef<'_>) -> String {
    if let Ok(Some(c)) = reference.resolve_char_ref() {
        return c.to_string();
    }
    resolve_entity(reference)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("&{};", &**reference))
}

fn attribute(element: &BytesStart<'_>, name: &str) -> Option<String> {
    element
        .try_get_attribute(name)
        .ok()??
        .normalized_value_with(XmlVersion::Implicit1_0, 8, resolve_entity)
        .ok()
        .map(|v| v.into_owned())
}

/// Elements whose text is not page content.
const SKIPPED: &[&str] = &[
    "ac:parameter",
    "ac:task-id",
    "ac:task-uuid",
    "ac:task-status",
    "ac:placeholder",
    "ac:adf-attribute",
    "ac:adf-fallback",
    "style",
    "script",
];

const PARAGRAPHS: &[&str] = &[
    "p",
    "div",
    "blockquote",
    "hr",
    "ac:rich-text-body",
    "ac:layout",
    "ac:layout-section",
    "ac:layout-cell",
    "ac:adf-content",
    "ac:task-body",
];

/// Plain text with whitespace collapsed, built line by line.
#[derive(Default)]
struct Text {
    lines: Vec<String>,
    line: String,
    space: bool,
    /// Written before the current line (a list marker, then indentation).
    prefix: String,
    /// Becomes the prefix once a line is written.
    continuation: String,
}

impl Text {
    fn inline(&mut self, s: &str) {
        for c in s.chars() {
            if c.is_whitespace() {
                self.space = !self.line.is_empty();
            } else {
                if mem::take(&mut self.space) {
                    self.line.push(' ');
                }
                self.line.push(c);
            }
        }
    }

    fn end_line(&mut self) {
        if !self.line.is_empty() {
            self.lines
                .push(format!("{}{}", self.prefix, mem::take(&mut self.line)));
            self.prefix = self.continuation.clone();
        }
        self.space = false;
    }

    fn blank(&mut self) {
        self.end_line();
        if self.lines.last().is_some_and(|l| !l.is_empty()) {
            self.lines.push(String::new());
        }
    }

    fn raw(&mut self, line: &str) {
        self.end_line();
        self.lines.push(format!("{}{line}", self.continuation));
    }

    fn finish(mut self) -> String {
        self.end_line();
        self.lines.join("\n").trim().to_owned()
    }

    fn finish_inline(mut self) -> String {
        self.end_line();
        self.lines
            .iter()
            .filter(|l| !l.is_empty())
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(" ")
    }
}

struct List {
    ordered: bool,
    next: usize,
}

#[derive(Default)]
struct Link {
    fallback: Option<String>,
    has_body: bool,
}

struct Macro {
    language: Option<String>,
}

/// Walks storage-format events and splits the text at h1–h3 headings.
#[derive(Default)]
struct Converter {
    sections: Vec<Section>,
    path: Vec<(u8, String)>,
    body: Text,
    /// A section heading being read, with its level.
    heading: Option<(u8, Text)>,
    /// Open elements, innermost last.
    stack: Vec<String>,
    skip: usize,
    /// `ac:parameter` name and text being read.
    param: Option<(String, String)>,
    lists: Vec<List>,
    /// Continuation indent of each open list item.
    items: Vec<String>,
    table_depth: usize,
    row: Option<Vec<String>>,
    cell: Option<Text>,
    pre: Option<String>,
    links: Vec<Link>,
    macros: Vec<Macro>,
}

impl Converter {
    fn sink(&mut self) -> &mut Text {
        match (&mut self.heading, &mut self.cell) {
            (Some((_, heading)), _) => heading,
            (None, Some(cell)) => cell,
            (None, None) => &mut self.body,
        }
    }

    fn in_body(&self) -> bool {
        self.heading.is_none() && self.cell.is_none()
    }

    fn paragraph_break(&mut self) {
        if self.pre.is_some() {
            return;
        }
        if self.in_body() && self.items.is_empty() {
            self.body.blank();
        } else {
            self.sink().end_line();
        }
    }

    fn line_break(&mut self) {
        match &mut self.pre {
            Some(pre) => pre.push('\n'),
            None => self.sink().end_line(),
        }
    }

    fn text(&mut self, s: &str) {
        if let Some((_, value)) = &mut self.param {
            return value.push_str(s);
        }
        if self.skip > 0 {
            return;
        }
        match &mut self.pre {
            Some(pre) => pre.push_str(s),
            None => self.sink().inline(s),
        }
    }

    fn cdata(&mut self, s: &str) {
        if self.skip > 0 || self.pre.is_some() {
            return self.text(s);
        }
        if self.stack.last().is_some_and(|e| e == "ac:plain-text-body") {
            let language = self.macros.last().and_then(|m| m.language.clone());
            self.code_block(language.as_deref(), s);
        } else {
            self.text(s);
        }
    }

    fn code_block(&mut self, language: Option<&str>, code: &str) {
        let code = code.trim_matches(['\n', '\r']);
        if code.trim().is_empty() {
            return;
        }
        if !self.in_body() {
            return self.sink().inline(code);
        }
        self.paragraph_break();
        self.body
            .raw(&format!("```{}", language.unwrap_or_default()));
        for line in code.lines() {
            self.body.raw(line.trim_end());
        }
        self.body.raw("```");
        self.paragraph_break();
    }

    fn start(&mut self, element: &BytesStart<'_>, empty: bool) {
        let qname = element.name();
        let name: &str = qname.as_ref();
        self.stack.push(name.to_owned());
        if self.skip > 0 {
            self.skip += 1;
            return;
        }
        if SKIPPED.contains(&name) {
            self.skip = 1;
            if name == "ac:parameter" {
                let param = attribute(element, "ac:name").unwrap_or_default();
                self.param = Some((param, String::new()));
            }
            return;
        }
        if PARAGRAPHS.contains(&name) {
            return self.paragraph_break();
        }
        match name {
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                let level = name.as_bytes()[1] - b'0';
                let boundary = level <= 3
                    && self.table_depth == 0
                    && self.items.is_empty()
                    && self.in_body()
                    && self.pre.is_none();
                if boundary {
                    self.heading = Some((level, Text::default()));
                } else {
                    self.paragraph_break();
                }
            }
            "br" => self.line_break(),
            "ul" | "ol" | "ac:task-list" => {
                self.lists.push(List {
                    ordered: name == "ol",
                    next: 1,
                });
                self.sink().end_line();
            }
            "li" | "ac:task" => self.start_item(),
            "table" => {
                self.table_depth += 1;
                if self.table_depth == 1 {
                    self.paragraph_break();
                }
            }
            "tr" if self.table_depth == 1 => self.row = Some(Vec::new()),
            "td" | "th" if self.table_depth == 1 && self.row.is_some() => {
                self.cell = Some(Text::default());
            }
            "td" | "th" => self.sink().end_line(),
            "pre" if self.pre.is_none() && self.in_body() => {
                self.paragraph_break();
                self.pre = Some(String::new());
            }
            "ac:structured-macro" => self.macros.push(Macro { language: None }),
            "ac:link" => self.links.push(Link::default()),
            "ac:plain-text-link-body" | "ac:link-body" => {
                if let Some(link) = self.links.last_mut() {
                    link.has_body = true;
                }
            }
            "time" if empty => {
                if let Some(datetime) = attribute(element, "datetime") {
                    self.text(&datetime);
                }
            }
            _ if name.starts_with("ri:") => {
                if let Some(link) = self.links.last_mut()
                    && link.fallback.is_none()
                {
                    link.fallback = [
                        "ri:content-title",
                        "ri:filename",
                        "ri:space-key",
                        "ri:value",
                    ]
                    .iter()
                    .find_map(|a| attribute(element, a));
                }
            }
            _ => {}
        }
    }

    fn end(&mut self, name: &str) {
        self.stack.pop();
        if self.skip > 0 {
            self.skip -= 1;
            if self.skip == 0
                && let Some((param, value)) = self.param.take()
                && param == "language"
                && let Some(code_macro) = self.macros.last_mut()
            {
                code_macro.language = Some(value.trim().to_owned());
            }
            return;
        }
        if PARAGRAPHS.contains(&name) {
            return self.paragraph_break();
        }
        match name {
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                let level = name.as_bytes()[1] - b'0';
                if self.heading.as_ref().is_some_and(|(l, _)| *l == level) {
                    self.end_heading();
                } else {
                    self.paragraph_break();
                }
            }
            "ul" | "ol" | "ac:task-list" => {
                self.lists.pop();
                if self.lists.is_empty() {
                    self.paragraph_break();
                } else {
                    self.sink().end_line();
                }
            }
            "li" | "ac:task" => self.end_item(),
            "table" => {
                self.table_depth = self.table_depth.saturating_sub(1);
                if self.table_depth == 0 {
                    self.row = None;
                    self.cell = None;
                    self.paragraph_break();
                }
            }
            "tr" if self.table_depth == 1 => {
                if let Some(cells) = self.row.take()
                    && cells.iter().any(|c| !c.is_empty())
                {
                    self.body.end_line();
                    self.body.inline(&cells.join(" | "));
                    self.body.end_line();
                }
            }
            "td" | "th" if self.table_depth == 1 && self.cell.is_some() => {
                let cell = self.cell.take().unwrap_or_default().finish_inline();
                if let Some(row) = &mut self.row {
                    row.push(cell);
                }
            }
            "td" | "th" => self.sink().end_line(),
            "pre" => {
                if let Some(code) = self.pre.take() {
                    self.code_block(None, &code);
                }
            }
            "ac:structured-macro" => {
                self.macros.pop();
            }
            "ac:link" => {
                if let Some(link) = self.links.pop()
                    && !link.has_body
                    && let Some(fallback) = link.fallback
                {
                    self.text(&fallback);
                }
            }
            _ => {}
        }
    }

    fn start_item(&mut self) {
        if !self.in_body() {
            return self.sink().end_line();
        }
        self.body.end_line();
        let marker = match self.lists.last_mut() {
            Some(list) if list.ordered => {
                let n = list.next;
                list.next += 1;
                format!("{n}. ")
            }
            _ => "- ".to_owned(),
        };
        let indent = "  ".repeat(self.lists.len().saturating_sub(1));
        let continuation = format!("{indent}{}", " ".repeat(marker.len()));
        self.body.prefix = format!("{indent}{marker}");
        self.body.continuation = continuation.clone();
        self.items.push(continuation);
    }

    fn end_item(&mut self) {
        if !self.in_body() {
            return self.sink().end_line();
        }
        self.body.end_line();
        self.items.pop();
        let parent = self.items.last().cloned().unwrap_or_default();
        self.body.prefix = parent.clone();
        self.body.continuation = parent;
    }

    fn end_heading(&mut self) {
        let Some((level, heading)) = self.heading.take() else {
            return;
        };
        let title = heading.finish_inline();
        if title.is_empty() {
            return;
        }
        self.close_section();
        while self.path.last().is_some_and(|(l, _)| *l >= level) {
            self.path.pop();
        }
        self.path.push((level, title));
    }

    fn close_section(&mut self) {
        let text = mem::take(&mut self.body).finish();
        self.sections.push(Section {
            path: self.path.iter().map(|(_, title)| title.clone()).collect(),
            text,
        });
    }

    fn finish(mut self) -> Vec<Section> {
        if let Some((_, heading)) = self.heading.take() {
            self.body.inline(&heading.finish_inline());
        }
        self.close_section();
        self.sections
    }
}

#[cfg(test)]
mod tests;
