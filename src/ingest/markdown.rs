//! Ingest pre-audited markdown files from a directory: one document per file, one chunk per heading section.

use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::Url;

use crate::domain::{ChunkDraft, Document, Labels, Sensitivity, SourceKind};

use super::Source;

const INTRO: &str = "intro";

#[derive(Debug, clap::Args)]
pub struct MarkdownArgs {
    /// Root directory of Drew-audited markdown files.
    #[arg(long, env = "DREW_DOCS_DIR")]
    pub docs: Option<PathBuf>,
    /// Optional base URL for citations (e.g. a GitHub blob URL ending in `/`).
    #[arg(long, env = "DREW_DOCS_BASE_URL")]
    pub docs_base_url: Option<String>,
}

pub struct Connector {
    root: PathBuf,
    base_url: Option<String>,
}

impl Connector {
    pub fn from_args(args: &MarkdownArgs) -> anyhow::Result<Option<Self>> {
        let Some(root) = args.docs.clone() else {
            return Ok(None);
        };
        anyhow::ensure!(
            root.is_dir(),
            "docs path {} is not a directory",
            root.display()
        );
        if let Some(base) = &args.docs_base_url {
            Url::parse(base).map_err(|e| anyhow::anyhow!("invalid docs base URL: {e}"))?;
        }
        Ok(Some(Self {
            root,
            base_url: args.docs_base_url.clone(),
        }))
    }

    fn document(&self, path: &Path) -> anyhow::Result<Document> {
        let rel = path
            .strip_prefix(&self.root)
            .map_err(|_| anyhow::anyhow!("path outside docs root"))?;
        let external_id = rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        let text = fs::read_to_string(path)?;
        let title = document_title(&text, path);
        let updated_at = file_mtime(path)?;
        let file_url = self.file_url(&external_id);
        let chunks = chunks(&file_url, &text)?;
        Ok(Document {
            source: SourceKind::Markdown,
            external_id,
            title,
            url: file_url,
            updated_at,
            labels: Labels {
                acl: vec![],
                customer: None,
                sensitivity: Sensitivity::Internal,
            },
            chunks,
        })
    }

    fn file_url(&self, external_id: &str) -> String {
        match &self.base_url {
            Some(base) => format!("{}{external_id}", base.trim_end_matches('/')),
            None => external_id.to_owned(),
        }
    }

    fn collect_md_files(&self) -> anyhow::Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        collect_md_recursive(&self.root, &mut files)?;
        files.sort();
        Ok(files)
    }
}

fn collect_md_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        if path.is_dir() {
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            collect_md_recursive(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "md") {
            out.push(path);
        }
    }
    Ok(())
}

fn file_mtime(path: &Path) -> anyhow::Result<DateTime<Utc>> {
    let meta = fs::metadata(path)?;
    let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    Ok(DateTime::<Utc>::from(modified))
}

/// First `# ` heading in the file, else the file stem.
fn document_title(content: &str, path: &Path) -> String {
    for line in content.lines() {
        if let Some(title) = atx_heading(line)
            && title.level == 1
        {
            return title.text;
        }
    }
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Untitled".into())
}

struct AtxTitle {
    level: usize,
    text: String,
}

fn atx_heading(line: &str) -> Option<AtxTitle> {
    let trimmed = line.trim();
    let hashes = trimmed.chars().take_while(|&c| c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let text = trimmed[hashes..].trim().to_owned();
    if text.is_empty() {
        return None;
    }
    Some(AtxTitle {
        level: hashes,
        text,
    })
}

#[derive(Debug, PartialEq)]
struct Section {
    path: Vec<String>,
    text: String,
}

/// One chunk per non-empty heading section; headings inside fenced code blocks are ignored.
pub fn chunks(file_url: &str, content: &str) -> anyhow::Result<Vec<ChunkDraft>> {
    let body = strip_front_matter(content);
    let sections = sections(body)?;
    let mut used = HashSet::from([INTRO.to_owned()]);
    let mut drafts = Vec::new();
    for section in sections {
        if section.text.trim().is_empty() {
            continue;
        }
        let (locator, heading, url) = match section.path.last() {
            None => (INTRO.to_owned(), None, file_url.to_owned()),
            Some(title) => {
                let fragment = github_anchor(title);
                let url = if file_url.contains("://") {
                    format!("{file_url}#{fragment}")
                } else {
                    file_url.to_owned()
                };
                (
                    unique(slug(title), &mut used),
                    Some(section.path.join(" > ")),
                    url,
                )
            }
        };
        drafts.push(ChunkDraft {
            locator,
            heading,
            url,
            text: section.text.trim().to_owned(),
            authoritative: true,
        });
    }
    Ok(drafts)
}

fn strip_front_matter(content: &str) -> &str {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return content;
    }
    let after = trimmed.strip_prefix("---").unwrap_or(trimmed);
    if let Some(end) = after.find("\n---") {
        let rest = &after[end + 4..];
        rest.strip_prefix('\n').unwrap_or(rest)
    } else {
        content
    }
}

fn sections(content: &str) -> anyhow::Result<Vec<Section>> {
    let mut out = Vec::new();
    let mut path: Vec<String> = Vec::new();
    let mut current_text = String::new();
    let mut in_fence = false;
    let mut fence_marker: Option<String> = None;

    let mut flush = |path: &Vec<String>, text: &mut String| {
        out.push(Section {
            path: path.clone(),
            text: std::mem::take(text),
        });
    };

    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(marker) = fence_marker.as_ref() {
            if trimmed.starts_with(marker.as_str()) {
                in_fence = false;
                fence_marker = None;
            }
            current_text.push_str(line);
            current_text.push('\n');
            continue;
        }
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            let marker = if trimmed.starts_with("```") {
                "```"
            } else {
                "~~~"
            };
            in_fence = true;
            fence_marker = Some(marker.to_owned());
            current_text.push_str(line);
            current_text.push('\n');
            continue;
        }
        if !in_fence && let Some(h) = atx_heading(line) {
            flush(&path, &mut current_text);
            while path.len() >= h.level {
                path.pop();
            }
            path.push(h.text);
            continue;
        }
        current_text.push_str(line);
        current_text.push('\n');
    }
    flush(&path, &mut current_text);
    Ok(out)
}

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

/// GitHub-style heading anchor for fragment links.
pub fn github_anchor(heading: &str) -> String {
    let mut out = String::new();
    for c in heading.chars() {
        if c.is_whitespace() {
            if !out.ends_with('-') && !out.is_empty() {
                out.push('-');
            }
        } else if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if c == '-' || c == '_' {
            out.push(c);
        }
    }
    out.trim_matches('-').to_string()
}

#[async_trait]
impl Source for Connector {
    fn kind(&self) -> SourceKind {
        SourceKind::Markdown
    }

    async fn fetch_all(&self) -> anyhow::Result<Vec<Document>> {
        let paths = self.collect_md_files()?;
        paths.iter().map(|p| self.document(p)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_front_matter() {
        let md = "---\ntitle: x\n---\n\n# Title\n\nBody";
        let sections = sections(strip_front_matter(md)).unwrap();
        assert_eq!(sections.len(), 2);
        assert!(sections[1].path.contains(&"Title".to_owned()));
    }

    #[test]
    fn ignores_headings_in_fences() {
        let md = "# Real\n\n```\n# Fake\n```\n\nMore";
        let drafts = chunks("https://example.test/a.md", md).unwrap();
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].heading.as_deref(), Some("Real"));
        assert!(drafts[0].text.contains("# Fake"));
    }

    #[test]
    fn nested_headings() {
        let md = "# Page\n\nIntro.\n\n## Section\n\nDetail.";
        let drafts = chunks("https://example.test/doc.md", md).unwrap();
        assert_eq!(drafts.len(), 2);
        assert_eq!(drafts[0].heading.as_deref(), Some("Page"));
        assert!(drafts[0].text.contains("Intro."));
        assert_eq!(drafts[1].heading.as_deref(), Some("Page > Section"));
        assert!(drafts[1].url.contains("#section"));
    }

    #[test]
    fn github_anchor_slug() {
        assert_eq!(github_anchor("Hello World!"), "hello-world");
    }
}
