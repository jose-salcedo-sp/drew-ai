use std::time::Instant;

use chrono::{DateTime, Utc};
use clap::{CommandFactory, Parser};
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{basic_auth, method, path, path_regex, query_param, query_param_is_missing},
};

use super::*;

const DREW: &str = "557058:drew";
const ROLES: &str = include_str!("../../../tests/fixtures/confluence/roles.storage.xml");
const SPACE_PAGES_FIRST: &str =
    include_str!("../../../tests/fixtures/confluence/space_pages_first.json");
const READ_RESTRICTIONS: &str =
    include_str!("../../../tests/fixtures/confluence/read_restrictions.json");

fn args(base_url: &str) -> Args {
    Args {
        confluence_base_url: Some(base_url.to_owned()),
        confluence_space_key: Some("ARM".to_owned()),
        confluence_email: Some("drew-bot@example.com".to_owned()),
        confluence_api_token: Some("secret-token".to_owned()),
        drew_confluence_account_id: Some(DREW.to_owned()),
        confluence_cql: None,
    }
}

#[test]
fn the_filter_narrows_the_search_and_composes_with_since() {
    let narrowed = Args {
        confluence_cql: Some(format!(r#"contributor = "{DREW}""#)),
        ..args("https://example.atlassian.net")
    };
    let connector = Connector::from_args(&narrowed).unwrap().unwrap();
    assert_eq!(
        connector.search_cql(None),
        r#"space = "ARM" and type = page and (contributor = "557058:drew") order by lastmodified asc"#
    );
    assert_eq!(
        connector.search_cql(Some(time("2026-10-09T12:30:00Z"))),
        r#"space = "ARM" and type = page and (contributor = "557058:drew") and lastmodified >= "2026-10-08 12:30" order by lastmodified asc"#
    );
}

fn connector(server: &MockServer) -> Connector {
    Connector::from_args(&args(&server.uri()))
        .unwrap()
        .expect("configured")
}

fn time(s: &str) -> DateTime<Utc> {
    s.parse().unwrap()
}

fn page(id: &str, title: &str, author: &str, updated_at: &str, storage: &str) -> Value {
    json!({
        "id": id,
        "status": "current",
        "title": title,
        "spaceId": "98",
        "authorId": author,
        "version": { "number": 1, "createdAt": updated_at },
        "body": { "storage": { "representation": "storage", "value": storage } },
        "_links": { "webui": format!("/spaces/ARM/pages/{id}") }
    })
}

fn json_body(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_raw(body.to_owned(), "application/json")
}

async fn mount_space(server: &MockServer) {
    Mock::given(method("GET"))
        .and(path("/wiki/api/v2/spaces"))
        .and(query_param("keys", "ARM"))
        .and(basic_auth("drew-bot@example.com", "secret-token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [{ "id": "98", "key": "ARM", "name": "ARM", "type": "global" }],
            "_links": { "base": "https://example.atlassian.net/wiki" }
        })))
        .mount(server)
        .await;
}

async fn mount_space_pages(server: &MockServer, pages: Vec<Value>) {
    Mock::given(method("GET"))
        .and(path("/wiki/api/v2/spaces/98/pages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": pages,
            "_links": {}
        })))
        .mount(server)
        .await;
}

/// Pages without labels or read restrictions, unless a higher-priority mock matches.
async fn mount_unlabelled_unrestricted(server: &MockServer) {
    Mock::given(method("GET"))
        .and(path_regex(r"^/wiki/api/v2/pages/\d+/labels$"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [],
            "_links": {}
        })))
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path_regex(
            r"^/wiki/rest/api/content/\d+/restriction/byOperation/read$",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "operation": "read",
            "restrictions": {
                "user": { "results": [], "start": 0, "limit": 100, "size": 0 },
                "group": { "results": [], "start": 0, "limit": 100, "size": 0 }
            }
        })))
        .mount(server)
        .await;
}

fn internal_space_labels() -> Labels {
    Labels {
        acl: vec!["space:ARM".to_owned()],
        customer: None,
        sensitivity: Sensitivity::Internal,
    }
}

#[test]
fn unconfigured_is_none() {
    assert!(Connector::from_args(&Args::default()).unwrap().is_none());
    let blank = Args {
        confluence_base_url: Some(" ".to_owned()),
        ..Args::default()
    };
    assert!(Connector::from_args(&blank).unwrap().is_none());
}

#[test]
fn partial_config_is_an_error() {
    let partial = Args {
        confluence_api_token: None,
        drew_confluence_account_id: Some(String::new()),
        ..args("https://example.atlassian.net")
    };
    let error = Connector::from_args(&partial)
        .err()
        .expect("error")
        .to_string();
    assert!(error.contains("CONFLUENCE_API_TOKEN"), "{error}");
    assert!(error.contains("DREW_CONFLUENCE_ACCOUNT_ID"), "{error}");
    assert!(!error.contains("CONFLUENCE_SPACE_KEY"), "{error}");
}

#[test]
fn base_url_may_include_wiki() {
    let connector = Connector::from_args(&args("https://example.atlassian.net/wiki/"))
        .unwrap()
        .unwrap();
    assert_eq!(connector.site, "https://example.atlassian.net");
}

#[test]
fn args_parse_from_flags_and_hide_the_token() {
    mod sibling {
        #[derive(clap::Args)]
        pub struct Args {
            #[arg(long)]
            pub other: Option<String>,
        }
    }

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        confluence: Args,
        #[command(flatten)]
        sibling: sibling::Args,
    }
    Cli::command().debug_assert();
    let cli = Cli::parse_from([
        "drew",
        "--confluence-base-url=https://example.atlassian.net",
        "--confluence-space-key=ARM",
        "--confluence-email=drew-bot@example.com",
        "--confluence-api-token=secret-token",
        "--drew-confluence-account-id=557058:drew",
    ]);
    assert!(Connector::from_args(&cli.confluence).unwrap().is_some());
    assert!(!format!("{:?}", cli.confluence).contains("secret-token"));
}

#[test]
fn chunks_by_heading_section() {
    let url = "https://example.atlassian.net/wiki/spaces/ARM/pages/102";
    let chunk = |locator: &str, heading: Option<&str>, anchor: &str, text: &str| ChunkDraft {
        locator: locator.to_owned(),
        heading: heading.map(str::to_owned),
        url: format!("{url}{anchor}"),
        text: text.to_owned(),
        authoritative: true,
    };
    let expected = vec![
        chunk(
            "intro",
            None,
            "",
            "This page explains how ARM models SAP roles.",
        ),
        chunk(
            "roles",
            Some("Roles"),
            "#Roles",
            "A role bundles authorizations. Users are assigned through the AGR_USERS table.",
        ),
        chunk(
            "single-roles",
            Some("Roles > Single roles"),
            "#Single-roles",
            "- Hold authorization objects such as S_TCODE.\n\
             - Are assigned directly\n  \
             1. via PFCG\n  \
             2. via SU01",
        ),
        chunk(
            "composite-roles",
            Some("Roles > Composite roles"),
            "#Composite-roles",
            "A composite role contains single roles.\n\
             \n\
             ```sql\n\
             SELECT agr_name, uname\n  \
             FROM agr_users\n \
             WHERE agr_name LIKE 'Z%';\n\
             ```",
        ),
        chunk(
            "examples",
            Some("Roles > Composite roles > Examples"),
            "#Examples",
            "Role | Contains\nZ_FINANCE | Z_AP, Z_AR\nZ_EMPTY |",
        ),
        chunk(
            "mappings",
            Some("Mappings"),
            "#Mappings",
            "See Q&A and the FAQ for details.",
        ),
        chunk(
            "examples-2",
            Some("Mappings > Examples"),
            "#Examples",
            "Mapping examples live in Jira.\n\
             \n\
             Not a section\n\
             \n\
             Minor headings stay in their section.",
        ),
        chunk(
            "examples-3",
            Some("Mappings > Examples"),
            "#Examples",
            "Reviewed on 2026-09-30.",
        ),
    ];
    let actual = chunks(url, ROLES, true).unwrap();
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual, expected);
    }
    assert_eq!(actual.len(), expected.len());
}

#[test]
fn locators_ignore_whether_sections_are_empty() {
    let url = "https://example.atlassian.net/wiki/spaces/ARM/pages/1";
    let locators = |storage: &str| -> Vec<String> {
        chunks(url, storage, false)
            .unwrap()
            .into_iter()
            .map(|c| c.locator)
            .collect()
    };
    assert_eq!(
        locators("<h1>Intro</h1><p>a</p><h2>Notes</h2><h2>Notes</h2><p>b</p>"),
        ["intro-2", "notes-2"]
    );
    assert_eq!(
        locators("<p>lead</p><h1>Intro</h1><p>a</p><h2>Notes</h2><p>c</p><h2>Notes</h2><p>b</p>"),
        ["intro", "intro-2", "notes", "notes-2"]
    );
    assert!(locators("").is_empty());
    assert!(locators("<h1>Only a heading</h1><p>&nbsp;</p>").is_empty());
}

#[test]
fn heading_slugs_and_anchors() {
    assert_eq!(
        slug("  S_TCODE & AGR_USERS (2024) "),
        "s-tcode-agr-users-2024"
    );
    assert_eq!(slug("???"), "section");
    let url = chunks(
        "https://example.atlassian.net/wiki/spaces/ARM/pages/1",
        "<h2>What is S_TCODE?</h2><p>x</p>",
        false,
    )
    .unwrap()
    .remove(0)
    .url;
    assert_eq!(
        url,
        "https://example.atlassian.net/wiki/spaces/ARM/pages/1#What-is-S_TCODE?"
    );
}

#[test]
fn invalid_storage_is_an_error() {
    assert!(
        chunks(
            "https://example.atlassian.net/wiki/x",
            "<p>broken <b",
            false
        )
        .is_err()
    );
}

#[tokio::test]
async fn fetches_every_page_of_the_space_across_result_pages() {
    let server = MockServer::start().await;
    mount_space(&server).await;
    mount_unlabelled_unrestricted(&server).await;
    Mock::given(method("GET"))
        .and(path("/wiki/api/v2/spaces/98/pages"))
        .and(query_param("body-format", "storage"))
        .and(query_param_is_missing("cursor"))
        .respond_with(json_body(SPACE_PAGES_FIRST))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/wiki/api/v2/spaces/98/pages"))
        .and(query_param("cursor", "eyJpZCI6IjEwMSJ9"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [page("102", "Roles", "5b10ac8d82e05b22cc7d4ef5", "2026-10-03T08:00:00.000Z", ROLES)],
            "_links": { "base": "https://example.atlassian.net/wiki" }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let documents = connector(&server).fetch_changed(None).await.unwrap();

    let base = format!("{}/wiki/spaces/ARM/pages", server.uri());
    assert_eq!(documents.len(), 2);
    let first = &documents[0];
    assert_eq!(first.source, SourceKind::Confluence);
    assert_eq!(first.external_id, "101");
    assert_eq!(first.title, "AGR_USERS");
    assert_eq!(first.url, format!("{base}/101"));
    assert_eq!(first.updated_at, time("2026-10-02T15:04:05Z"));
    assert_eq!(first.labels, internal_space_labels());
    assert_eq!(
        first.chunks,
        vec![
            ChunkDraft {
                locator: "intro".to_owned(),
                heading: None,
                url: format!("{base}/101"),
                text: "AGR_USERS maps users to roles.".to_owned(),
                authoritative: true,
            },
            ChunkDraft {
                locator: "validity".to_owned(),
                heading: Some("Validity".to_owned()),
                url: format!("{base}/101#Validity"),
                text: "FROM_DAT and TO_DAT bound the assignment.".to_owned(),
                authoritative: true,
            },
        ]
    );
    let second = &documents[1];
    assert_eq!(second.external_id, "102");
    assert_eq!(second.updated_at, time("2026-10-03T08:00:00Z"));
    assert_eq!(second.chunks.len(), 8);
    assert!(second.chunks.iter().all(|c| !c.authoritative));
}

#[tokio::test]
async fn since_uses_cql_and_keeps_only_pages_changed_after_it() {
    let server = MockServer::start().await;
    mount_unlabelled_unrestricted(&server).await;
    Mock::given(path("/wiki/api/v2/spaces"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&server)
        .await;
    let cql = r#"space = "ARM" and type = page and lastmodified >= "2026-10-04 12:00" order by lastmodified asc"#;
    Mock::given(method("GET"))
        .and(path("/wiki/rest/api/content/search"))
        .and(query_param("cql", cql))
        .and(query_param_is_missing("cursor"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                { "id": "1", "type": "page", "status": "current", "title": "Before" },
                { "id": "2", "type": "page", "status": "current", "title": "Just after" }
            ],
            "start": 0,
            "limit": 2,
            "size": 2,
            "_links": {
                "base": "https://example.atlassian.net/wiki",
                "context": "/wiki",
                "next": "/rest/api/content/search?next=true&cursor=c2&limit=2"
            }
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/wiki/rest/api/content/search"))
        .and(query_param("cursor", "c2"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                { "id": "3", "type": "page", "status": "current", "title": "Later" },
                { "id": "2", "type": "page", "status": "current", "title": "Just after" }
            ],
            "start": 2,
            "limit": 2,
            "size": 2,
            "_links": {}
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/wiki/api/v2/pages"))
        .and(query_param("id", "1,2,3"))
        .and(query_param("body-format", "storage"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                page("1", "Before", DREW, "2026-10-05T11:59:59.000Z", "<p>old</p>"),
                page("2", "Just after", DREW, "2026-10-05T12:00:01.000Z", "<p>new</p>"),
                page("3", "Later", DREW, "2026-10-07T00:00:00.000Z", "<p>newer</p>"),
            ],
            "_links": {}
        })))
        .expect(1)
        .mount(&server)
        .await;

    let documents = connector(&server)
        .fetch_changed(Some(time("2026-10-05T12:00:00Z")))
        .await
        .unwrap();

    let ids: Vec<_> = documents.iter().map(|d| d.external_id.as_str()).collect();
    assert_eq!(ids, ["2", "3"]);
}

#[tokio::test]
async fn records_labels_and_read_restrictions() {
    let server = MockServer::start().await;
    mount_space(&server).await;
    mount_unlabelled_unrestricted(&server).await;
    mount_space_pages(
        &server,
        vec![
            page(
                "201",
                "Restricted",
                DREW,
                "2026-10-01T00:00:00.000Z",
                "<p>a</p>",
            ),
            page(
                "202",
                "Public",
                DREW,
                "2026-10-01T00:00:00.000Z",
                "<p>b</p>",
            ),
        ],
    )
    .await;
    Mock::given(method("GET"))
        .and(path("/wiki/api/v2/pages/201/labels"))
        .and(query_param_is_missing("cursor"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                { "id": "1", "name": "drew-public", "prefix": "global" },
                { "id": "2", "name": "customer-zeta", "prefix": "my" },
                { "id": "3", "name": "sap", "prefix": "global" }
            ],
            "_links": { "next": "/wiki/api/v2/pages/201/labels?limit=250&cursor=l2" }
        })))
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/wiki/api/v2/pages/201/labels"))
        .and(query_param("cursor", "l2"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                { "id": "4", "name": "drew-confidential", "prefix": "global" },
                { "id": "5", "name": "customer-acme", "prefix": "global" }
            ],
            "_links": {}
        })))
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(
            "/wiki/rest/api/content/201/restriction/byOperation/read",
        ))
        .respond_with(json_body(READ_RESTRICTIONS))
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/wiki/api/v2/pages/202/labels"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [{ "id": "1", "name": "drew-public", "prefix": "global" }],
            "_links": {}
        })))
        .with_priority(1)
        .mount(&server)
        .await;

    let documents = connector(&server).fetch_changed(None).await.unwrap();

    assert_eq!(
        documents[0].labels,
        Labels {
            acl: vec![
                "group:arm-maintainers".to_owned(),
                "user:557058:drew".to_owned(),
                "user:5b10ac8d82e05b22cc7d4ef5".to_owned(),
            ],
            customer: Some("acme".to_owned()),
            sensitivity: Sensitivity::Confidential,
        }
    );
    assert_eq!(
        documents[1].labels,
        Labels {
            sensitivity: Sensitivity::Public,
            ..internal_space_labels()
        }
    );
}

#[tokio::test]
async fn retries_after_429_honouring_retry_after() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/wiki/api/v2/spaces"))
        .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "1"))
        .up_to_n_times(1)
        .with_priority(1)
        .expect(1)
        .mount(&server)
        .await;
    mount_space(&server).await;
    mount_space_pages(&server, vec![]).await;

    let started = Instant::now();
    let documents = connector(&server).fetch_changed(None).await.unwrap();

    assert!(documents.is_empty());
    assert!(started.elapsed() >= Duration::from_secs(1));
}

#[tokio::test]
async fn gives_up_after_repeated_429s() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/wiki/api/v2/spaces"))
        .respond_with(ResponseTemplate::new(429).insert_header("Retry-After", "0"))
        .expect(u64::from(MAX_RETRIES) + 1)
        .mount(&server)
        .await;

    let error = connector(&server).fetch_changed(None).await.unwrap_err();

    assert!(error.to_string().contains("429"), "{error:#}");
}

#[tokio::test]
async fn missing_space_is_an_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/wiki/api/v2/spaces"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [],
            "_links": {}
        })))
        .mount(&server)
        .await;

    let error = connector(&server).fetch_changed(None).await.unwrap_err();

    assert!(error.to_string().contains("ARM"), "{error:#}");
}
