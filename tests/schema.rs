#![allow(
    clippy::unwrap_used,
    reason = "integration-test helpers; clippy exempts only #[test] functions"
)]

use drew::llm::EMBEDDING_DIM;
use pgvector::Vector;
use sqlx::PgPool;

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn a_chunk_has_one_current_version(pool: PgPool) -> sqlx::Result<()> {
    sqlx::query(
        "INSERT INTO documents (id, source, external_id, title, url, source_updated_at)
         VALUES ('confluence:1', 'confluence', '1', 'Roles', 'https://example.test/1', now())",
    )
    .execute(&pool)
    .await?;

    let insert = |version: &'static str| {
        sqlx::query(
            "INSERT INTO chunks (id, version, document_id, ordinal, url, text, authoritative, acl,
                                 sensitivity, embedding, embedding_model, embedding_model_version)
             VALUES ('confluence:1#intro', $1, 'confluence:1', 0, 'https://example.test/1', 'text',
                     true, '{}', 'internal', $2, 'fake', '1')",
        )
        .bind(version)
        .bind(Vector::from(vec![0.0; EMBEDDING_DIM]))
    };

    insert("v1").execute(&pool).await?;
    assert!(insert("v2").execute(&pool).await.is_err());

    sqlx::query("UPDATE chunks SET superseded_at = now() WHERE version = 'v1'")
        .execute(&pool)
        .await?;
    insert("v2").execute(&pool).await?;
    Ok(())
}
