//! Keep `help_doc_chunks` in step with the guides compiled into the binary.
//!
//! Two independent halves, so text search works before the (slower) vectors
//! exist:
//!   - [`ensure_text_index`] — synchronous, cheap: hash the compiled-in
//!     corpus, and when it differs from the hash stored at the last build,
//!     replace every row + FTS entry in one transaction.
//!   - [`ensure_embeddings`] — async: fill missing sources in the frozen
//!     active generation through `embedding_runtime`. Runs on the AI
//!     queue from `prewarm_chat` (app) and before a turn in the CLI/evals.

use std::sync::Arc;

use crate::ai::provider::AIProvider;
use crate::db::help_docs::PREF_HELP_CORPUS_HASH;
use crate::db::Database;
use crate::models::error::Result;

use super::corpus::{corpus, corpus_hash};

/// Pure: does the stored index match the compiled-in corpus? A missing hash
/// or an empty table (a fresh DB, or a migration that recreated the tables)
/// both mean "rebuild".
pub fn needs_text_rebuild(stored_hash: Option<&str>, current_hash: &str, stored_rows: i64) -> bool {
    stored_rows == 0 || stored_hash != Some(current_hash)
}

/// Rebuild the text index when the corpus changed. Returns `true` when it
/// rebuilt. Cheap when nothing changed (one hash + one COUNT).
pub fn ensure_text_index(db: &Database) -> Result<bool> {
    let current = corpus_hash();
    let stored = db.get_preference(PREF_HELP_CORPUS_HASH)?;
    let rows = db.count_help_doc_chunks()?;
    if !needs_text_rebuild(stored.as_deref(), &current, rows) {
        return Ok(false);
    }
    let rows: Vec<(crate::models::HelpChunk, String)> = corpus().iter().map(|c| (c.clone(), fts_heading(c))).collect();
    let n = db.replace_help_doc_chunks(&rows)?;
    db.set_preference(PREF_HELP_CORPUS_HASH, &current)?;
    log(
        "info",
        format!("help index: rebuilt {n} guide sections from the bundled docs"),
    );
    Ok(true)
}

/// What the FTS `heading` column indexes: page title and heading, so a
/// query naming the feature hits the section.
fn fts_heading(c: &crate::models::HelpChunk) -> String {
    if c.heading == c.page_title {
        c.page_title.clone()
    } else {
        format!("{} › {}", c.page_title, c.heading)
    }
}

/// Fill missing help sources in the verified active generation. An unbuilt
/// index stays on text search without making provider requests.
pub async fn ensure_embeddings(db: &Arc<Database>, _provider: &dyn AIProvider) -> Result<u32> {
    super::super::embedding_runtime::update_active(
        db,
        crate::db::embedding_generations::Corpus::HelpDocs,
        None,
        usize::MAX,
    )
    .await
}

/// Text index first (synchronous), then vectors. The one call the app's
/// prewarm and the CLI make before a chat turn.
pub async fn ensure_index(db: &Arc<Database>, provider: &dyn AIProvider) -> Result<()> {
    ensure_text_index(db)?;
    ensure_embeddings(db, provider).await?;
    Ok(())
}

fn log(level: &str, message: impl Into<String>) {
    crate::services::logger::log(level, "ai", message);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuild_when_hash_missing_or_stale_or_table_empty() {
        assert!(needs_text_rebuild(None, "abc", 10));
        assert!(needs_text_rebuild(Some("old"), "abc", 10));
        assert!(needs_text_rebuild(Some("abc"), "abc", 0));
        assert!(!needs_text_rebuild(Some("abc"), "abc", 10));
    }

    #[test]
    fn ensure_text_index_builds_once_then_is_a_no_op() {
        let db = Database::new_for_testing().unwrap();
        assert!(ensure_text_index(&db).unwrap());
        assert_eq!(db.count_help_doc_chunks().unwrap() as usize, corpus().len());
        assert!(!ensure_text_index(&db).unwrap());
        assert_eq!(
            db.get_preference(PREF_HELP_CORPUS_HASH).unwrap().as_deref(),
            Some(corpus_hash().as_str())
        );
    }

    #[test]
    fn ensure_text_index_rebuilds_after_a_stale_hash() {
        let db = Database::new_for_testing().unwrap();
        ensure_text_index(&db).unwrap();
        db.set_preference(PREF_HELP_CORPUS_HASH, "stale").unwrap();
        assert!(ensure_text_index(&db).unwrap());
        assert_eq!(db.count_help_doc_chunks().unwrap() as usize, corpus().len());
    }

    #[test]
    fn text_index_finds_sections_by_feature_name_in_every_language() {
        let db = Database::new_for_testing().unwrap();
        ensure_text_index(&db).unwrap();
        for q in ["Ollama", "OpenRouter", "calendario", "Kalender", "calendrier"] {
            assert!(
                !db.fts_search_help_docs(q, 5, None).unwrap().is_empty(),
                "no FTS hit for {q}"
            );
        }
    }

    #[tokio::test]
    async fn ensure_embeddings_asks_nothing_of_a_provider_without_an_embedding_model() {
        use crate::ai::provider::FakeAiProvider;
        let db = Arc::new(Database::new_for_testing().unwrap());
        ensure_text_index(&db).unwrap();
        let provider = FakeAiProvider::default().without_embedding_model();

        assert_eq!(ensure_embeddings(&db, &provider).await.unwrap(), 0);

        assert!(provider.embed_calls().is_empty());
    }

    #[tokio::test]
    async fn unverified_legacy_help_never_sends_inference() {
        use crate::ai::provider::FakeAiProvider;
        let db = Arc::new(Database::new_for_testing().unwrap());
        ensure_text_index(&db).unwrap();
        let provider = FakeAiProvider::default().with_embedding_dim(4096);
        assert_eq!(ensure_embeddings(&db, &provider).await.unwrap(), 0);
        assert!(provider.embed_calls().is_empty());
    }
}
