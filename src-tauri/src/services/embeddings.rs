use std::sync::Arc;

use crate::services::app_handle::AppHandle;
use serde::{Deserialize, Serialize};

use crate::db::Database;
use crate::models::error::Result;
use crate::services::task_queue;

pub(crate) const DEFAULT_EMBEDDING_MODEL: &str = "nomic-embed-text";

/// User-tunable config for which emails get embedded (used for AI Search /
/// chat retrieval). Persisted in `user_preferences` like the other AI configs.
///
/// Categories default to `["primary"]` to avoid wasting model time on
/// promotions / newsletters which rarely show up in chat retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingsConfig {
    /// Gmail categories eligible for embedding. Empty = all categories.
    pub categories: Vec<String>,
}

impl Default for EmbeddingsConfig {
    fn default() -> Self {
        Self {
            categories: vec!["primary".to_string()],
        }
    }
}

impl EmbeddingsConfig {
    /// Returns true when `category` is allowed. Empty `categories` means allow all.
    pub fn is_category_allowed(&self, category: &str) -> bool {
        self.categories.is_empty() || self.categories.iter().any(|c| c.eq_ignore_ascii_case(category))
    }
}

fn embeddings_config_key(account_id: &str) -> String {
    format!("embeddings_categories:{}", account_id)
}

pub fn get_embeddings_config(db: &Database, account_id: &str) -> Result<EmbeddingsConfig> {
    let defaults = EmbeddingsConfig::default();
    let categories = db
        .get_preference(&embeddings_config_key(account_id))?
        .and_then(|v| serde_json::from_str(&v).ok())
        .unwrap_or(defaults.categories);
    Ok(EmbeddingsConfig { categories })
}

pub fn save_embeddings_config(db: &Database, account_id: &str, config: &EmbeddingsConfig) -> Result<()> {
    db.set_preference(
        &embeddings_config_key(account_id),
        &serde_json::to_string(&config.categories)?,
    )?;
    Ok(())
}

/// Chunk size in characters for splitting long emails
const CHUNK_SIZE: usize = 1000;
/// Overlap between consecutive chunks
const CHUNK_OVERLAP: usize = 200;
/// Only chunk emails longer than this threshold
const CHUNK_THRESHOLD: usize = 1500;
/// Hard cap on chunks per email (avoids runaway on huge HTML bodies)
const MAX_CHUNKS_PER_EMAIL: usize = 6;
/// Maximum characters to consider from body (avoids processing entire mega-emails)
const MAX_BODY_CHARS: usize = 8000;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingProgress {
    pub status: String,
    pub current: u32,
    pub total: u32,
    pub message: String,
}

/// Tell the progress indicators that the run ended because the user stopped
/// it. Goes through the events seam, so it needs no `AppHandle`.
fn emit_cancelled(current: u32, total: u32, message: &str) {
    crate::services::events::emit(
        "embedding-progress",
        EmbeddingProgress {
            status: "cancelled".to_string(),
            current,
            total,
            message: message.to_string(),
        },
    );
}

/// Strip Re:/Fwd:/Fw: prefixes from subject to get clean thread topic
fn strip_reply_prefixes(subject: &str) -> String {
    let mut s = subject.trim().to_string();
    loop {
        let trimmed = s.trim_start();
        let lower = trimmed.to_lowercase();
        if lower.starts_with("re:") || lower.starts_with("fwd:") || lower.starts_with("fw:") {
            if let Some(colon_pos) = trimmed.find(':') {
                s = trimmed[colon_pos + 1..].trim().to_string();
            } else {
                break;
            }
        } else {
            break;
        }
    }
    s
}

/// Clean body text — strip HTML, collapse whitespace, strip empty lines, cap length
fn clean_body(body: &str) -> String {
    // Detect and strip HTML
    let stripped = if body.contains("<html") || body.contains("<body") || body.contains("<!DOCTYPE") {
        crate::util::html::strip_html_tags(body)
    } else {
        body.to_string()
    };

    let cleaned: String = stripped
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    // Cap length early so chunking doesn't explode on huge emails
    if cleaned.len() > MAX_BODY_CHARS {
        let boundary = floor_char_boundary(&cleaned, MAX_BODY_CHARS);
        cleaned[..boundary].to_string()
    } else {
        cleaned
    }
}

/// Snap a byte index to the nearest valid UTF-8 char boundary at or below `pos`
fn floor_char_boundary(s: &str, mut pos: usize) -> usize {
    if pos >= s.len() {
        return s.len();
    }
    while pos > 0 && !s.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

/// Snap a byte index to the nearest valid UTF-8 char boundary at or above `pos`
fn ceil_char_boundary(s: &str, mut pos: usize) -> usize {
    if pos >= s.len() {
        return s.len();
    }
    while pos < s.len() && !s.is_char_boundary(pos) {
        pos += 1;
    }
    pos
}

/// Safely slice a &str using byte indices, snapping both to char boundaries.
fn safe_slice(s: &str, start: usize, end: usize) -> &str {
    let start = floor_char_boundary(s, start);
    let end = floor_char_boundary(s, end);
    if start > end {
        return "";
    }
    &s[start..end]
}

/// Split long text into overlapping chunks for embedding.
/// Short texts (< CHUNK_THRESHOLD) are returned as a single chunk.
/// Uses char-by-char iteration to avoid UTF-8 boundary issues.
fn chunk_text(body: &str) -> Vec<String> {
    let clean = clean_body(body);

    if clean.len() <= CHUNK_THRESHOLD {
        return vec![clean];
    }

    // Collect all char boundaries for O(1) boundary-safe slicing
    let char_positions: Vec<usize> = clean
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(clean.len()))
        .collect();
    let total_chars = char_positions.len() - 1;

    // Convert char counts to boundary-safe byte positions
    let chunk_size_chars = CHUNK_SIZE; // treat as char count approximation
    let overlap_chars = CHUNK_OVERLAP;

    let mut chunks = Vec::new();
    let mut start_char = 0;

    while start_char < total_chars {
        let end_char = (start_char + chunk_size_chars).min(total_chars);
        let start_byte = char_positions[start_char];
        let end_byte = char_positions[end_char];

        // Try to break at sentence boundary near the end
        let actual_end_byte = if end_char < total_chars {
            let search_start_byte = char_positions[end_char.saturating_sub(100)];
            let slice = &clean[search_start_byte..end_byte];
            if let Some(pos) = slice.rfind(['.', '!', '?']) {
                let boundary = search_start_byte + pos + 1;
                if boundary > char_positions[start_char + chunk_size_chars / 2] {
                    floor_char_boundary(&clean, boundary)
                } else {
                    // Word boundary fallback
                    clean[start_byte..end_byte]
                        .rfind(char::is_whitespace)
                        .map(|p| start_byte + p + 1)
                        .unwrap_or(end_byte)
                }
            } else {
                clean[start_byte..end_byte]
                    .rfind(char::is_whitespace)
                    .map(|p| start_byte + p + 1)
                    .unwrap_or(end_byte)
            }
        } else {
            end_byte
        };
        let actual_end_byte = ceil_char_boundary(&clean, actual_end_byte);
        let actual_end_byte = actual_end_byte.max(start_byte + 1);
        let actual_end_byte = actual_end_byte.min(clean.len());
        let actual_end_byte = floor_char_boundary(&clean, actual_end_byte);

        let chunk = safe_slice(&clean, start_byte, actual_end_byte).trim().to_string();
        if !chunk.is_empty() {
            chunks.push(chunk);
        }

        if actual_end_byte >= clean.len() {
            break;
        }

        // Find next start_char position — walk back ~overlap_chars
        let current_end_char = char_positions
            .iter()
            .position(|&p| p >= actual_end_byte)
            .unwrap_or(total_chars);
        let next_start_char = current_end_char.saturating_sub(overlap_chars);
        if next_start_char <= start_char {
            start_char += 1; // guarantee progress
        } else {
            start_char = next_start_char;
        }
    }

    // Drop tiny trailing chunks
    chunks.retain(|c| c.len() >= 100);

    // Cap chunk count
    if chunks.len() > MAX_CHUNKS_PER_EMAIL {
        chunks.truncate(MAX_CHUNKS_PER_EMAIL);
    }

    if chunks.is_empty() {
        vec![clean]
    } else {
        chunks
    }
}

/// Create embedding texts for an email — one per chunk.
/// Each chunk includes the subject/sender header for full context.
pub(crate) fn create_embedding_texts(subject: &str, sender: &str, body: &str) -> Vec<String> {
    let clean_subject = strip_reply_prefixes(subject);
    let header = format!("Subject: {}\nFrom: {}\nContent: ", clean_subject, sender);

    let chunks = chunk_text(body);

    chunks.into_iter().map(|chunk| format!("{}{}", header, chunk)).collect()
}

/// Generate embeddings for emails that don't have them yet (single batch).
///
/// `account_label` is included in user-facing log/progress messages so the
/// output panel makes it clear which mailbox is being processed. Pass the
/// account email/name when known.
pub async fn generate_embeddings(
    db: &Arc<Database>,
    account_id: Option<&str>,
    app: Option<AppHandle>,
    batch_size: i32,
    account_label: Option<&str>,
) -> Result<u32> {
    generate_embeddings_inner(db, account_id, app, batch_size, account_label, true).await
}

/// Ordinary sync updates only the frozen active generation. An unverified
/// legacy installation uses FTS until the user builds a managed replacement.
async fn generate_embeddings_inner(
    db: &Arc<Database>,
    account_id: Option<&str>,
    _app: Option<AppHandle>,
    batch_size: i32,
    _account_label: Option<&str>,
    _emit_lifecycle: bool,
) -> Result<u32> {
    if !db.is_ai_enabled()? || task_queue::cancel_requested() {
        return Ok(0);
    }
    super::embedding_runtime::update_active(
        db,
        crate::db::embedding_generations::Corpus::Emails,
        account_id,
        batch_size.max(0) as usize,
    )
    .await
}

pub fn count_pending_embeddings(db: &Arc<Database>, account_id: Option<&str>) -> Result<i32> {
    let Some(active) = db.active_embedding_write_lease()? else {
        return db.count_emails_without_embeddings(account_id);
    };
    Ok(db
        .embedding_pending_count(
            active.generation_id,
            crate::db::embedding_generations::Corpus::Emails,
            account_id,
        )?
        .min(i32::MAX as i64) as i32)
}

/// Replacement builds preserve the active index. All corpora move together;
/// scoped refreshes use incremental indexing rather than deleting shared data.
pub async fn regenerate_embeddings(
    db: &Arc<Database>,
    account_id: Option<&str>,
    _app: Option<AppHandle>,
    _batch_size: i32,
    _account_label: Option<&str>,
) -> Result<u32> {
    if !db.is_ai_enabled()? {
        return Err(crate::models::error::AppError::AiDisabled);
    }
    if task_queue::cancel_requested() {
        emit_cancelled(0, 0, "Search index rebuild stopped before it started");
        return Ok(0);
    }
    let active = db.active_embedding_write_lease()?.ok_or_else(|| {
        crate::models::error::AppError::InvalidInput(
            "Build and verify the initial embedding index in AI Settings before requesting an account rebuild".into(),
        )
    })?;
    let profile = db.embedding_generation_profile(active.generation_id)?;
    let mut corpora = vec![crate::db::embedding_generations::Corpus::Emails];
    if db.is_memory_enabled()? {
        corpora.push(crate::db::embedding_generations::Corpus::MemoryFacts);
    }
    if db.is_help_docs_enabled()? {
        corpora.push(crate::db::embedding_generations::Corpus::HelpDocs);
    }
    let preview = super::embedding_runtime::preview_build(db, &profile, &corpora).await?;
    let lease = super::embedding_runtime::commit_build(db, &preview).await?;
    super::embedding_runtime::run_build_scoped(db, &lease, account_id).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_prefixes() {
        assert_eq!(strip_reply_prefixes("Re: Hello"), "Hello");
        assert_eq!(strip_reply_prefixes("Fwd: Re: Hello"), "Hello");
        assert_eq!(strip_reply_prefixes("FW: RE: Hello"), "Hello");
        assert_eq!(strip_reply_prefixes("Hello"), "Hello");
        assert_eq!(strip_reply_prefixes("Re: Fwd: Test"), "Test");
    }

    #[test]
    fn chunking_short_text() {
        let chunks = chunk_text("Short email body");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Short email body");
    }

    #[test]
    fn chunking_long_text() {
        let long = "word ".repeat(500); // 2500 chars
        let chunks = chunk_text(&long);
        assert!(chunks.len() > 1, "Expected multiple chunks, got {}", chunks.len());
        // Each chunk should be roughly CHUNK_SIZE
        for chunk in &chunks {
            assert!(
                chunk.len() <= CHUNK_SIZE + 200,
                "Chunk too large: {} chars",
                chunk.len()
            );
        }
    }

    #[test]
    fn embedding_texts_include_header() {
        let texts = create_embedding_texts("Re: Invoice", "billing@example.com", "Short body");
        assert_eq!(texts.len(), 1);
        assert!(texts[0].contains("Subject: Invoice")); // Re: stripped
        assert!(texts[0].contains("From: billing@example.com"));
        assert!(texts[0].contains("Short body"));
    }

    #[test]
    fn chunking_utf8_safety() {
        // Long Spanish text with multi-byte chars that would panic naive slicing
        let text = "Información de ofertas de empleo para ingenieros de software. ".repeat(50);
        let chunks = chunk_text(&text);
        assert!(!chunks.is_empty());
        // Each chunk should be valid UTF-8 (if it wasn't, the format! would panic)
        for chunk in &chunks {
            let _ = format!("chunk: {}", chunk);
        }
    }

    #[test]
    fn chunking_html_stripping() {
        let html = r#"<!DOCTYPE html><html><body><p>Hello <b>world</b> with áéí</p></body></html>"#;
        let chunks = chunk_text(html);
        assert_eq!(chunks.len(), 1);
        // HTML tags should be stripped
        assert!(!chunks[0].contains("<p>"));
        assert!(chunks[0].contains("Hello"));
    }

    #[test]
    fn chunking_respects_max_chunks() {
        let huge = "word ".repeat(5000); // 25000 chars
        let chunks = chunk_text(&huge);
        assert!(chunks.len() <= MAX_CHUNKS_PER_EMAIL);
    }

    fn seed_emails(db: &Database, count: usize) {
        let conn = db.connection();
        conn.execute(
            "INSERT INTO accounts (id, provider, email, name, created_at)
             VALUES ('acc1', 'gmail', 'owner@example.com', 'Test', 0)",
            [],
        )
        .unwrap();
        for i in 0..count {
            conn.execute(
                "INSERT INTO emails
                     (id, account_id, thread_id, subject, sender, sender_email, sender_domain,
                      recipients_json, cc_json, snippet, timestamp, is_read, category, mailbox, created_at)
                 VALUES (?1, 'acc1', ?1, 'Quarterly report', 'Test Sender', 'sender@example.com',
                         'example.com', '[]', '[]', ?2, ?3, 0, 'primary', 'inbox', 0)",
                rusqlite::params![format!("e-{i}"), format!("Short body {i}"), 100 + i as i64],
            )
            .unwrap();
        }
    }

    fn embedded_emails(db: &Database) -> i64 {
        db.reader()
            .query_row("SELECT COUNT(DISTINCT email_id) FROM embedding_chunks", [], |r| {
                r.get(0)
            })
            .unwrap()
    }

    fn last_embedding_status(events: &crate::services::events::VecEventSink) -> Option<String> {
        let payloads = events.payloads_for("embedding-progress");
        payloads.last().and_then(|p| p["status"].as_str().map(str::to_string))
    }

    fn test_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("test runtime")
    }

    /// A rebuild deletes the whole index first: one that was stopped while
    /// still queued must not delete anything.
    #[test]
    fn a_rebuild_cancelled_while_queued_keeps_the_index() {
        use crate::services::task_queue::TaskQueue;

        let _seam = crate::services::events::seam_test_lock();
        let events = crate::services::events::install_for_testing();
        let db = Arc::new(Database::new_for_testing().expect("test db"));
        seed_emails(&db, 1);
        db.store_embedding_chunks("e-0", "acc1", &[vec![0.1_f32; 768]], "test-model", "hash")
            .unwrap();
        let queue = TaskQueue::new(1, "test_rebuild_cancel");

        let rebuilt = test_runtime().block_on(async {
            let (gate_tx, gate_rx) = tokio::sync::oneshot::channel::<()>();
            queue
                .submit_named("blocker", async move {
                    let _ = gate_rx.await;
                })
                .await;
            let (tx, rx) = tokio::sync::oneshot::channel();
            let db = db.clone();
            queue
                .submit_named("embeddings:rebuild:all", async move {
                    let _ = tx.send(regenerate_embeddings(&db, None, None, 500, None).await);
                })
                .await;
            queue.cancel_matching(|name| name.starts_with("embeddings:"));
            let _ = gate_tx.send(());
            rx.await.expect("task finished").expect("rebuild")
        });

        assert_eq!(rebuilt, 0);
        assert_eq!(embedded_emails(&db), 1, "the index must survive");
        assert_eq!(last_embedding_status(&events).as_deref(), Some("cancelled"));
        crate::services::events::install(Arc::new(crate::services::events::NoopEventSink));
    }
}
