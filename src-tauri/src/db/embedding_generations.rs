//! Durable, provider-independent embedding index lifecycle.
//!
//! Stage 3 owns provider probing, immutable inference sessions, permission checks,
//! formatted inputs and typed retrieval. Legacy writers deliberately keep their
//! fixed tables; they cannot populate or activate these verified generations.
//! Every vec0 mutation uses a top-level transaction, never a nested savepoint:
//! sqlite-vec 0.1.9 implements transactions but has no savepoint callbacks.

use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::Database;
use crate::models::error::{AppError, Result};

mod runtime;
pub use runtime::{
    EmbeddingSourceSnapshot, GenerationCorpusStatus, GenerationVectorHit, SourceContent, TypedQueryEmbedding,
};

const LEASE_SECONDS: i64 = 60;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "ts",
    derive(ts_rs::TS),
    ts(export, export_to = "../../src/types/generated/")
)]
pub enum DimensionPolicy {
    Native,
    Explicit(usize),
}

/// Immutable, credential-free identity. Formatting versions must change whenever
/// the exact query/document transformation changes. A model name is not proof of
/// revision identity; `model_revision == None` remains visibly opaque to callers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "ts",
    derive(ts_rs::TS),
    ts(export, export_to = "../../src/types/generated/")
)]
pub struct EmbeddingProfile {
    pub version: u32,
    pub backend: String,
    pub endpoint: Option<String>,
    pub model: String,
    pub model_revision: Option<String>,
    pub dimension_policy: DimensionPolicy,
    pub dimensions: usize,
    pub query_format: String,
    pub document_format: String,
    pub preprocessing: String,
    pub normalization: String,
    pub encoding: String,
    pub distance_metric: String,
}

impl EmbeddingProfile {
    /// Normalize identity before persisting it. Endpoints cannot contain auth,
    /// query parameters or fragments: credentials never enter the profile/hash.
    pub fn canonicalized(&self) -> Result<Self> {
        let mut profile = self.clone();
        if self.version != 1 || !(1..=8192).contains(&self.dimensions) {
            return invalid("Unsupported embedding profile version or dimensions");
        }
        if let DimensionPolicy::Explicit(n) = self.dimension_policy {
            if n != self.dimensions {
                return invalid("Explicit embedding dimensions differ from detected dimensions");
            }
        }
        for value in [
            &self.backend,
            &self.model,
            &self.query_format,
            &self.document_format,
            &self.preprocessing,
            &self.normalization,
        ] {
            if value.trim().is_empty() {
                return invalid("Embedding identity fields must not be empty");
            }
        }
        if self.encoding != "float32-le" || self.distance_metric != "cosine" {
            return invalid("Only float32-le cosine indexes are supported");
        }
        if let Some(endpoint) = &self.endpoint {
            let mut url =
                url::Url::parse(endpoint).map_err(|_| AppError::InvalidInput("Invalid embedding endpoint".into()))?;
            if !matches!(url.scheme(), "http" | "https")
                || url.host_str().is_none()
                || !url.username().is_empty()
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some()
            {
                return invalid("Embedding endpoint must be HTTP(S) without credentials, query or fragment");
            }
            let path = url.path().trim_end_matches('/').to_string();
            url.set_path(&path);
            profile.endpoint = Some(url.to_string().trim_end_matches('/').to_string());
        }
        Ok(profile)
    }

    pub fn fingerprint(&self) -> Result<String> {
        let canonical = self.canonicalized()?;
        Ok(input_hash(&serde_json::to_vec(&canonical)?))
    }
}

/// SHA-256 of the complete exact input, without truncation or model-name mixing.
pub fn input_hash(input: &[u8]) -> String {
    hex::encode(Sha256::digest(input))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "ts",
    derive(ts_rs::TS),
    ts(export, export_to = "../../src/types/generated/")
)]
pub enum Corpus {
    Emails,
    MemoryFacts,
    HelpDocs,
}

impl Corpus {
    pub const ALL: [Self; 3] = [Self::Emails, Self::MemoryFacts, Self::HelpDocs];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Emails => "emails",
            Self::MemoryFacts => "memory_facts",
            Self::HelpDocs => "help_docs",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRevision {
    pub corpus: Corpus,
    pub source_id: String,
    pub account_id: Option<String>,
    pub revision: i64,
}

/// Opaque ownership proof; each resume gets a new epoch and random owner token.
#[derive(Clone, Debug)]
pub struct BuildLease {
    pub generation_id: i64,
    pub epoch: i64,
    pub owner: String,
    pub fingerprint: String,
}

/// Frozen active-generation write identity. It grants no provider permission;
/// Stage 3 must pair it with the immutable profile and current authorization.
#[derive(Clone, Debug)]
pub struct ActiveGenerationLease {
    pub config_revision: i64,
    pub generation_id: i64,
    pub epoch: i64,
    pub fingerprint: String,
}

enum WriteTarget<'a> {
    Build(&'a BuildLease),
    Active(&'a ActiveGenerationLease),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationControl {
    pub revision: i64,
    pub active_id: Option<i64>,
    pub pending_id: Option<i64>,
    pub retained_id: Option<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(
    feature = "ts",
    derive(ts_rs::TS),
    ts(export, export_to = "../../src/types/generated/")
)]
pub struct GenerationStatus {
    #[cfg_attr(feature = "ts", ts(type = "number"))]
    pub id: i64,
    pub fingerprint: Option<String>,
    pub dimensions: usize,
    pub state: String,
    #[cfg_attr(feature = "ts", ts(type = "number"))]
    pub epoch: i64,
    #[cfg_attr(feature = "ts", ts(type = "number"))]
    pub completed: i64,
    #[cfg_attr(feature = "ts", ts(type = "number"))]
    pub failed: i64,
    pub error: Option<String>,
    pub validation: Option<String>,
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(AppError::InvalidInput(message.into()))
}

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

fn vector_table(generation: i64, corpus: Corpus) -> Result<String> {
    if generation <= 0 {
        return invalid("Legacy generation has no verified vector table");
    }
    Ok(format!("vec_{}_g{generation}", corpus.as_str()))
}

fn control(conn: &Connection) -> Result<GenerationControl> {
    Ok(conn.query_row(
        "SELECT revision,active_id,pending_id,retained_id FROM embedding_control WHERE singleton=1",
        [],
        |r| {
            Ok(GenerationControl {
                revision: r.get(0)?,
                active_id: r.get(1)?,
                pending_id: r.get(2)?,
                retained_id: r.get(3)?,
            })
        },
    )?)
}

/// Lifecycle metadata changes do not change source selection. Preserve an
/// already-current active scope while fencing in-flight work at the new revision.
/// A scope stale from real settings changes deliberately stays stale.
fn carry_active_scope(conn: &Connection, previous_revision: i64) -> Result<()> {
    conn.execute("UPDATE embedding_generations SET scope_revision=(SELECT revision FROM embedding_control) WHERE id=(SELECT active_id FROM embedding_control) AND scope_revision=?1",[previous_revision])?;
    Ok(())
}

fn check_lease(conn: &Connection, lease: &BuildLease) -> Result<usize> {
    check_lease_at(conn, lease, now())
}

fn check_lease_at(conn: &Connection, lease: &BuildLease, timestamp: i64) -> Result<usize> {
    let dimensions: Option<usize> = conn
        .query_row(
            "SELECT g.dimensions FROM embedding_generations g
             JOIN embedding_control c ON c.pending_id=g.id
             WHERE g.id=?1 AND g.epoch=?2 AND g.owner=?3 AND g.fingerprint=?4
               AND g.lease_until>?5 AND g.config_revision=c.revision AND g.state IN ('building','validating')",
            params![
                lease.generation_id,
                lease.epoch,
                lease.owner,
                lease.fingerprint,
                timestamp
            ],
            |r| r.get::<_, i64>(0).map(|n| n as usize),
        )
        .optional()?;
    dimensions.ok_or_else(|| AppError::InvalidInput("Stale, stopped or expired embedding worker".into()))
}

fn check_write_target(conn: &Connection, target: &WriteTarget<'_>) -> Result<usize> {
    match target {
        WriteTarget::Build(lease) => check_lease(conn, lease),
        WriteTarget::Active(lease) => {
            let dimensions:Option<i64>=conn.query_row("SELECT g.dimensions FROM embedding_generations g JOIN embedding_control c ON c.active_id=g.id WHERE g.id=?1 AND g.epoch=?2 AND g.fingerprint=?3 AND g.state='active' AND c.revision=?4 AND g.scope_revision=c.revision",params![lease.generation_id,lease.epoch,lease.fingerprint,lease.config_revision],|r|r.get(0)).optional()?;
            dimensions
                .map(|n| n as usize)
                .ok_or_else(|| AppError::InvalidInput("Active embedding generation changed".into()))
        }
    }
}

/// Snapshot the existing category/age/count policy in the same write transaction
/// as candidate creation or explicit policy refresh. It is independent of the
/// profile fingerprint, so unchanged vectors can be reused across scoped builds.
fn snapshot_email_scope(conn: &Connection, generation: i64) -> Result<()> {
    let preference = |key: &str| -> Result<Option<String>> {
        Ok(conn
            .query_row("SELECT value FROM user_preferences WHERE key=?1", [key], |r| r.get(0))
            .optional()?)
    };
    for (corpus, key) in [
        (Corpus::MemoryFacts, "memory_enabled"),
        (Corpus::HelpDocs, "help_docs_enabled"),
    ] {
        let enabled = preference(key)?
            .map(|v| v.eq_ignore_ascii_case("true"))
            .unwrap_or(corpus != Corpus::MemoryFacts);
        conn.execute("UPDATE embedding_generation_corpora SET enabled=requested_enabled AND ?3 WHERE generation_id=?1 AND corpus=?2",params![generation,corpus.as_str(),enabled])?;
    }
    let days = preference("ai_max_email_age_days")?
        .and_then(|s| s.trim().parse::<i64>().ok())
        .unwrap_or(365);
    let max_count = preference("ai_max_email_count")?
        .and_then(|s| s.trim().parse::<i64>().ok())
        .unwrap_or(1000);
    let accounts: Vec<String> = {
        let mut stmt = conn.prepare("SELECT id FROM accounts")?;
        let rows = stmt.query_map([], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?;
        rows
    };
    conn.execute(
        "DELETE FROM embedding_generation_email_scopes WHERE generation_id=?1",
        [generation],
    )?;
    for account in accounts {
        let categories: Vec<String> = preference(&format!("embeddings_categories:{account}"))?
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| vec!["primary".into()]);
        let count: i64 = conn.query_row(
            "SELECT count(*) FROM emails WHERE account_id=?1 AND is_deleted=0",
            [&account],
            |r| r.get(0),
        )?;
        let cutoff = if max_count > 0 && count <= max_count {
            None
        } else {
            super::ai_processing_cutoff(days, now())
        };
        conn.execute("INSERT INTO embedding_generation_email_scopes(generation_id,account_id,categories_json,min_timestamp) VALUES(?1,?2,?3,?4)",params![generation,account,serde_json::to_string(&categories)?,cutoff])?;
    }
    // Purge newly excluded rows without ever deleting retained source material.
    for corpus in Corpus::ALL {
        let table = vector_table(generation, corpus)?;
        conn.execute(&format!("DELETE FROM {table} WHERE rowid IN (SELECT id FROM embedding_generation_chunks c WHERE c.generation_id=?1 AND c.corpus=?2 AND NOT EXISTS(SELECT 1 FROM embedding_generation_applicable_sources s WHERE s.generation_id=c.generation_id AND s.corpus=c.corpus AND s.source_id=c.source_id AND s.revision=c.source_revision))"),params![generation,corpus.as_str()])?;
    }
    conn.execute("DELETE FROM embedding_generation_chunks AS c WHERE c.generation_id=?1 AND NOT EXISTS(SELECT 1 FROM embedding_generation_applicable_sources s WHERE s.generation_id=c.generation_id AND s.corpus=c.corpus AND s.source_id=c.source_id AND s.revision=c.source_revision)",[generation])?;
    conn.execute("DELETE FROM embedding_generation_items AS i WHERE i.generation_id=?1 AND NOT EXISTS(SELECT 1 FROM embedding_generation_applicable_sources s WHERE s.generation_id=i.generation_id AND s.corpus=i.corpus AND s.source_id=i.source_id AND s.revision=i.source_revision)",[generation])?;
    Ok(())
}

fn check_source(conn: &Connection, source: &SourceRevision, generation: i64) -> Result<()> {
    let matches: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM embedding_generation_applicable_sources s
         WHERE s.generation_id=?1 AND s.corpus=?2 AND s.source_id=?3
           AND s.revision=?4 AND s.account_id IS ?5)",
        params![
            generation,
            source.corpus.as_str(),
            source.source_id,
            source.revision,
            source.account_id
        ],
        |r| r.get(0),
    )?;
    if !matches {
        return invalid("Embedding source changed, was deleted or is not applicable");
    }
    Ok(())
}

fn delete_source_rows(conn: &Connection, generation: i64, source: &SourceRevision) -> Result<()> {
    let table = vector_table(generation, source.corpus)?;
    conn.execute(
        &format!("DELETE FROM {table} WHERE rowid IN (SELECT id FROM embedding_generation_chunks WHERE generation_id=?1 AND corpus=?2 AND source_id=?3)"),
        params![generation, source.corpus.as_str(), source.source_id],
    )?;
    conn.execute(
        "DELETE FROM embedding_generation_chunks WHERE generation_id=?1 AND corpus=?2 AND source_id=?3",
        params![generation, source.corpus.as_str(), source.source_id],
    )?;
    Ok(())
}

/// Creates per-generation transactional invalidation triggers. No source delete
/// path needs to enumerate generations: cascades, folder/account cleanup, body
/// edits and source-ID reuse all update the durable source ledger.
fn create_indexes(conn: &Connection, generation: i64, dimensions: usize) -> Result<()> {
    for corpus in Corpus::ALL {
        let table = vector_table(generation, corpus)?;
        let name = corpus.as_str();
        conn.execute_batch(&format!(
            "CREATE VIRTUAL TABLE {table} USING vec0(embedding float[{dimensions}] distance_metric=cosine);
             CREATE TRIGGER invalidate_{table} BEFORE UPDATE ON embedding_sources
             WHEN old.corpus='{name}' BEGIN
               DELETE FROM {table} WHERE rowid IN (
                 SELECT id FROM embedding_generation_chunks
                 WHERE generation_id={generation} AND corpus='{name}' AND source_id=old.source_id);
               DELETE FROM embedding_generation_chunks
                 WHERE generation_id={generation} AND corpus='{name}' AND source_id=old.source_id;
               DELETE FROM embedding_generation_items
                 WHERE generation_id={generation} AND corpus='{name}' AND source_id=old.source_id;
               UPDATE embedding_generations SET validation_json=NULL
                 WHERE id={generation};
             END;"
        ))?;
    }
    Ok(())
}

impl Database {
    /// Existing raw-vector APIs can only address unverified legacy data. Returning
    /// no candidates preserves FTS fallback; DB errors must still propagate.
    pub(crate) fn legacy_embedding_search_available(&self) -> Result<bool> {
        let state: String = self
            .reader()
            .query_row("SELECT state FROM embedding_generations WHERE id=0", [], |r| r.get(0))?;
        Ok(state != "legacy_unverified")
    }

    /// Inspect physical active tables at startup without recreating them. Missing
    /// tables invalidate the active pointer's usability and enter repair state.
    /// Remove broken cleanup triggers so source edits/deletes remain possible.
    pub fn check_embedding_generation_health(&self) -> Result<()> {
        let mut conn = self.connection();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let generations: Vec<(i64, i64)> = {
            let mut stmt = tx.prepare("SELECT id,dimensions FROM embedding_generations WHERE id>0")?;
            let rows = stmt
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<rusqlite::Result<_>>()?;
            rows
        };
        for (generation, dimensions) in generations {
            let mut broken = false;
            for corpus in Corpus::ALL {
                let table = vector_table(generation, corpus)?;
                let exists: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name=?1 AND type='table')",
                    [&table],
                    |r| r.get(0),
                )?;
                if !exists {
                    tx.execute_batch(&format!("DROP TRIGGER IF EXISTS invalidate_{table}"))?;
                    broken = true;
                } else {
                    let schema: String =
                        tx.query_row("SELECT sql FROM sqlite_master WHERE name=?1", [&table], |r| r.get(0))?;
                    let sample = tx
                        .query_row(&format!("SELECT vec_length(embedding) FROM {table} LIMIT 1"), [], |r| {
                            r.get::<_, i64>(0)
                        })
                        .optional();
                    let invalid_schema = !schema.contains(&format!("float[{dimensions}]"))
                        || !schema.to_lowercase().contains("using vec0");
                    let invalid_sample = match sample {
                        Ok(Some(length)) => length != dimensions,
                        Ok(None) => false,
                        Err(rusqlite::Error::SqliteFailure(error, message))
                            if matches!(
                                error.code,
                                rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
                            ) =>
                        {
                            return Err(rusqlite::Error::SqliteFailure(error, message).into())
                        }
                        Err(_) => true,
                    };
                    if invalid_schema || invalid_sample {
                        tx.execute_batch(&format!("DROP TRIGGER IF EXISTS invalidate_{table}"))?;
                        broken = true;
                    }
                }
            }
            if broken {
                tx.execute("UPDATE embedding_generations SET state='repair',epoch=epoch+1,owner=NULL,lease_until=NULL,validation_json=NULL,error='Embedding index is missing or invalid; rebuild required' WHERE id=?1",[generation])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn embedding_generation_control(&self) -> Result<GenerationControl> {
        control(&self.reader())
    }

    pub fn embedding_generation_profile(&self, generation: i64) -> Result<EmbeddingProfile> {
        let json: Option<String> = self.reader().query_row(
            "SELECT profile_json FROM embedding_generations WHERE id=?1",
            [generation],
            |r| r.get(0),
        )?;
        match json {
            Some(json) => Ok(serde_json::from_str(&json)?),
            None => invalid("Legacy embedding profile is unverified"),
        }
    }

    pub fn embedding_generation_status(&self, generation: i64) -> Result<GenerationStatus> {
        Ok(self.reader().query_row(
            "SELECT id,fingerprint,dimensions,state,epoch,
               (SELECT count(*) FROM embedding_generation_items WHERE generation_id=g.id AND status='complete'),
               (SELECT count(*) FROM embedding_generation_items WHERE generation_id=g.id AND status='failed'),
               error,validation_json FROM embedding_generations g WHERE id=?1",
            [generation],
            |r| {
                Ok(GenerationStatus {
                    id: r.get(0)?,
                    fingerprint: r.get(1)?,
                    dimensions: r.get::<_, i64>(2)? as usize,
                    state: r.get(3)?,
                    epoch: r.get(4)?,
                    completed: r.get(5)?,
                    failed: r.get(6)?,
                    error: r.get(7)?,
                    validation: r.get(8)?,
                })
            },
        )?)
    }

    /// Propose configuration and create all three replacement tables atomically.
    /// Callers must probe/authorize provider identity before creating a candidate.
    pub fn prepare_embedding_generation(
        &self,
        profile: &EmbeddingProfile,
        enabled_corpora: &[Corpus],
        expected_revision: i64,
    ) -> Result<BuildLease> {
        self.prepare_embedding_generation_with_preferences(profile, enabled_corpora, expected_revision, &[])
    }

    /// The settings/candidate commit boundary. Validated service plans stage
    /// versioned credentials first, then publish their references and every
    /// proposed setting together with the replacement generation in this one
    /// top-level SQLite transaction. No network or keychain operation runs here.
    pub fn prepare_embedding_generation_with_preferences(
        &self,
        profile: &EmbeddingProfile,
        enabled_corpora: &[Corpus],
        expected_revision: i64,
        preferences: &[(String, Option<String>)],
    ) -> Result<BuildLease> {
        let profile = profile.canonicalized()?;
        let fingerprint = profile.fingerprint()?;
        let owner = uuid::Uuid::new_v4().to_string();
        let mut conn = self.connection();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current = control(&tx)?;
        if current.revision != expected_revision || current.pending_id.is_some() {
            return invalid("Embedding configuration changed or another build is pending");
        }
        for (key, value) in preferences {
            match value {
                Some(value) => {
                    tx.execute("INSERT INTO user_preferences(key,value) VALUES(?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value WHERE value IS NOT excluded.value",params![key,value])?;
                }
                None => {
                    tx.execute("DELETE FROM user_preferences WHERE key=?1", [key])?;
                }
            }
        }
        let prepared_revision = control(&tx)?.revision;
        let revision = prepared_revision + 1;
        tx.execute(
            "INSERT INTO embedding_generations(profile_json,fingerprint,dimensions,state,epoch,owner,lease_until,config_revision,created_at)
             VALUES(?1,?2,?3,'preparing',1,?4,?5,?6,?7)",
            params![serde_json::to_string(&profile)?, fingerprint, profile.dimensions as i64, owner, now()+LEASE_SECONDS, revision, now()],
        )?;
        let generation = tx.last_insert_rowid();
        for corpus in Corpus::ALL {
            tx.execute(
                "INSERT INTO embedding_generation_corpora(generation_id,corpus,enabled,requested_enabled) VALUES(?1,?2,?3,?3)",
                params![generation, corpus.as_str(), enabled_corpora.contains(&corpus)],
            )?;
        }
        create_indexes(&tx, generation, profile.dimensions)?;
        snapshot_email_scope(&tx, generation)?;
        tx.execute(
            "UPDATE embedding_generations SET state='building',scope_revision=?2 WHERE id=?1",
            params![generation, revision],
        )?;
        tx.execute(
            "UPDATE embedding_control SET revision=?1,pending_id=?2 WHERE singleton=1",
            params![revision, generation],
        )?;
        if !preferences.is_empty() {
            if let Some(active) = current.active_id {
                let healthy: bool = tx.query_row(
                    "SELECT state='active' FROM embedding_generations WHERE id=?1",
                    [active],
                    |r| r.get(0),
                )?;
                if healthy {
                    // This explicit combined settings operation refreshes current
                    // selection/consent without relabelling the active vectors.
                    // Same-destination model/dimension replacements can therefore
                    // continue querying the previous immutable profile while built.
                    snapshot_email_scope(&tx, active)?;
                    tx.execute("UPDATE embedding_generations SET scope_revision=?2,mutation_version=mutation_version+1,validation_json=NULL WHERE id=?1",params![active,revision])?;
                }
            }
        } else {
            carry_active_scope(&tx, prepared_revision)?;
        }
        tx.commit()?;
        Ok(BuildLease {
            generation_id: generation,
            epoch: 1,
            owner,
            fingerprint,
        })
    }

    /// Pending, revision-stamped inputs. The service fetches and formats source
    /// content under a matching reader snapshot; a later edit fences its result.
    pub fn embedding_generation_sources(
        &self,
        generation: i64,
        corpus: Corpus,
        limit: usize,
    ) -> Result<Vec<SourceRevision>> {
        let conn = self.reader();
        let mut stmt = conn.prepare(
            "SELECT s.source_id,s.account_id,s.revision FROM embedding_generation_applicable_sources s
             WHERE s.generation_id=?1 AND s.corpus=?2
               AND NOT EXISTS(SELECT 1 FROM embedding_generation_items i WHERE i.generation_id=?1
                 AND i.corpus=s.corpus AND i.source_id=s.source_id AND i.source_revision=s.revision AND i.status='complete')
             ORDER BY s.source_id LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(params![generation, corpus.as_str(), limit as i64], |r| {
                Ok(SourceRevision {
                    corpus,
                    source_id: r.get(0)?,
                    account_id: r.get(1)?,
                    revision: r.get(2)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn heartbeat_embedding_generation(&self, lease: &BuildLease) -> Result<()> {
        self.heartbeat_embedding_generation_at(lease, now())
    }

    pub(crate) fn heartbeat_embedding_generation_at(&self, lease: &BuildLease, timestamp: i64) -> Result<()> {
        let mut conn = self.connection();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        check_lease_at(&tx, lease, timestamp)?;
        tx.execute(
            "UPDATE embedding_generations SET lease_until=?2 WHERE id=?1",
            params![lease.generation_id, timestamp + LEASE_SECONDS],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Capture the active profile/epoch for ordinary incremental indexing. A
    /// switch, rollback, repair or source edit fences an in-flight response.
    pub fn active_embedding_write_lease(&self) -> Result<Option<ActiveGenerationLease>> {
        Ok(self.reader().query_row("SELECT g.id,g.epoch,g.fingerprint,c.revision FROM embedding_generations g JOIN embedding_control c ON c.active_id=g.id WHERE g.state='active'",[],|r|Ok(ActiveGenerationLease{generation_id:r.get(0)?,epoch:r.get(1)?,fingerprint:r.get(2)?,config_revision:r.get(3)?})).optional()?)
    }

    /// Refresh source-selection policy after a settings/account change without
    /// changing the embedding profile. Stage 3 calls this before enumerating work;
    /// a stale scope is rejected by the ordinary active-write boundary.
    pub fn refresh_active_embedding_scope(&self, lease: &ActiveGenerationLease) -> Result<()> {
        let mut conn = self.connection();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let valid:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM embedding_generations g JOIN embedding_control c ON c.active_id=g.id WHERE g.id=?1 AND g.epoch=?2 AND g.fingerprint=?3 AND g.state='active' AND c.revision=?4)",params![lease.generation_id,lease.epoch,lease.fingerprint,lease.config_revision],|r|r.get(0))?;
        if !valid {
            return invalid("Active embedding generation or configuration changed");
        }
        snapshot_email_scope(&tx, lease.generation_id)?;
        tx.execute("UPDATE embedding_generations SET scope_revision=?2,mutation_version=mutation_version+1,validation_json=NULL WHERE id=?1",params![lease.generation_id,lease.config_revision])?;
        tx.commit()?;
        Ok(())
    }

    /// Commit all chunks of a source together with a frozen build identity.
    pub fn store_generation_embeddings(
        &self,
        lease: &BuildLease,
        source: &SourceRevision,
        inputs: &[String],
        vectors: &[Vec<f32>],
    ) -> Result<()> {
        self.store_embeddings_for_target(WriteTarget::Build(lease), lease.generation_id, source, inputs, vectors)
    }

    pub fn store_active_generation_embeddings(
        &self,
        lease: &ActiveGenerationLease,
        source: &SourceRevision,
        inputs: &[String],
        vectors: &[Vec<f32>],
    ) -> Result<()> {
        self.store_embeddings_for_target(WriteTarget::Active(lease), lease.generation_id, source, inputs, vectors)
    }

    fn store_embeddings_for_target(
        &self,
        target: WriteTarget<'_>,
        generation: i64,
        source: &SourceRevision,
        inputs: &[String],
        vectors: &[Vec<f32>],
    ) -> Result<()> {
        if inputs.is_empty() || inputs.len() != vectors.len() || inputs.iter().any(|s| s.is_empty()) {
            return invalid("Embedding batch input/result association is invalid");
        }
        let mut conn = self.connection();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let dimensions = check_write_target(&tx, &target)?;
        check_source(&tx, source, generation)?;
        if vectors
            .iter()
            .any(|v| v.len() != dimensions || v.iter().any(|n| !n.is_finite()) || v.iter().all(|n| *n == 0.0))
        {
            return invalid("Embedding vector has wrong dimensions, non-finite values or zero norm");
        }
        delete_source_rows(&tx, generation, source)?;
        let table = vector_table(generation, source.corpus)?;
        for (index, (input, vector)) in inputs.iter().zip(vectors).enumerate() {
            tx.execute("INSERT INTO embedding_generation_chunks(generation_id,corpus,source_id,source_revision,chunk_index,input_hash) VALUES(?1,?2,?3,?4,?5,?6)",
                params![generation,source.corpus.as_str(),source.source_id,source.revision,index as i64,input_hash(input.as_bytes())])?;
            let rowid = tx.last_insert_rowid();
            let blob: Vec<u8> = vector.iter().flat_map(|n| n.to_le_bytes()).collect();
            tx.execute(
                &format!("INSERT INTO {table}(rowid,embedding) VALUES(?1,?2)"),
                params![rowid, blob],
            )?;
        }
        let hash = input_hash(&serde_json::to_vec(inputs)?);
        tx.execute("INSERT INTO embedding_generation_items(generation_id,corpus,source_id,source_revision,input_hash,status,chunk_count) VALUES(?1,?2,?3,?4,?5,'complete',?6) ON CONFLICT(generation_id,corpus,source_id) DO UPDATE SET source_revision=excluded.source_revision,input_hash=excluded.input_hash,status='complete',chunk_count=excluded.chunk_count,error=NULL",
            params![generation,source.corpus.as_str(),source.source_id,source.revision,hash,vectors.len() as i64])?;
        tx.execute("UPDATE embedding_generations SET validation_json=NULL,mutation_version=mutation_version+1,state=CASE WHEN state='validating' THEN 'building' ELSE state END WHERE id=?1", [generation])?;
        tx.commit()?;
        Ok(())
    }

    /// Error strings must be sanitized by the service (no request bodies/tokens).
    pub fn fail_generation_source(&self, lease: &BuildLease, source: &SourceRevision, error: &str) -> Result<()> {
        let mut conn = self.connection();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        check_lease(&tx, lease)?;
        check_source(&tx, source, lease.generation_id)?;
        delete_source_rows(&tx, lease.generation_id, source)?;
        tx.execute("INSERT INTO embedding_generation_items(generation_id,corpus,source_id,source_revision,input_hash,status,error) VALUES(?1,?2,?3,?4,'','failed',?5) ON CONFLICT(generation_id,corpus,source_id) DO UPDATE SET source_revision=excluded.source_revision,status='failed',error=excluded.error",
            params![lease.generation_id,source.corpus.as_str(),source.source_id,source.revision,error])?;
        tx.execute(
            "UPDATE embedding_generations SET validation_json=NULL,mutation_version=mutation_version+1 WHERE id=?1",
            [lease.generation_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// Pause/fail immediately fences responses already in flight. Completed
    /// vectors remain available to a later explicit resume.
    pub fn stop_embedding_generation(&self, generation: i64, failed: bool, error: Option<&str>) -> Result<()> {
        let mut conn = self.connection();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let changed=tx.execute("UPDATE embedding_generations SET state=?2,error=?3,epoch=epoch+1,owner=NULL,lease_until=NULL,validation_json=NULL WHERE id=?1 AND id=(SELECT pending_id FROM embedding_control) AND state IN ('building','validating','preparing')",
            params![generation,if failed {"failed"} else {"paused"},error])?;
        if changed != 1 {
            return invalid("No running candidate to stop");
        }
        tx.commit()?;
        Ok(())
    }

    /// Recover only expired owners: opening a second app/CLI must not cancel a
    /// healthy worker in another process. No inference is restarted here.
    pub fn recover_embedding_generations(&self) -> Result<usize> {
        let conn = self.connection();
        Ok(conn.execute("UPDATE embedding_generations SET state='interrupted',epoch=epoch+1,owner=NULL,lease_until=NULL,validation_json=NULL,error='Embedding worker lease expired' WHERE state IN ('preparing','building','validating') AND (lease_until IS NULL OR lease_until<=?1)", [now()])?)
    }

    /// The service must re-probe identity and authorization before explicit
    /// resume. Passing a matching fingerprint cannot restore revoked permissions.
    pub fn resume_embedding_generation(
        &self,
        generation: i64,
        expected_fingerprint: &str,
        expected_revision: i64,
    ) -> Result<BuildLease> {
        let mut conn = self.connection();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current = control(&tx)?;
        if current.pending_id != Some(generation) || current.revision != expected_revision {
            return invalid("Embedding configuration changed");
        }
        let owner = uuid::Uuid::new_v4().to_string();
        let changed=tx.execute("UPDATE embedding_generations SET state='building',epoch=epoch+1,owner=?2,lease_until=?3,config_revision=?4,error=NULL,validation_json=NULL WHERE id=?1 AND fingerprint=?5 AND state IN ('paused','failed','interrupted')",
            params![generation,owner,now()+LEASE_SECONDS,expected_revision,expected_fingerprint])?;
        if changed != 1 {
            return invalid("Candidate is not resumable or profile changed");
        }
        snapshot_email_scope(&tx, generation)?;
        tx.execute(
            "UPDATE embedding_generations SET scope_revision=?2,mutation_version=mutation_version+1 WHERE id=?1",
            params![generation, expected_revision],
        )?;
        let epoch = tx.query_row(
            "SELECT epoch FROM embedding_generations WHERE id=?1",
            [generation],
            |r| r.get(0),
        )?;
        tx.commit()?;
        Ok(BuildLease {
            generation_id: generation,
            epoch,
            owner,
            fingerprint: expected_fingerprint.into(),
        })
    }

    /// Scan on a reader snapshot; then publish validation only if the source and
    /// generation mutation watermarks still match. Provider work never holds a
    /// DB lock. The final activation transaction only compares these fences.
    pub fn validate_embedding_generation(&self, lease: &BuildLease) -> Result<()> {
        let (source_version, mutation_version, schema_version, dimensions) = {
            let mut conn = self.reader();
            let snapshot = conn.transaction()?;
            let dimensions = check_lease(&snapshot, lease)?;
            let source_version: i64 =
                snapshot.query_row("SELECT source_version FROM embedding_control", [], |r| r.get(0))?;
            let mutation_version: i64 = snapshot.query_row(
                "SELECT mutation_version FROM embedding_generations WHERE id=?1",
                [lease.generation_id],
                |r| r.get(0),
            )?;
            let schema_version: i64 = snapshot.query_row("PRAGMA schema_version", [], |r| r.get(0))?;
            validate_generation(&snapshot, lease.generation_id, dimensions)?;
            snapshot.commit()?;
            (source_version, mutation_version, schema_version, dimensions)
        };
        let mut conn = self.connection();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        check_lease(&tx, lease)?;
        let current_schema: i64 = tx.query_row("PRAGMA schema_version", [], |r| r.get(0))?;
        if current_schema != schema_version {
            return invalid("Embedding schema changed during validation");
        }
        let changed=tx.execute("UPDATE embedding_generations SET state='validating',validation_json=?2,validated_source_version=?3,validated_mutation_version=?4,validated_schema_version=?5 WHERE id=?1 AND mutation_version=?4 AND config_revision=(SELECT revision FROM embedding_control) AND ?3=(SELECT source_version FROM embedding_control)",
            params![lease.generation_id,serde_json::json!({"version":1,"dimensions":dimensions,"validated_at":now()}).to_string(),source_version,mutation_version,schema_version])?;
        if changed != 1 {
            return invalid("Embedding sources or configuration changed during validation");
        }
        tx.commit()?;
        Ok(())
    }

    pub fn activate_embedding_generation(&self, lease: &BuildLease) -> Result<()> {
        let mut conn = self.connection();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        check_lease(&tx, lease)?;
        let ready:bool=tx.query_row("SELECT state='validating' AND validation_json IS NOT NULL AND config_revision=(SELECT revision FROM embedding_control) AND validated_source_version=(SELECT source_version FROM embedding_control) AND validated_mutation_version=mutation_version FROM embedding_generations WHERE id=?1", [lease.generation_id],|r|r.get(0))?;
        if !ready {
            return invalid("Candidate requires validation or configuration changed");
        }
        let schema: i64 = tx.query_row("PRAGMA schema_version", [], |r| r.get(0))?;
        let validated_schema: i64 = tx.query_row(
            "SELECT validated_schema_version FROM embedding_generations WHERE id=?1",
            [lease.generation_id],
            |r| r.get(0),
        )?;
        if schema != validated_schema {
            return invalid("Embedding schema changed after validation");
        }
        let previous = control(&tx)?.active_id;
        if let Some(id) = previous {
            tx.execute("UPDATE embedding_generations SET state='retained',epoch=epoch+1,owner=NULL,lease_until=NULL WHERE id=?1", [id])?;
        }
        tx.execute("UPDATE embedding_generations SET state='active',epoch=epoch+1,owner=NULL,lease_until=NULL,scope_revision=(SELECT revision+1 FROM embedding_control) WHERE id=?1", [lease.generation_id])?;
        tx.execute("UPDATE embedding_control SET retained_id=active_id,active_id=?1,pending_id=NULL,revision=revision+1 WHERE singleton=1", [lease.generation_id])?;
        tx.commit()?;
        Ok(())
    }

    /// Reopen retained data as the candidate, preserving the active generation.
    /// Catch-up and normal validation/activation are mandatory before switching.
    pub fn prepare_embedding_rollback(&self, expected_revision: i64, fingerprint: &str) -> Result<BuildLease> {
        let mut conn = self.connection();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current = control(&tx)?;
        if current.revision != expected_revision || current.pending_id.is_some() {
            return invalid("Embedding configuration changed or build pending");
        }
        let generation = current
            .retained_id
            .ok_or_else(|| AppError::InvalidInput("No retained generation".into()))?;
        let owner = uuid::Uuid::new_v4().to_string();
        let revision = current.revision + 1;
        let changed=tx.execute("UPDATE embedding_generations SET state='building',epoch=epoch+1,owner=?2,lease_until=?3,config_revision=?4,validation_json=NULL WHERE id=?1 AND state='retained' AND fingerprint=?5",
            params![generation,owner,now()+LEASE_SECONDS,revision,fingerprint])?;
        if changed != 1 {
            return invalid("Retained profile does not match verified provider identity");
        }
        snapshot_email_scope(&tx, generation)?;
        tx.execute(
            "UPDATE embedding_generations SET scope_revision=?2,mutation_version=mutation_version+1 WHERE id=?1",
            params![generation, revision],
        )?;
        let epoch = tx.query_row(
            "SELECT epoch FROM embedding_generations WHERE id=?1",
            [generation],
            |r| r.get(0),
        )?;
        tx.execute(
            "UPDATE embedding_control SET pending_id=?1,retained_id=NULL,revision=?2",
            params![generation, revision],
        )?;
        carry_active_scope(&tx, current.revision)?;
        tx.commit()?;
        Ok(BuildLease {
            generation_id: generation,
            epoch,
            owner,
            fingerprint: fingerprint.into(),
        })
    }

    /// Drop only an explicitly selected inactive generation. DDL, metadata and
    /// control pointers commit together; the active generation cannot be dropped.
    pub fn discard_embedding_generation(&self, generation: i64) -> Result<()> {
        if generation <= 0 {
            return invalid("Legacy generation must be preserved");
        }
        let mut conn = self.connection();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current = control(&tx)?;
        if current.active_id == Some(generation) {
            return invalid("Cannot discard the active embedding generation");
        }
        let state: String = tx.query_row(
            "SELECT state FROM embedding_generations WHERE id=?1",
            [generation],
            |r| r.get(0),
        )?;
        if matches!(state.as_str(), "preparing" | "building" | "validating") {
            return invalid("Stop the generation before discarding it");
        }
        for corpus in Corpus::ALL {
            let table = vector_table(generation, corpus)?;
            tx.execute_batch(&format!(
                "DROP TRIGGER IF EXISTS invalidate_{table}; DROP TABLE IF EXISTS {table};"
            ))?;
        }
        tx.execute("UPDATE embedding_control SET pending_id=CASE WHEN pending_id=?1 THEN NULL ELSE pending_id END,retained_id=CASE WHEN retained_id=?1 THEN NULL ELSE retained_id END,revision=revision+1", [generation])?;
        carry_active_scope(&tx, current.revision)?;
        tx.execute("DELETE FROM embedding_generations WHERE id=?1", [generation])?;
        tx.commit()?;
        Ok(())
    }

    /// Same-fingerprint scoped rebuild optimization. Only current, verified,
    /// unchanged source rows can be copied; no legacy or cross-profile copying.
    pub fn copy_generation_source(
        &self,
        lease: &BuildLease,
        from_generation: i64,
        source: &SourceRevision,
    ) -> Result<()> {
        if from_generation == lease.generation_id {
            return invalid("Cannot copy a generation into itself");
        }
        let mut conn = self.connection();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        check_lease(&tx, lease)?;
        check_source(&tx, source, lease.generation_id)?;
        let compatible:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM embedding_generations g JOIN embedding_generation_items i ON i.generation_id=g.id WHERE g.id=?1 AND g.fingerprint=?2 AND g.state IN ('active','retained') AND i.corpus=?3 AND i.source_id=?4 AND i.source_revision=?5 AND i.status='complete')",
            params![from_generation,lease.fingerprint,source.corpus.as_str(),source.source_id,source.revision],|r|r.get(0))?;
        if !compatible {
            return invalid("Source generation is unverified, stale or has a different fingerprint");
        }
        let from = vector_table(from_generation, source.corpus)?;
        let to = vector_table(lease.generation_id, source.corpus)?;
        // Read before mutating; all source rows are bounded by one document.
        let rows: Vec<(i64, String, Vec<u8>)> = {
            let mut stmt=tx.prepare(&format!("SELECT c.chunk_index,c.input_hash,v.embedding FROM embedding_generation_chunks c JOIN {from} v ON v.rowid=c.id WHERE c.generation_id=?1 AND c.corpus=?2 AND c.source_id=?3 AND c.source_revision=?4 ORDER BY c.chunk_index"))?;
            let rows = stmt
                .query_map(
                    params![
                        from_generation,
                        source.corpus.as_str(),
                        source.source_id,
                        source.revision
                    ],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows
        };
        let expected: i64 = tx.query_row(
            "SELECT chunk_count FROM embedding_generation_items WHERE generation_id=?1 AND corpus=?2 AND source_id=?3",
            params![from_generation, source.corpus.as_str(), source.source_id],
            |r| r.get(0),
        )?;
        if rows.is_empty() || rows.len() as i64 != expected {
            return invalid("Source generation is missing vectors");
        }
        delete_source_rows(&tx, lease.generation_id, source)?;
        for (index, hash, blob) in rows {
            tx.execute("INSERT INTO embedding_generation_chunks(generation_id,corpus,source_id,source_revision,chunk_index,input_hash) VALUES(?1,?2,?3,?4,?5,?6)",params![lease.generation_id,source.corpus.as_str(),source.source_id,source.revision,index,hash])?;
            tx.execute(
                &format!("INSERT INTO {to}(rowid,embedding) VALUES(?1,?2)"),
                params![tx.last_insert_rowid(), blob],
            )?;
        }
        tx.execute("INSERT INTO embedding_generation_items(generation_id,corpus,source_id,source_revision,input_hash,status,chunk_count) SELECT ?1,corpus,source_id,source_revision,input_hash,status,chunk_count FROM embedding_generation_items WHERE generation_id=?2 AND corpus=?3 AND source_id=?4 ON CONFLICT(generation_id,corpus,source_id) DO UPDATE SET source_revision=excluded.source_revision,input_hash=excluded.input_hash,status='complete',chunk_count=excluded.chunk_count,error=NULL",params![lease.generation_id,from_generation,source.corpus.as_str(),source.source_id])?;
        tx.execute("UPDATE embedding_generations SET validation_json=NULL,mutation_version=mutation_version+1,state='building' WHERE id=?1", [lease.generation_id])?;
        tx.commit()?;
        Ok(())
    }
}

fn validate_generation(conn: &Connection, generation: i64, dimensions: usize) -> Result<()> {
    let incomplete:i64=conn.query_row("SELECT count(*) FROM embedding_generation_applicable_sources s WHERE s.generation_id=?1 AND NOT EXISTS(SELECT 1 FROM embedding_generation_items i WHERE i.generation_id=?1 AND i.corpus=s.corpus AND i.source_id=s.source_id AND i.source_revision=s.revision AND i.status='complete')",[generation],|r|r.get(0))?;
    let failed: i64 = conn.query_row(
        "SELECT count(*) FROM embedding_generation_items WHERE generation_id=?1 AND status='failed'",
        [generation],
        |r| r.get(0),
    )?;
    if incomplete != 0 || failed != 0 {
        return invalid("Embedding generation has incomplete coverage or failed items");
    }
    let stale_items:i64=conn.query_row("SELECT count(*) FROM embedding_generation_items i WHERE i.generation_id=?1 AND NOT EXISTS(SELECT 1 FROM embedding_generation_applicable_sources s WHERE s.generation_id=i.generation_id AND s.corpus=i.corpus AND s.source_id=i.source_id AND s.revision=i.source_revision)",[generation],|r|r.get(0))?;
    if stale_items != 0 {
        return invalid("Embedding items are stale or outside enabled corpora");
    }
    for corpus in Corpus::ALL {
        let table = vector_table(generation, corpus)?;
        let bad:i64=conn.query_row(&format!("SELECT count(*) FROM embedding_generation_chunks c LEFT JOIN embedding_sources s ON s.corpus=c.corpus AND s.source_id=c.source_id LEFT JOIN embedding_generation_items i ON i.generation_id=c.generation_id AND i.corpus=c.corpus AND i.source_id=c.source_id WHERE c.generation_id=?1 AND c.corpus=?2 AND (s.revision IS NOT c.source_revision OR s.eligible IS NOT 1 OR i.status IS NOT 'complete' OR NOT EXISTS(SELECT 1 FROM {table} v WHERE v.rowid=c.id))"),params![generation,corpus.as_str()],|r|r.get(0))?;
        let orphan:i64=conn.query_row(&format!("SELECT count(*) FROM {table} v WHERE NOT EXISTS(SELECT 1 FROM embedding_generation_chunks c WHERE c.id=v.rowid AND c.generation_id=?1 AND c.corpus=?2)"),params![generation,corpus.as_str()],|r|r.get(0))?;
        let empty_items:i64=conn.query_row("SELECT count(*) FROM embedding_generation_items i WHERE i.generation_id=?1 AND i.corpus=?2 AND (i.chunk_count=0 OR i.chunk_count<>(SELECT count(*) FROM embedding_generation_chunks c WHERE c.generation_id=i.generation_id AND c.corpus=i.corpus AND c.source_id=i.source_id))",params![generation,corpus.as_str()],|r|r.get(0))?;
        if bad != 0 || orphan != 0 || empty_items != 0 {
            return invalid("Embedding vector/metadata/source consistency check failed");
        }
        // Execute the actual extension KNN path with a stored vector. Empty or
        // disabled corpora still exercise schema/dimension validation via probe.
        let probe: Option<Vec<u8>> = conn
            .query_row(&format!("SELECT embedding FROM {table} LIMIT 1"), [], |r| r.get(0))
            .optional()?;
        let probe = probe.unwrap_or_else(|| {
            std::iter::repeat_n(1.0_f32, dimensions)
                .flat_map(f32::to_le_bytes)
                .collect()
        });
        let _: i64 = conn.query_row(
            &format!("SELECT count(*) FROM (SELECT rowid FROM {table} WHERE embedding MATCH ?1 AND k=1)"),
            [probe],
            |r| r.get(0),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
