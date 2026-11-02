use super::*;
use tempfile::TempDir;

fn profile(dimensions: usize) -> EmbeddingProfile {
    EmbeddingProfile {
        version: 1,
        backend: "synthetic".into(),
        endpoint: Some("HTTP://LOCALHOST:1234/v1/".into()),
        model: "fixture-embedding".into(),
        model_revision: Some("fixture-revision-1".into()),
        dimension_policy: DimensionPolicy::Explicit(dimensions),
        dimensions,
        query_format: "query-v1".into(),
        document_format: "document-v1".into(),
        preprocessing: "full-content-v1".into(),
        normalization: "none".into(),
        encoding: "float32-le".into(),
        distance_metric: "cosine".into(),
    }
}

fn file_db() -> (TempDir, Database) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::new(dir.path().to_path_buf()).unwrap();
    (dir, db)
}

fn seed(conn: &Connection) {
    conn.execute_batch(
        "INSERT INTO accounts(id,provider,email,name,created_at) VALUES('account-a','gmail','fixture@example.test','Fixture',0);
         INSERT INTO emails(id,account_id,thread_id,subject,sender,sender_email,recipients_json,snippet,timestamp,created_at)
           VALUES('email-a','account-a','thread-a','Fixture subject','Fixture sender','sender@example.test','[]','Fixture snippet',1,1);
         INSERT INTO email_bodies(email_id,body) VALUES('email-a','Complete synthetic body');
         INSERT INTO memory_facts(id,account_id,subject_kind,subject_key,fact,created_at,updated_at)
           VALUES('fact-a','account-a','user','self','Synthetic fact',1,1);
         INSERT INTO help_doc_chunks(chunk_id,lang,page,section_index,page_title,heading,content,created_at)
           VALUES('help-a','en','fixture',0,'Fixture guide','Fixture heading','Synthetic guide content',1);",
    ).unwrap();
}

fn prepare(db: &Database, dimensions: usize) -> BuildLease {
    db.prepare_embedding_generation(
        &profile(dimensions),
        &Corpus::ALL,
        db.embedding_generation_control().unwrap().revision,
    )
    .unwrap()
}

fn populate(db: &Database, lease: &BuildLease, dimensions: usize) {
    for corpus in Corpus::ALL {
        for source in db
            .embedding_generation_sources(lease.generation_id, corpus, 100)
            .unwrap()
        {
            db.store_generation_embeddings(
                lease,
                &source,
                &[format!("complete formatted {}", source.source_id)],
                &[vector(dimensions, 0)],
            )
            .unwrap();
        }
    }
}

fn vector(dimensions: usize, direction: usize) -> Vec<f32> {
    let mut v = vec![0.0; dimensions];
    v[direction] = 1.0;
    v
}

fn activate(db: &Database, lease: &BuildLease) {
    db.validate_embedding_generation(lease).unwrap();
    db.activate_embedding_generation(lease).unwrap();
}

fn count(db: &Database, table: &str) -> i64 {
    db.reader()
        .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
        .unwrap()
}

fn source(db: &Database, generation: i64, corpus: Corpus) -> SourceRevision {
    db.embedding_generation_sources(generation, corpus, 100)
        .unwrap()
        .remove(0)
}

#[test]
fn profile_fingerprints_cover_every_identity_field_and_full_input() {
    let p = profile(768);
    let mut canonical = p.clone();
    canonical.endpoint = Some("http://localhost:1234/v1".into());
    assert_eq!(p.fingerprint().unwrap(), canonical.fingerprint().unwrap());
    let mut changed = p.clone();
    changed.model_revision = Some("revision-2".into());
    assert_ne!(p.fingerprint().unwrap(), changed.fingerprint().unwrap());
    changed = p.clone();
    changed.dimension_policy = DimensionPolicy::Native;
    assert_ne!(p.fingerprint().unwrap(), changed.fingerprint().unwrap());
    changed = p.clone();
    changed.query_format = "query-v2".into();
    assert_ne!(p.fingerprint().unwrap(), changed.fingerprint().unwrap());
    let mut body = "x".repeat(1000);
    let first = input_hash(body.as_bytes());
    body.push('y');
    assert_ne!(first, input_hash(body.as_bytes()));
    for endpoint in [
        "http://user:password@localhost/v1",
        "http://localhost/v1?key=secret",
        "file:///tmp/model",
        "https://localhost/v1#secret",
    ] {
        changed = p.clone();
        changed.endpoint = Some(endpoint.into());
        assert!(changed.fingerprint().is_err());
    }
    for dimensions in [0, 8193] {
        assert!(profile(dimensions).fingerprint().is_err());
    }
    changed = p;
    changed.dimension_policy = DimensionPolicy::Explicit(4096);
    assert!(changed.fingerprint().is_err());
}

#[test]
fn populated_file_migration_preserves_legacy_bytes_rowids_sources_and_settings() {
    // Register the extension without opening any user data, then create V040.
    drop(Database::new_for_testing().unwrap());
    let dir = tempfile::tempdir().unwrap();
    let mut conn = Connection::open(dir.path().join("emailops.db")).unwrap();
    crate::db::embedded::migrations::runner()
        .set_target(refinery::Target::Version(40))
        .run(&mut conn)
        .unwrap();
    seed(&conn);
    let bytes: Vec<u8> = vector(768, 0).into_iter().flat_map(f32::to_le_bytes).collect();
    conn.execute_batch("INSERT INTO embedding_chunks(rowid,email_id,account_id,chunk_index,embedding_model,content_hash,created_at) VALUES(41,'email-a','account-a',0,'legacy','legacy-hash',1);
       INSERT INTO memory_fact_chunks(rowid,fact_id,embedding_model,created_at) VALUES(42,'fact-a','legacy',1);
       UPDATE help_doc_chunks SET rowid=43,embedding_model='legacy';
       INSERT INTO user_preferences(key,value) VALUES('ai_embedding_model','legacy-model'),('openai_compatible_embedding_dimensions','requested'),('unrelated-setting','preserve');").unwrap();
    for (table, rowid) in [("vec_emails", 41), ("vec_memory_facts", 42), ("vec_help_docs", 43)] {
        conn.execute(
            &format!("INSERT INTO {table}(rowid,embedding) VALUES(?1,?2)"),
            params![rowid, bytes],
        )
        .unwrap();
    }
    drop(conn);
    let db = Database::new(dir.path().to_path_buf()).unwrap();
    assert_eq!(db.embedding_generation_status(0).unwrap().state, "legacy_unverified");
    assert!(db.embedding_generation_profile(0).is_err());
    let c = db.embedding_generation_control().unwrap();
    assert_eq!((c.active_id, c.pending_id), (None, None));
    for (table, rowid) in [("vec_emails", 41), ("vec_memory_facts", 42), ("vec_help_docs", 43)] {
        let actual: Vec<u8> = db
            .reader()
            .query_row(&format!("SELECT embedding FROM {table} WHERE rowid=?1"), [rowid], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(actual, bytes);
    }
    for table in ["emails", "email_bodies", "memory_facts", "help_doc_chunks"] {
        assert_eq!(count(&db, table), 1);
    }
    assert_eq!(count(&db, "embedding_sources"), 3);
    assert_eq!(
        db.get_preference("unrelated-setting").unwrap().as_deref(),
        Some("preserve")
    );
    assert_eq!(
        db.get_preference("openai_compatible_embedding_dimensions")
            .unwrap()
            .as_deref(),
        Some("requested")
    );
    assert_eq!(
        db.get_preference("ai_embedding_model").unwrap().as_deref(),
        Some("legacy-model")
    );
    assert!(db
        .vec_search(&vector(768, 0), Some("account-a"), None, 10)
        .unwrap()
        .is_empty());
    assert!(db
        .vec_search_memory_facts(&vector(768, 0), "account-a", 10)
        .unwrap()
        .is_empty());
    assert!(db
        .vec_search_help_docs(&vector(768, 0), "legacy", 10, None)
        .unwrap()
        .is_empty());
    assert!(!db
        .fts_search("Fixture", Some("account-a"), None, 10)
        .unwrap()
        .is_empty());
    assert!(db.discard_embedding_generation(0).is_err());
}

#[test]
fn all_three_dimensions_and_corpora_persist_query_and_reopen() {
    for dimensions in [768, 2560, 4096] {
        let (dir, db) = file_db();
        seed(&db.connection());
        let lease = prepare(&db, dimensions);
        let original = source(&db, lease.generation_id, Corpus::Emails);
        assert!(db
            .store_generation_embeddings(&lease, &original, &["input".into()], &[vector(dimensions - 1, 0)])
            .is_err());
        for bad in [f32::NAN, f32::INFINITY] {
            let mut v = vector(dimensions, 0);
            v[1] = bad;
            assert!(db
                .store_generation_embeddings(&lease, &original, &["input".into()], &[v])
                .is_err());
        }
        assert!(db
            .store_generation_embeddings(&lease, &original, &["input".into()], &[vec![0.0; dimensions]])
            .is_err());
        assert!(db
            .store_generation_embeddings(
                &lease,
                &original,
                &["one".into(), "two".into()],
                &[vector(dimensions, 0)]
            )
            .is_err());
        populate(&db, &lease, dimensions);
        activate(&db, &lease);
        for corpus in Corpus::ALL {
            let table = vector_table(lease.generation_id, corpus).unwrap();
            let bytes: Vec<u8> = vector(dimensions, 0).into_iter().flat_map(f32::to_le_bytes).collect();
            let distance: f64 = db
                .reader()
                .query_row(
                    &format!("SELECT distance FROM {table} WHERE embedding MATCH ?1 AND k=1"),
                    [bytes],
                    |r| r.get(0),
                )
                .unwrap();
            assert!(distance.abs() < 1e-6);
            assert_eq!(count(&db, &table), 1);
        }
        assert!(db
            .store_generation_embeddings(&lease, &original, &["late".into()], &[vector(dimensions, 0)])
            .is_err());
        drop(db);
        let reopened = Database::new(dir.path().to_path_buf()).unwrap();
        assert_eq!(
            reopened.embedding_generation_control().unwrap().active_id,
            Some(lease.generation_id)
        );
        assert_eq!(
            reopened
                .embedding_generation_profile(lease.generation_id)
                .unwrap()
                .dimensions,
            dimensions
        );
        assert_eq!(
            reopened.embedding_generation_status(lease.generation_id).unwrap().state,
            "active"
        );
    }
}

#[test]
fn incomplete_or_failed_candidate_cannot_replace_active_and_retry_can_complete() {
    let (_dir, db) = file_db();
    seed(&db.connection());
    let first = prepare(&db, 768);
    populate(&db, &first, 768);
    activate(&db, &first);
    let next = prepare(&db, 4096);
    assert!(db.validate_embedding_generation(&next).is_err());
    let s = source(&db, next.generation_id, Corpus::Emails);
    db.fail_generation_source(&next, &s, "synthetic provider failure")
        .unwrap();
    assert_eq!(db.embedding_generation_status(next.generation_id).unwrap().failed, 1);
    assert!(db.validate_embedding_generation(&next).is_err());
    assert_eq!(
        db.embedding_generation_control().unwrap().active_id,
        Some(first.generation_id)
    );
    populate(&db, &next, 4096);
    activate(&db, &next);
    assert_eq!(
        db.embedding_generation_control().unwrap().retained_id,
        Some(first.generation_id)
    );
}

#[test]
fn source_store_failure_rolls_back_vectors_and_metadata_after_reopen() {
    let (dir, db) = file_db();
    seed(&db.connection());
    let lease = prepare(&db, 768);
    let s = source(&db, lease.generation_id, Corpus::Emails);
    db.store_generation_embeddings(&lease, &s, &["old".into()], &[vector(768, 0)])
        .unwrap();
    db.connection().execute_batch("CREATE TEMP TRIGGER reject_second_chunk BEFORE INSERT ON embedding_generation_chunks WHEN new.chunk_index=1 BEGIN SELECT RAISE(ABORT,'synthetic chunk failure'); END;").unwrap();
    assert!(db
        .store_generation_embeddings(
            &lease,
            &s,
            &["new0".into(), "new1".into()],
            &[vector(768, 1), vector(768, 1)]
        )
        .is_err());
    drop(db);
    let db = Database::new(dir.path().to_path_buf()).unwrap();
    let table = vector_table(lease.generation_id, Corpus::Emails).unwrap();
    assert_eq!(count(&db, &table), 1);
    let hash: String = db
        .reader()
        .query_row("SELECT input_hash FROM embedding_generation_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(hash, input_hash(b"old"));
    let bytes: Vec<u8> = db
        .reader()
        .query_row(&format!("SELECT embedding FROM {table}"), [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        bytes,
        vector(768, 0)
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>()
    );
}

#[test]
fn extension_create_write_drop_and_explicit_rollback_survive_reopen() {
    let (dir, db) = file_db();
    let bytes: Vec<u8> = vector(4096, 0).into_iter().flat_map(f32::to_le_bytes).collect();
    {
        let mut conn = db.connection();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).unwrap();
        tx.execute_batch("CREATE VIRTUAL TABLE synthetic_vec USING vec0(embedding float[4096] distance_metric=cosine)")
            .unwrap();
        tx.execute("INSERT INTO synthetic_vec(rowid,embedding) VALUES(1,?1)", [&bytes])
            .unwrap();
        tx.rollback().unwrap();
        assert!(!conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='synthetic_vec')",
                [],
                |r| r.get::<_, bool>(0)
            )
            .unwrap());
        conn.execute_batch(
            "CREATE VIRTUAL TABLE synthetic_vec USING vec0(embedding float[4096] distance_metric=cosine)",
        )
        .unwrap();
        conn.execute("INSERT INTO synthetic_vec(rowid,embedding) VALUES(7,?1)", [&bytes])
            .unwrap();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).unwrap();
        tx.execute_batch("DELETE FROM synthetic_vec; DROP TABLE synthetic_vec;")
            .unwrap();
        tx.rollback().unwrap();
    }
    drop(db);
    let db = Database::new(dir.path().to_path_buf()).unwrap();
    assert_eq!(count(&db, "synthetic_vec"), 1);
    let actual: Vec<u8> = db
        .reader()
        .query_row("SELECT embedding FROM synthetic_vec WHERE rowid=7", [], |r| r.get(0))
        .unwrap();
    assert_eq!(actual, bytes);
}

#[test]
fn activation_failure_is_atomic_and_source_watermark_blocks_late_changes() {
    let (_dir, db) = file_db();
    seed(&db.connection());
    let first = prepare(&db, 768);
    populate(&db, &first, 768);
    activate(&db, &first);
    let next = prepare(&db, 2560);
    populate(&db, &next, 2560);
    db.validate_embedding_generation(&next).unwrap();
    db.connection().execute_batch("CREATE TEMP TRIGGER reject_activation BEFORE UPDATE OF active_id ON embedding_control BEGIN SELECT RAISE(ABORT,'synthetic activation failure'); END;").unwrap();
    assert!(db.activate_embedding_generation(&next).is_err());
    assert_eq!(
        db.embedding_generation_status(first.generation_id).unwrap().state,
        "active"
    );
    assert_eq!(
        db.embedding_generation_status(next.generation_id).unwrap().state,
        "validating"
    );
    assert_eq!(
        db.embedding_generation_control().unwrap().active_id,
        Some(first.generation_id)
    );
    db.connection().execute_batch("DROP TRIGGER reject_activation; INSERT INTO help_doc_chunks(chunk_id,lang,page,section_index,page_title,heading,content,created_at) VALUES('help-new','en','fixture',1,'Fixture','New','New text',1);").unwrap();
    assert!(db.activate_embedding_generation(&next).is_err());
    populate(&db, &next, 2560);
    activate(&db, &next);
}

#[test]
fn pause_resume_and_expired_restart_fence_workers_without_restarting_processing() {
    let (dir, db) = file_db();
    seed(&db.connection());
    let lease = prepare(&db, 768);
    let s = source(&db, lease.generation_id, Corpus::Emails);
    db.store_generation_embeddings(&lease, &s, &["saved".into()], &[vector(768, 0)])
        .unwrap();
    db.stop_embedding_generation(lease.generation_id, false, None).unwrap();
    assert!(db
        .store_generation_embeddings(&lease, &s, &["late".into()], &[vector(768, 0)])
        .is_err());
    let rev = db.embedding_generation_control().unwrap().revision;
    assert!(db
        .resume_embedding_generation(lease.generation_id, "different-profile", rev)
        .is_err());
    let resumed = db
        .resume_embedding_generation(lease.generation_id, &lease.fingerprint, rev)
        .unwrap();
    assert!(resumed.epoch > lease.epoch);
    assert_ne!(resumed.owner, lease.owner);
    assert_eq!(
        db.embedding_generation_status(lease.generation_id).unwrap().completed,
        1
    );
    db.connection()
        .execute(
            "UPDATE embedding_generations SET lease_until=0 WHERE id=?1",
            [lease.generation_id],
        )
        .unwrap();
    drop(db);
    let db = Database::new(dir.path().to_path_buf()).unwrap();
    assert_eq!(
        db.embedding_generation_status(lease.generation_id).unwrap().state,
        "interrupted"
    );
    assert!(db.heartbeat_embedding_generation(&resumed).is_err());
    assert_eq!(
        db.embedding_generation_status(lease.generation_id).unwrap().completed,
        1
    );
    assert_eq!(
        db.embedding_generation_control().unwrap().pending_id,
        Some(lease.generation_id)
    );
}

#[test]
fn second_process_cannot_cancel_healthy_owner_or_create_competing_candidate() {
    let (dir, db) = file_db();
    seed(&db.connection());
    let lease = prepare(&db, 768);
    let other = Database::new(dir.path().to_path_buf()).unwrap();
    assert_eq!(
        other.embedding_generation_status(lease.generation_id).unwrap().state,
        "building"
    );
    assert!(other
        .prepare_embedding_generation(
            &profile(4096),
            &Corpus::ALL,
            other.embedding_generation_control().unwrap().revision
        )
        .is_err());
    db.heartbeat_embedding_generation(&lease).unwrap();
    // A stopped candidate is claimed once, even by separate DB connections.
    db.stop_embedding_generation(lease.generation_id, false, None).unwrap();
    let revision = db.embedding_generation_control().unwrap().revision;
    let winner = other
        .resume_embedding_generation(lease.generation_id, &lease.fingerprint, revision)
        .unwrap();
    assert!(db
        .resume_embedding_generation(lease.generation_id, &lease.fingerprint, revision)
        .is_err());
    assert!(db.heartbeat_embedding_generation(&lease).is_err());
    other.heartbeat_embedding_generation(&winner).unwrap();
}

#[test]
fn body_edit_delete_and_id_reuse_reject_late_results_and_purge_every_generation() {
    let (_dir, db) = file_db();
    seed(&db.connection());
    let first = prepare(&db, 768);
    populate(&db, &first, 768);
    activate(&db, &first);
    let next = prepare(&db, 4096);
    let stale = source(&db, next.generation_id, Corpus::Emails);
    populate(&db, &next, 4096);
    db.connection()
        .execute(
            "UPDATE email_bodies SET body='Changed complete input' WHERE email_id='email-a'",
            [],
        )
        .unwrap();
    assert!(db
        .store_generation_embeddings(&next, &stale, &["stale".into()], &[vector(4096, 0)])
        .is_err());
    for generation in [first.generation_id, next.generation_id] {
        assert_eq!(count(&db, &vector_table(generation, Corpus::Emails).unwrap()), 0);
    }
    let fresh = source(&db, next.generation_id, Corpus::Emails);
    assert!(fresh.revision > stale.revision);
    db.store_generation_embeddings(&next, &fresh, &["fresh".into()], &[vector(4096, 0)])
        .unwrap();
    db.connection()
        .execute("DELETE FROM emails WHERE id='email-a'", [])
        .unwrap();
    assert!(db
        .store_generation_embeddings(&next, &fresh, &["late".into()], &[vector(4096, 0)])
        .is_err());
    assert_eq!(
        count(&db, &vector_table(next.generation_id, Corpus::Emails).unwrap()),
        0
    );
    db.connection().execute_batch("INSERT INTO emails(id,account_id,thread_id,subject,sender,sender_email,recipients_json,snippet,timestamp,created_at) VALUES('email-a','account-a','thread-new','Replacement','Fixture','fixture@example.test','[]','new',1,1);").unwrap();
    assert!(source(&db, next.generation_id, Corpus::Emails).revision > fresh.revision);
    assert!(db
        .store_generation_embeddings(&next, &fresh, &["old identity".into()], &[vector(4096, 0)])
        .is_err());
}

#[test]
fn source_transaction_failure_restores_source_ledger_and_managed_vectors() {
    let (_dir, db) = file_db();
    seed(&db.connection());
    let lease = prepare(&db, 768);
    populate(&db, &lease, 768);
    let before = db.embedding_generation_status(lease.generation_id).unwrap().completed;
    {
        let mut conn = db.connection();
        let tx = conn.transaction().unwrap();
        tx.execute("UPDATE memory_facts SET fact='Changed' WHERE id='fact-a'", [])
            .unwrap();
        assert_eq!(
            tx.query_row("SELECT count(*) FROM vec_memory_facts_g1", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            0
        );
        tx.rollback().unwrap();
    }
    assert_eq!(count(&db, "vec_memory_facts_g1"), 1);
    assert_eq!(
        db.embedding_generation_status(lease.generation_id).unwrap().completed,
        before
    );
    db.validate_embedding_generation(&lease).unwrap();
}

#[test]
fn fact_help_and_account_deletions_clean_pending_active_and_retained_vectors() {
    let (_dir, db) = file_db();
    seed(&db.connection());
    let first = prepare(&db, 768);
    populate(&db, &first, 768);
    activate(&db, &first);
    let second = prepare(&db, 2560);
    populate(&db, &second, 2560);
    activate(&db, &second);
    let third = prepare(&db, 4096);
    populate(&db, &third, 4096);
    db.connection().execute("DELETE FROM help_doc_chunks", []).unwrap();
    db.delete_account("account-a").unwrap();
    for lease in [&first, &second, &third] {
        for corpus in Corpus::ALL {
            assert_eq!(count(&db, &vector_table(lease.generation_id, corpus).unwrap()), 0);
        }
    }
    assert_eq!(count(&db, "embedding_generation_chunks"), 0);
    assert_eq!(count(&db, "embedding_generation_items"), 0);
}

#[test]
fn active_incremental_writes_are_fenced_by_source_and_generation_changes() {
    let (_dir, db) = file_db();
    seed(&db.connection());
    let first = prepare(&db, 768);
    populate(&db, &first, 768);
    activate(&db, &first);
    let active = db.active_embedding_write_lease().unwrap().unwrap();
    db.connection()
        .execute("UPDATE memory_facts SET fact='New fact' WHERE id='fact-a'", [])
        .unwrap();
    let s = source(&db, first.generation_id, Corpus::MemoryFacts);
    db.store_active_generation_embeddings(&active, &s, &["new fact input".into()], &[vector(768, 1)])
        .unwrap();
    assert_eq!(count(&db, "vec_memory_facts_g1"), 1);
    let next = prepare(&db, 4096);
    populate(&db, &next, 4096);
    activate(&db, &next);
    assert!(db
        .store_active_generation_embeddings(&active, &s, &["late".into()], &[vector(768, 0)])
        .is_err());
    assert_eq!(
        db.embedding_generation_status(first.generation_id).unwrap().state,
        "retained"
    );
}

#[test]
fn rollback_catches_up_and_restores_profile_without_restoring_revoked_permissions() {
    let (_dir, db) = file_db();
    seed(&db.connection());
    let first = prepare(&db, 768);
    populate(&db, &first, 768);
    activate(&db, &first);
    let second = prepare(&db, 4096);
    populate(&db, &second, 4096);
    activate(&db, &second);
    db.set_preference("ai_enabled", "false").unwrap();
    db.connection()
        .execute("UPDATE emails SET subject='Changed' WHERE id='email-a'", [])
        .unwrap();
    let rev = db.embedding_generation_control().unwrap().revision;
    assert!(db.prepare_embedding_rollback(rev, "wrong provider identity").is_err());
    let rollback = db.prepare_embedding_rollback(rev, &first.fingerprint).unwrap();
    assert!(db.validate_embedding_generation(&rollback).is_err());
    populate(&db, &rollback, 768);
    activate(&db, &rollback);
    let current = db.embedding_generation_control().unwrap();
    assert_eq!(current.active_id, Some(first.generation_id));
    assert_eq!(current.retained_id, Some(second.generation_id));
    assert_eq!(
        db.embedding_generation_profile(current.active_id.unwrap())
            .unwrap()
            .dimensions,
        768
    );
    assert_eq!(db.get_preference("ai_enabled").unwrap().as_deref(), Some("false"));
}

#[test]
fn real_embedding_preference_writers_fence_validated_candidates() {
    for key in [
        "ai_embedding_model",
        "ai_provider",
        "openai_compatible_base_url",
        "openai_compatible_embedding_source",
        "openai_compatible_embedding_validated",
        "openai_compatible_embedding_dimensions",
        "openrouter_embedding_validated_model",
        "openrouter_embedding_dimensions",
        "openrouter_zdr",
        "ai_enabled",
        "memory_enabled",
        "help_docs_enabled",
    ] {
        let (_dir, db) = file_db();
        let lease = prepare(&db, 768);
        db.validate_embedding_generation(&lease).unwrap();
        db.set_preference(key, "synthetic-change").unwrap();
        assert!(db.activate_embedding_generation(&lease).is_err(), "unfenced key {key}");
    }
}

#[test]
fn copies_require_identical_fingerprint_and_current_verified_source() {
    let (_dir, db) = file_db();
    seed(&db.connection());
    let first = prepare(&db, 768);
    populate(&db, &first, 768);
    activate(&db, &first);
    let same = prepare(&db, 768);
    for corpus in Corpus::ALL {
        let s = source(&db, same.generation_id, corpus);
        db.copy_generation_source(&same, first.generation_id, &s).unwrap();
    }
    activate(&db, &same);
    let different = prepare(&db, 4096);
    let s = source(&db, different.generation_id, Corpus::Emails);
    assert!(db.copy_generation_source(&different, first.generation_id, &s).is_err());
    assert!(db.copy_generation_source(&different, 0, &s).is_err());
    db.stop_embedding_generation(different.generation_id, false, None)
        .unwrap();
    db.discard_embedding_generation(different.generation_id).unwrap();
    let mut incompatible = profile(768);
    incompatible.model = "different-same-dimension-model".into();
    let next = db
        .prepare_embedding_generation(
            &incompatible,
            &Corpus::ALL,
            db.embedding_generation_control().unwrap().revision,
        )
        .unwrap();
    let s = source(&db, next.generation_id, Corpus::Emails);
    assert!(db.copy_generation_source(&next, same.generation_id, &s).is_err());
}

#[test]
fn discard_is_atomic_and_never_drops_active_or_legacy_data() {
    let (dir, db) = file_db();
    seed(&db.connection());
    let first = prepare(&db, 768);
    populate(&db, &first, 768);
    activate(&db, &first);
    assert!(db.discard_embedding_generation(first.generation_id).is_err());
    let next = prepare(&db, 4096);
    db.stop_embedding_generation(next.generation_id, false, None).unwrap();
    db.connection().execute_batch("CREATE TEMP TRIGGER reject_generation_delete BEFORE DELETE ON embedding_generations BEGIN SELECT RAISE(ABORT,'synthetic cleanup failure'); END;").unwrap();
    assert!(db.discard_embedding_generation(next.generation_id).is_err());
    for corpus in Corpus::ALL {
        assert_eq!(count(&db, &vector_table(next.generation_id, corpus).unwrap()), 0);
    }
    drop(db);
    let db = Database::new(dir.path().to_path_buf()).unwrap();
    assert_eq!(
        db.embedding_generation_control().unwrap().pending_id,
        Some(next.generation_id)
    );
    db.discard_embedding_generation(next.generation_id).unwrap();
    assert_eq!(db.embedding_generation_control().unwrap().pending_id, None);
    assert_eq!(count(&db, "vec_emails_g1"), 1);
    assert!(db.embedding_generation_status(next.generation_id).is_err());
}

#[test]
fn missing_active_table_enters_repair_and_is_not_recreated_at_startup() {
    let (dir, db) = file_db();
    seed(&db.connection());
    let lease = prepare(&db, 4096);
    populate(&db, &lease, 4096);
    activate(&db, &lease);
    db.connection().execute_batch("DROP TABLE vec_emails_g1").unwrap();
    drop(db);
    let db = Database::new(dir.path().to_path_buf()).unwrap();
    assert_eq!(
        db.embedding_generation_status(lease.generation_id).unwrap().state,
        "repair"
    );
    assert!(db.active_embedding_write_lease().unwrap().is_none());
    let exists: bool = db
        .reader()
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='vec_emails_g1')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(!exists);
    db.connection()
        .execute("UPDATE emails SET subject='Still editable' WHERE id='email-a'", [])
        .unwrap();
    assert!(!db
        .fts_search("Fixture", Some("account-a"), None, 10)
        .unwrap()
        .is_empty());
}

#[test]
fn readers_keep_old_generation_snapshot_during_atomic_activation() {
    let (dir, db) = file_db();
    seed(&db.connection());
    let first = prepare(&db, 768);
    populate(&db, &first, 768);
    activate(&db, &first);
    let next = prepare(&db, 4096);
    populate(&db, &next, 4096);
    db.validate_embedding_generation(&next).unwrap();
    let other = Database::new(dir.path().to_path_buf()).unwrap();
    let mut reader = other.reader();
    let snapshot = reader.transaction().unwrap();
    assert_eq!(control(&snapshot).unwrap().active_id, Some(first.generation_id));
    db.activate_embedding_generation(&next).unwrap();
    assert_eq!(control(&snapshot).unwrap().active_id, Some(first.generation_id));
    assert_eq!(
        snapshot
            .query_row("SELECT count(*) FROM vec_emails_g1", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    snapshot.commit().unwrap();
    assert_eq!(control(&reader).unwrap().active_id, Some(next.generation_id));
}

#[test]
fn explicitly_disabled_corpora_do_not_block_email_generation_activation() {
    let (_dir, db) = file_db();
    seed(&db.connection());
    let lease = db
        .prepare_embedding_generation(
            &profile(768),
            &[Corpus::Emails],
            db.embedding_generation_control().unwrap().revision,
        )
        .unwrap();
    assert!(db
        .embedding_generation_sources(lease.generation_id, Corpus::HelpDocs, 10)
        .unwrap()
        .is_empty());
    populate(&db, &lease, 768);
    activate(&db, &lease);
    assert_eq!(count(&db, "vec_emails_g1"), 1);
    assert_eq!(count(&db, "vec_memory_facts_g1"), 0);
    assert_eq!(count(&db, "vec_help_docs_g1"), 0);
}

#[test]
fn no_op_sync_source_and_settings_writes_keep_revisions_and_vectors() {
    let (_dir, db) = file_db();
    seed(&db.connection());
    let lease = prepare(&db, 768);
    populate(&db, &lease, 768);
    activate(&db, &lease);
    db.set_preference("ai_embedding_model", "fixture-model").unwrap();
    let active = db.active_embedding_write_lease().unwrap().unwrap();
    db.refresh_active_embedding_scope(&active).unwrap();
    let before: i64 = db
        .reader()
        .query_row("SELECT source_version FROM embedding_control", [], |r| r.get(0))
        .unwrap();
    let revision = db.embedding_generation_control().unwrap().revision;
    db.connection()
        .execute_batch(
            "UPDATE emails SET subject=subject,sender=sender,snippet=snippet,category=category,mailbox=mailbox;
        UPDATE email_bodies SET body=body;
        UPDATE memory_facts SET fact=fact,status=status;
        UPDATE help_doc_chunks SET content=content,heading=heading;
        UPDATE accounts SET enabled=enabled;",
        )
        .unwrap();
    let mut email = db.get_email("email-a").unwrap().unwrap();
    email.body = db.get_email_body("email-a").unwrap();
    db.insert_email(&email).unwrap();
    db.insert_emails_batch(&[email]).unwrap();
    db.set_preference("ai_embedding_model", "fixture-model").unwrap();
    assert_eq!(db.embedding_generation_control().unwrap().revision, revision);
    let after: i64 = db
        .reader()
        .query_row("SELECT source_version FROM embedding_control", [], |r| r.get(0))
        .unwrap();
    assert_eq!(before, after);
    for corpus in Corpus::ALL {
        assert_eq!(count(&db, &vector_table(lease.generation_id, corpus).unwrap()), 1);
    }
}

#[test]
fn frozen_scope_preserves_category_and_age_policy_and_refreshes_explicitly() {
    let (_dir, db) = file_db();
    seed(&db.connection());
    db.connection().execute_batch("INSERT INTO emails(id,account_id,thread_id,subject,sender,sender_email,recipients_json,snippet,timestamp,created_at)
       VALUES('email-promotions','account-a','thread-p','Promotion','Fixture','fixture@example.test','[]','promo',1,1),
             ('email-recent','account-a','thread-r','Recent','Fixture','fixture@example.test','[]','recent',unixepoch(),1);
       UPDATE emails SET category='promotions' WHERE id='email-promotions';").unwrap();
    db.set_preference("ai_max_email_count", "0").unwrap();
    db.set_preference("ai_max_email_age_days", "30").unwrap();
    let lease = prepare(&db, 768);
    let ids: Vec<String> = db
        .embedding_generation_sources(lease.generation_id, Corpus::Emails, 100)
        .unwrap()
        .into_iter()
        .map(|s| s.source_id)
        .collect();
    assert_eq!(ids, vec!["email-recent"]);
    populate(&db, &lease, 768);
    activate(&db, &lease);
    // Scope is persisted independently from the embedding-space identity.
    assert_eq!(
        db.embedding_generation_profile(lease.generation_id)
            .unwrap()
            .fingerprint()
            .unwrap(),
        lease.fingerprint
    );
    db.set_preference("ai_max_email_age_days", "0").unwrap();
    db.set_preference("embeddings_categories:account-a", "[]").unwrap();
    let active = db.active_embedding_write_lease().unwrap().unwrap();
    let s = SourceRevision {
        corpus: Corpus::Emails,
        source_id: "email-recent".into(),
        account_id: Some("account-a".into()),
        revision: 1,
    };
    assert!(db
        .store_active_generation_embeddings(&active, &s, &["input".into()], &[vector(768, 0)])
        .is_err());
    db.refresh_active_embedding_scope(&active).unwrap();
    assert_eq!(
        db.embedding_generation_sources(lease.generation_id, Corpus::Emails, 100)
            .unwrap()
            .len(),
        2
    );
    db.set_preference("embeddings_categories:account-a", "[\"primary\"]")
        .unwrap();
    let next = prepare(&db, 768);
    populate(&db, &next, 768);
    activate(&db, &next);
    assert_eq!(
        count(&db, &vector_table(next.generation_id, Corpus::Emails).unwrap()),
        2
    );
    assert_eq!(
        db.embedding_generation_profile(next.generation_id)
            .unwrap()
            .fingerprint()
            .unwrap(),
        lease.fingerprint
    );
}

#[test]
fn simultaneous_connections_claim_only_one_pending_generation() {
    use std::sync::{Arc, Barrier};
    let (dir, db) = file_db();
    let other = Database::new(dir.path().to_path_buf()).unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let gate = barrier.clone();
    let worker = std::thread::spawn(move || {
        gate.wait();
        other
            .prepare_embedding_generation(&profile(4096), &Corpus::ALL, 0)
            .is_ok()
    });
    barrier.wait();
    let parent = db.prepare_embedding_generation(&profile(768), &Corpus::ALL, 0).is_ok();
    assert_ne!(parent, worker.join().unwrap());
    assert_eq!(count(&db, "embedding_generations"), 2);
}

#[test]
fn cancellation_from_another_connection_rejects_in_flight_inference_result() {
    use std::sync::mpsc;
    let (dir, db) = file_db();
    seed(&db.connection());
    let lease = prepare(&db, 768);
    let s = source(&db, lease.generation_id, Corpus::Emails);
    let other = Database::new(dir.path().to_path_buf()).unwrap();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (continue_tx, continue_rx) = mpsc::channel();
    let worker_lease = lease.clone();
    let worker = std::thread::spawn(move || {
        ready_tx.send(()).unwrap();
        continue_rx.recv().unwrap();
        other
            .store_generation_embeddings(&worker_lease, &s, &["late result".into()], &[vector(768, 0)])
            .is_err()
    });
    ready_rx.recv().unwrap();
    db.stop_embedding_generation(lease.generation_id, false, None).unwrap();
    continue_tx.send(()).unwrap();
    assert!(worker.join().unwrap());
    assert_eq!(count(&db, "vec_emails_g1"), 0);
    assert_eq!(
        db.embedding_generation_status(lease.generation_id).unwrap().completed,
        0
    );
}

#[test]
fn missing_partial_copy_and_orphan_metadata_fail_validation() {
    let (_dir, db) = file_db();
    seed(&db.connection());
    let first = prepare(&db, 768);
    let s = source(&db, first.generation_id, Corpus::Emails);
    db.store_generation_embeddings(
        &first,
        &s,
        &["one".into(), "two".into()],
        &[vector(768, 0), vector(768, 1)],
    )
    .unwrap();
    populate(&db, &first, 768);
    activate(&db, &first);
    let next = prepare(&db, 768);
    let s = source(&db, next.generation_id, Corpus::Emails);
    db.connection().execute("DELETE FROM vec_emails_g1 WHERE rowid=(SELECT min(id) FROM embedding_generation_chunks WHERE generation_id=1 AND corpus='emails')",[]).unwrap();
    assert!(db.copy_generation_source(&next, first.generation_id, &s).is_err());
    populate(&db, &next, 768);
    let blob: Vec<u8> = vector(768, 0).into_iter().flat_map(f32::to_le_bytes).collect();
    db.connection()
        .execute("INSERT INTO vec_emails_g2(rowid,embedding) VALUES(99999,?1)", [blob])
        .unwrap();
    assert!(db.validate_embedding_generation(&next).is_err());
}

#[test]
fn disabling_help_on_resume_reconciles_coverage_without_embedding_disabled_sources() {
    let (_dir, db) = file_db();
    seed(&db.connection());
    let lease = prepare(&db, 768);
    db.stop_embedding_generation(lease.generation_id, false, None).unwrap();
    db.set_preference("help_docs_enabled", "false").unwrap();
    let resumed = db
        .resume_embedding_generation(
            lease.generation_id,
            &lease.fingerprint,
            db.embedding_generation_control().unwrap().revision,
        )
        .unwrap();
    assert!(db
        .embedding_generation_sources(lease.generation_id, Corpus::HelpDocs, 10)
        .unwrap()
        .is_empty());
    populate(&db, &resumed, 768);
    activate(&db, &resumed);
    assert_eq!(count(&db, "vec_help_docs_g1"), 0);
}

#[test]
fn failed_additive_migration_rolls_back_without_touching_legacy_data() {
    drop(Database::new_for_testing().unwrap());
    let dir = tempfile::tempdir().unwrap();
    let mut conn = Connection::open(dir.path().join("emailops.db")).unwrap();
    crate::db::embedded::migrations::runner()
        .set_target(refinery::Target::Version(40))
        .run(&mut conn)
        .unwrap();
    seed(&conn);
    conn.execute_batch("INSERT INTO user_preferences(key,value) VALUES('unrelated-setting','preserve'); CREATE TABLE embedding_control(incompatible_column INTEGER);").unwrap();
    drop(conn);
    assert!(Database::new(dir.path().to_path_buf()).is_err());
    let conn = Connection::open(dir.path().join("emailops.db")).unwrap();
    let generations_exist: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='embedding_generations')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(!generations_exist);
    for table in ["emails", "memory_facts", "help_doc_chunks"] {
        assert_eq!(
            conn.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get::<_, i64>(0))
                .unwrap(),
            1
        );
    }
    let setting: String = conn
        .query_row(
            "SELECT value FROM user_preferences WHERE key='unrelated-setting'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(setting, "preserve");
    let version: i64 = conn
        .query_row("SELECT max(version) FROM refinery_schema_history", [], |r| r.get(0))
        .unwrap();
    assert_eq!(version, 40);
    conn.execute_batch("DROP TABLE embedding_control").unwrap();
    drop(conn);
    assert!(Database::new(dir.path().to_path_buf()).is_ok());
}

#[test]
fn failed_source_statement_restores_vector_cleanup_even_if_outer_transaction_commits() {
    let (_dir, db) = file_db();
    seed(&db.connection());
    let lease = prepare(&db, 768);
    populate(&db, &lease, 768);
    db.connection().execute_batch("CREATE TEMP TRIGGER reject_source_ledger AFTER UPDATE ON embedding_sources BEGIN SELECT RAISE(ABORT,'synthetic ledger rejection'); END;").unwrap();
    {
        let mut conn = db.connection();
        let tx = conn.transaction().unwrap();
        assert!(tx
            .execute("UPDATE emails SET subject='Rejected edit' WHERE id='email-a'", [])
            .is_err());
        tx.commit().unwrap();
    }
    assert_eq!(count(&db, "vec_emails_g1"), 1);
    assert_eq!(
        db.embedding_generation_status(lease.generation_id).unwrap().completed,
        3
    );
    db.validate_embedding_generation(&lease).unwrap();
}

#[test]
fn schema_changes_after_validation_block_activation_without_a_corpus_scan() {
    let (_dir, db) = file_db();
    seed(&db.connection());
    let lease = prepare(&db, 4096);
    populate(&db, &lease, 4096);
    db.validate_embedding_generation(&lease).unwrap();
    db.connection().execute_batch("DROP TABLE vec_emails_g1").unwrap();
    assert!(db.activate_embedding_generation(&lease).is_err());
    assert_eq!(db.embedding_generation_control().unwrap().active_id, None);
}

#[test]
fn junk_feedback_invalidates_all_managed_generations_and_fences_late_results() {
    let (_dir, db) = file_db();
    seed(&db.connection());
    let retained = prepare(&db, 768);
    populate(&db, &retained, 768);
    activate(&db, &retained);
    let active = prepare(&db, 2560);
    populate(&db, &active, 2560);
    activate(&db, &active);
    let pending = prepare(&db, 4096);
    let stale = source(&db, pending.generation_id, Corpus::Emails);
    populate(&db, &pending, 4096);

    db.set_junk_override("email-a", "account-a", Some("junk"), now())
        .unwrap();
    for generation in [retained.generation_id, active.generation_id, pending.generation_id] {
        assert_eq!(count(&db, &vector_table(generation, Corpus::Emails).unwrap()), 0);
        assert!(db
            .embedding_generation_sources(generation, Corpus::Emails, 10)
            .unwrap()
            .is_empty());
    }
    assert!(db
        .store_generation_embeddings(&pending, &stale, &["late result".into()], &[vector(4096, 0)])
        .is_err());

    let marked_revision: i64 = db
        .reader()
        .query_row(
            "SELECT revision FROM embedding_sources WHERE corpus='emails' AND source_id='email-a'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    db.set_junk_override("email-a", "account-a", Some("junk"), now())
        .unwrap();
    let repeated_revision: i64 = db
        .reader()
        .query_row(
            "SELECT revision FROM embedding_sources WHERE corpus='emails' AND source_id='email-a'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        marked_revision, repeated_revision,
        "repeated feedback is not a source mutation"
    );

    db.set_junk_override("email-a", "account-a", Some("not_junk"), now())
        .unwrap();
    let restored = source(&db, pending.generation_id, Corpus::Emails);
    assert!(restored.revision > stale.revision);
    db.store_generation_embeddings(&pending, &restored, &["restored input".into()], &[vector(4096, 0)])
        .unwrap();
    let active_lease = db.active_embedding_write_lease().unwrap().unwrap();
    db.refresh_active_embedding_scope(&active_lease).unwrap();
    db.store_active_generation_embeddings(&active_lease, &restored, &["restored input".into()], &[vector(2560, 0)])
        .unwrap();
    assert_eq!(
        count(&db, &vector_table(active.generation_id, Corpus::Emails).unwrap()),
        1
    );
    assert_eq!(
        count(&db, &vector_table(pending.generation_id, Corpus::Emails).unwrap()),
        1
    );
    assert_eq!(
        count(&db, &vector_table(retained.generation_id, Corpus::Emails).unwrap()),
        0
    );
}
