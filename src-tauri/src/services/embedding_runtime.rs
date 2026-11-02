//! Managed embedding execution shared by email, memory and bundled help.
//! No provider request runs inside a database transaction.
use super::{ai::AiService, embedding_session::EmbeddingSession, task_queue};
use crate::db::{
    embedding_generations::{
        BuildLease, Corpus, DimensionPolicy, EmbeddingProfile, EmbeddingSourceSnapshot, SourceContent,
        TypedQueryEmbedding,
    },
    Database,
};
use crate::models::error::{AppError, Result};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};

pub const QUERY_FORMAT: &str = "raw-query-v1";
pub const DOCUMENT_FORMAT: &str = "emailops-documents-v1";
pub const PREPROCESSING: &str = "emailops-chunk-v1";

fn check_format(profile: &EmbeddingProfile) -> Result<()> {
    if profile.query_format != QUERY_FORMAT
        || profile.document_format != DOCUMENT_FORMAT
        || profile.preprocessing != PREPROCESSING
        || profile.normalization != "none"
    {
        return Err(AppError::InvalidInput(
            "Unsupported embedding formatting; rebuild with the current formatting version".into(),
        ));
    }
    Ok(())
}

/// A backend preview is fenced against every embedding-settings writer. This
/// object contains no credential. Commit re-probes before creating any tables.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(
    feature = "ts",
    derive(ts_rs::TS),
    ts(export, export_to = "../../src/types/generated/")
)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddingBuildPreview {
    #[cfg_attr(feature = "ts", ts(type = "number"))]
    pub expected_revision: i64,
    pub profile: EmbeddingProfile,
    pub enabled_corpora: Vec<Corpus>,
    pub opaque_identity: bool,
    /// Binds proposed non-secret settings as well as the embedding profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settings_fingerprint: Option<String>,
}

pub async fn preview_build(
    db: &Arc<Database>,
    candidate: &EmbeddingProfile,
    enabled_corpora: &[Corpus],
) -> Result<EmbeddingBuildPreview> {
    check_format(candidate)?;
    let expected_revision = db.embedding_generation_control()?.revision;
    let profile = AiService::probe_embedding_profile(db.clone(), candidate).await?;
    if db.embedding_generation_control()?.revision != expected_revision {
        return Err(AppError::InvalidInput(
            "Embedding settings changed during probe; preview again".into(),
        ));
    }
    Ok(EmbeddingBuildPreview {
        expected_revision,
        opaque_identity: super::embedding_session::has_opaque_identity(&profile),
        profile,
        enabled_corpora: enabled_corpora.to_vec(),
        settings_fingerprint: None,
    })
}

pub async fn commit_build(db: &Arc<Database>, preview: &EmbeddingBuildPreview) -> Result<BuildLease> {
    if preview.settings_fingerprint.is_some() {
        return Err(AppError::InvalidInput(
            "A settings preview must use the atomic settings/build commit".into(),
        ));
    }
    check_format(&preview.profile)?;
    let verified = AiService::probe_embedding_profile(db.clone(), &preview.profile).await?;
    if verified.fingerprint()? != preview.profile.fingerprint()? {
        return Err(AppError::InvalidInput(
            "Embedding provider changed since preview; probe again".into(),
        ));
    }
    db.prepare_embedding_generation(&verified, &preview.enabled_corpora, preview.expected_revision)
}

/// Settings UI entry point: validate/probe proposed settings without publishing
/// their endpoint, key or model. The response contains no credentials. Commit
/// accepts the settings again and revalidates them against this summary/fence.
pub async fn preview_settings_build(
    db: &Arc<Database>,
    input: super::ai::AiSettingsInput,
    policy: DimensionPolicy,
    enabled_corpora: &[Corpus],
) -> Result<EmbeddingBuildPreview> {
    let expected_revision = db.embedding_generation_control()?.revision;
    let prepared = AiService::prepare_embedding_settings(db, input, policy, None).await?;
    check_format(&prepared.profile)?;
    if db.embedding_generation_control()?.revision != expected_revision {
        return Err(AppError::InvalidInput(
            "AI settings changed during preview; preview again".into(),
        ));
    }
    let settings_fingerprint = Some(settings_plan_fingerprint(&prepared.preference_writes)?);
    Ok(EmbeddingBuildPreview {
        expected_revision,
        opaque_identity: super::embedding_session::has_opaque_identity(&prepared.profile),
        profile: prepared.profile,
        enabled_corpora: enabled_corpora.to_vec(),
        settings_fingerprint,
    })
}

/// Publish all proposed preferences, current credential references and the
/// replacement job atomically. A keychain cannot join SQLite's transaction:
/// versioned new secrets are staged first and remain unreachable on DB failure.
/// Existing keys are never overwritten/deleted to prepare this transaction.
pub async fn commit_settings_build(
    db: &Arc<Database>,
    input: super::ai::AiSettingsInput,
    policy: DimensionPolicy,
    preview: &EmbeddingBuildPreview,
) -> Result<BuildLease> {
    if preview.settings_fingerprint.is_none() {
        return Err(AppError::InvalidInput(
            "Preview the proposed AI settings before committing them".into(),
        ));
    }
    let control = db.embedding_generation_control()?;
    if control.revision != preview.expected_revision || control.pending_id.is_some() {
        return Err(AppError::InvalidInput(
            "AI settings changed or a build is pending; preview again".into(),
        ));
    }
    let prepared = AiService::prepare_embedding_settings(db, input, policy, Some(&preview.profile)).await?;
    check_format(&prepared.profile)?;
    if prepared.profile.fingerprint()? != preview.profile.fingerprint()? {
        return Err(AppError::InvalidInput(
            "Proposed settings or provider changed since preview; preview again".into(),
        ));
    }
    let proposed_settings_fingerprint = settings_plan_fingerprint(&prepared.preference_writes)?;
    if preview.settings_fingerprint.as_deref() != Some(proposed_settings_fingerprint.as_str()) {
        return Err(AppError::InvalidInput(
            "Proposed AI settings changed since preview; preview again".into(),
        ));
    }
    if db.embedding_generation_control()?.revision != preview.expected_revision {
        return Err(AppError::InvalidInput(
            "AI settings changed during validation; preview again".into(),
        ));
    }
    let mut writes = prepared.preference_writes.clone();
    writes.extend(prepared.stage_credentials(db)?);
    db.prepare_embedding_generation_with_preferences(
        &prepared.profile,
        &preview.enabled_corpora,
        preview.expected_revision,
        &writes,
    )
}

fn settings_plan_fingerprint(writes: &[(String, Option<String>)]) -> Result<String> {
    // Commit timestamps are generated again during validation. They are not a
    // user-proposed setting and must not invalidate an otherwise identical plan.
    let proposed: Vec<_> = writes.iter().filter(|(key, _)| key != "ai_period_start").collect();
    Ok(crate::db::embedding_generations::input_hash(&serde_json::to_vec(
        &proposed,
    )?))
}

pub async fn resume_build(db: &Arc<Database>, generation: i64, expected_revision: i64) -> Result<BuildLease> {
    let profile = db.embedding_generation_profile(generation)?;
    check_format(&profile)?;
    let verified = AiService::probe_embedding_profile(db.clone(), &profile).await?;
    if verified.fingerprint()? != profile.fingerprint()? {
        return Err(AppError::InvalidInput(
            "Provider identity changed; create a replacement index".into(),
        ));
    }
    db.resume_embedding_generation(generation, &verified.fingerprint()?, expected_revision)
}

pub async fn rollback_build(db: &Arc<Database>, expected_revision: i64) -> Result<BuildLease> {
    let retained = db
        .embedding_generation_control()?
        .retained_id
        .ok_or_else(|| AppError::InvalidInput("No retained index is available".into()))?;
    let profile = db.embedding_generation_profile(retained)?;
    check_format(&profile)?;
    let selected_backend = AiService::embedding_provider(db)?;
    let server_endpoint_changed = profile.backend == super::ai_server::PROVIDER_ID
        && super::ai_server::load_server_settings(db)?.base_url != profile.endpoint;
    if selected_backend != profile.backend || server_endpoint_changed {
        return Err(AppError::InvalidInput("Rollback requires explicitly selecting and authorizing the retained provider destination in AI settings. Confirm its disclosure and enter its current key if required; rollback never restores old keys or consent.".into()));
    }
    let verified = AiService::probe_embedding_profile(db.clone(), &profile).await?;
    if verified.fingerprint()? != profile.fingerprint()? {
        return Err(AppError::InvalidInput(
            "Retained provider changed; rollback cannot reuse its vectors".into(),
        ));
    }
    db.prepare_embedding_rollback(expected_revision, &verified.fingerprint()?)
}

fn source_inputs(profile: &EmbeddingProfile, source: &EmbeddingSourceSnapshot) -> Result<Vec<String>> {
    check_format(profile)?;
    Ok(match &source.content {
        SourceContent::Email { subject, sender, body } => {
            super::embeddings::create_embedding_texts(subject, sender, body)
        }
        SourceContent::Text(text) => vec![text.clone()],
    })
}

fn authorize_source(db: &Database, generation: i64, source: &EmbeddingSourceSnapshot) -> Result<()> {
    if task_queue::cancel_requested() {
        return Err(AppError::InvalidInput("Embedding work stopped".into()));
    }
    if !db.is_ai_enabled()? {
        return Err(AppError::AiDisabled);
    }
    let enabled = match source.source.corpus {
        Corpus::Emails => true,
        Corpus::MemoryFacts => db.is_memory_enabled()?,
        Corpus::HelpDocs => db.is_help_docs_enabled()?,
    };
    if !enabled {
        return Err(AppError::InvalidInput("Embedding corpus is disabled".into()));
    }
    db.embedding_source_is_current(generation, &source.source)
}

/// Heartbeats and cancellation continue while inference is in flight. Dropping
/// the request future cancels HTTP I/O; an already accepted remote request may
/// still be processed by its server. Database ownership rejects late commits.
async fn embed_source(
    db: &Arc<Database>,
    session: &EmbeddingSession,
    generation: i64,
    build: Option<&BuildLease>,
    source: &EmbeddingSourceSnapshot,
    inputs: &[String],
) -> Result<Vec<Vec<f32>>> {
    let mut vectors = Vec::with_capacity(inputs.len());
    for input in inputs {
        authorize_source(db, generation, source)?;
        check_request_ownership(db, generation, build)?;
        let request = session.embed_for(source.source.corpus, input);
        tokio::pin!(request);
        let mut heartbeat = tokio::time::interval(Duration::from_secs(10));
        heartbeat.tick().await;
        let mut cancellation = tokio::time::interval(Duration::from_millis(200));
        cancellation.tick().await;
        loop {
            tokio::select! {
                response = &mut request => {vectors.push(response?); break;},
                _ = cancellation.tick() => { authorize_source(db,generation,source)?; check_request_ownership(db,generation,build)?; },
                _ = heartbeat.tick() => {
                    authorize_source(db,generation,source)?;
                    if let Some(lease) = build { db.heartbeat_embedding_generation(lease)?; }
                }
            }
        }
        authorize_source(db, generation, source)?;
    }
    Ok(vectors)
}

fn check_request_ownership(db: &Database, generation: i64, build: Option<&BuildLease>) -> Result<()> {
    if let Some(lease) = build {
        return db.embedding_build_is_current(lease);
    }
    if db
        .active_embedding_write_lease()?
        .is_some_and(|l| l.generation_id == generation)
    {
        return Ok(());
    }
    Err(AppError::InvalidInput(
        "Embedding generation retired during inference".into(),
    ))
}

/// Build every enabled corpus alongside the active index, then validate and
/// switch all three atomically. Failures preserve completed work for resume.
pub async fn run_build(db: &Arc<Database>, lease: &BuildLease) -> Result<u32> {
    run_build_scoped(db, lease, None).await
}

/// Own the durable lease from submission through queue pickup. Queued work is
/// not inference: restart never re-enqueues it, and cancellation pauses it.
pub async fn enqueue_build(db: Arc<Database>, queue: &task_queue::TaskQueue, lease: BuildLease) -> Result<()> {
    enqueue_build_with(
        db,
        queue,
        lease,
        Arc::new(super::clock::SystemClock),
        |db, lease| async move { run_build(&db, &lease).await },
    )
    .await
}

struct QueuedBuildGuard {
    db: Arc<Database>,
    lease: BuildLease,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
}

impl QueuedBuildGuard {
    fn start(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
    }
}

impl Drop for QueuedBuildGuard {
    fn drop(&mut self) {
        self.start();
        if let Err(error) = self.db.finish_embedding_build_if_owned(
            &self.lease,
            false,
            "Embedding queue ownership ended; explicit resume required",
        ) {
            super::logger::log(
                "warn",
                "embeddings",
                format!("Could not pause abandoned embedding queue owner: {error}"),
            );
        }
    }
}

async fn enqueue_build_with<F, Fut>(
    db: Arc<Database>,
    queue: &task_queue::TaskQueue,
    lease: BuildLease,
    clock: Arc<dyn super::clock::Clock>,
    run: F,
) -> Result<()>
where
    F: FnOnce(Arc<Database>, BuildLease) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<u32>> + Send + 'static,
{
    db.heartbeat_embedding_generation_at(&lease, clock.now_secs())?;
    let (stop, mut stopped) = tokio::sync::oneshot::channel();
    let info: Arc<std::sync::Mutex<Option<task_queue::TaskInfo>>> = Arc::new(std::sync::Mutex::new(None));
    let monitor_info = info.clone();
    let monitor_db = db.clone();
    let monitor_lease = lease.clone();
    tokio::spawn(async move {
        let mut cancel = tokio::time::interval(Duration::from_millis(200));
        let mut heartbeat = tokio::time::interval(Duration::from_secs(10));
        cancel.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _=&mut stopped=>break,
                _=cancel.tick()=>{
                    let cancelled=monitor_info.lock().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref().is_some_and(|task|task.cancel_requested());
                    if cancelled {
                        if let Err(error)=monitor_db.finish_embedding_build_if_owned(&monitor_lease,false,"Embedding build cancelled while queued; explicit resume required") {super::logger::log("warn","embeddings",format!("Could not pause cancelled queue owner: {error}"));}
                        break;
                    }
                },
                _=heartbeat.tick()=>{
                    if let Err(error)=monitor_db.heartbeat_embedding_generation_at(&monitor_lease,clock.now_secs()) {
                        super::logger::log("debug","embeddings",format!("Queued embedding worker fenced: {error}"));
                        if let Err(error)=monitor_db.finish_embedding_build_if_owned(&monitor_lease,false,"Embedding queue lease interrupted; explicit resume required") {super::logger::log("warn","embeddings",format!("Could not pause interrupted queue owner: {error}"));}
                        break;
                    }
                }
            }
        }
    });
    let mut guard = QueuedBuildGuard {
        db: db.clone(),
        lease: lease.clone(),
        stop: Some(stop),
    };
    let name = format!("embeddings:rebuild:g{}:e{}", lease.generation_id, lease.epoch);
    let task = queue
        .try_submit_named(&name, async move {
            guard.start();
            if task_queue::cancel_requested() {
                return;
            }
            // A stopped old queue entry must not send requests or alter a resumed epoch.
            if let Err(error) = db.embedding_build_is_current(&lease) {
                super::logger::log(
                    "debug",
                    "embeddings",
                    format!("Queued build no longer owns its generation: {error}"),
                );
                return;
            }
            if let Err(error) = run(db, lease).await {
                super::logger::log("warn", "embeddings", format!("Embedding index build stopped: {error}"));
            }
        })
        .await?;
    *info.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(task);
    Ok(())
}

/// A same-profile account rebuild can copy verified current sources outside
/// that account's email corpus. Profile changes always infer every source.
pub async fn run_build_scoped(db: &Arc<Database>, lease: &BuildLease, rebuild_account: Option<&str>) -> Result<u32> {
    super::logger::log(
        "info",
        "embeddings",
        format!(
            "Building replacement embedding generation {} alongside the active index",
            lease.generation_id
        ),
    );
    let result = run_build_inner(db, lease, rebuild_account).await;
    if result.is_err() {
        let paused =
            task_queue::cancel_requested() || db.embedding_generation_status(lease.generation_id)?.state == "paused";
        // Persist a stable non-sensitive status; provider bodies never enter DB.
        if let Err(error) = db.finish_embedding_build_if_owned(
            lease,
            !paused,
            if paused {
                "Stopped by user"
            } else {
                "Embedding build failed; verify provider and resume"
            },
        ) {
            super::logger::log("debug", "embeddings", format!("Build already fenced: {error}"));
        }
        super::events::emit(
            "embedding-progress",
            super::embeddings::EmbeddingProgress {
                status: if paused { "cancelled" } else { "failed" }.into(),
                current: 0,
                total: 0,
                message: if paused {
                    "Embedding build paused; completed sources are preserved"
                } else {
                    "Embedding build failed; verify provider and resume"
                }
                .into(),
            },
        );
    } else {
        super::logger::log(
            "success",
            "embeddings",
            format!("Activated embedding generation {}", lease.generation_id),
        );
        super::events::emit(
            "embedding-progress",
            super::embeddings::EmbeddingProgress {
                status: "complete".into(),
                current: result.as_ref().copied().unwrap_or_default(),
                total: result.as_ref().copied().unwrap_or_default(),
                message: "Replacement embedding index activated".into(),
            },
        );
    }
    result
}

async fn run_build_inner(db: &Arc<Database>, lease: &BuildLease, rebuild_account: Option<&str>) -> Result<u32> {
    db.heartbeat_embedding_generation(lease)?;
    let profile = db.embedding_generation_profile(lease.generation_id)?;
    check_format(&profile)?;
    // Provider/model load and the neutral identity request can outlive the
    // lease too; keep ownership alive before the first document request.
    let create_session = AiService::embedding_session(db.clone(), &profile);
    tokio::pin!(create_session);
    let mut tick = tokio::time::interval(Duration::from_millis(200));
    let mut last_heartbeat = std::time::Instant::now();
    let session = loop {
        tokio::select! {
            session = &mut create_session => break session?,
            _ = tick.tick() => {
                if task_queue::cancel_requested() {return Err(AppError::InvalidInput("Embedding work stopped".into()));}
                if last_heartbeat.elapsed() >= Duration::from_secs(10) {
                    db.heartbeat_embedding_generation(lease)?;
                    last_heartbeat=std::time::Instant::now();
                }
            }
        }
    };
    run_build_with_session_scoped(db, lease, &session, rebuild_account).await
}

#[cfg(test)]
async fn run_build_with_session(db: &Arc<Database>, lease: &BuildLease, session: &EmbeddingSession) -> Result<u32> {
    run_build_with_session_scoped(db, lease, session, None).await
}

async fn run_build_with_session_scoped(
    db: &Arc<Database>,
    lease: &BuildLease,
    session: &EmbeddingSession,
    rebuild_account: Option<&str>,
) -> Result<u32> {
    let profile = db.embedding_generation_profile(lease.generation_id)?;
    if session.profile().fingerprint()? != lease.fingerprint {
        return Err(AppError::InvalidInput(
            "Embedding session differs from the build profile".into(),
        ));
    }
    let mut done = 0;
    let total = Corpus::ALL
        .into_iter()
        .map(|corpus| db.embedding_pending_count(lease.generation_id, corpus, None))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .sum::<i64>()
        .min(u32::MAX as i64) as u32;
    report_build_progress(0, total);
    for corpus in Corpus::ALL {
        loop {
            db.heartbeat_embedding_generation(lease)?;
            let sources = db.embedding_source_snapshots(lease.generation_id, corpus, None, 32)?;
            if sources.is_empty() {
                break;
            }
            for source in sources {
                if rebuild_account.is_some_and(|account| {
                    corpus != Corpus::Emails || source.source.account_id.as_deref() != Some(account)
                }) {
                    if let Some(active) = db.active_embedding_write_lease()? {
                        if active.fingerprint == lease.fingerprint
                            && db.embedding_source_complete(active.generation_id, &source.source)?
                        {
                            db.copy_generation_source(lease, active.generation_id, &source.source)?;
                            done += 1;
                            report_build_progress(done, total);
                            continue;
                        }
                    }
                }
                let inputs = source_inputs(&profile, &source)?;
                let vectors = match embed_source(db, session, lease.generation_id, Some(lease), &source, &inputs).await
                {
                    Ok(v) => v,
                    Err(error) => {
                        if let Err(store_error) = db.fail_generation_source(
                            lease,
                            &source.source,
                            "Embedding request failed; verify provider, permissions and retry",
                        ) {
                            super::logger::log("debug", "embeddings", format!("Source already fenced: {store_error}"));
                        }
                        return Err(error);
                    }
                };
                db.store_generation_embeddings(lease, &source.source, &inputs, &vectors)?;
                done += 1;
                report_build_progress(done, total);
            }
        }
    }
    // Newly inserted/edited sources between corpus passes prevent validation;
    // caller can explicitly resume after the corpus settles.
    db.heartbeat_embedding_generation(lease)?;
    db.validate_embedding_generation(lease)?;
    db.activate_embedding_generation(lease)?;
    Ok(done)
}

fn report_build_progress(done: u32, total: u32) {
    task_queue::report_progress(done, total);
    super::events::emit(
        "embedding-progress",
        super::embeddings::EmbeddingProgress {
            status: "generating".into(),
            current: done,
            total,
            message: format!("Building replacement index: {done} of {total} sources"),
        },
    );
}

pub async fn update_active(db: &Arc<Database>, corpus: Corpus, account: Option<&str>, limit: usize) -> Result<u32> {
    let Some(lease) = db.active_embedding_write_lease()? else {
        return Ok(0);
    };
    let profile = db.embedding_generation_profile(lease.generation_id)?;
    check_format(&profile)?;
    db.refresh_active_embedding_scope(&lease)?;
    let sources = db.embedding_source_snapshots(lease.generation_id, corpus, account, limit)?;
    if sources.is_empty() {
        return Ok(0);
    }
    let session = AiService::embedding_session(db.clone(), &profile).await?;
    let mut done = 0;
    for source in sources {
        let inputs = source_inputs(&profile, &source)?;
        let vectors = embed_source(db, &session, lease.generation_id, None, &source, &inputs).await?;
        db.store_active_generation_embeddings(&lease, &source.source, &inputs, &vectors)?;
        done += 1;
    }
    Ok(done)
}

pub async fn embed_query(db: &Arc<Database>, input: &str) -> Result<Option<TypedQueryEmbedding>> {
    embed_query_for(db, Corpus::Emails, input).await
}

pub async fn embed_query_for(db: &Arc<Database>, corpus: Corpus, input: &str) -> Result<Option<TypedQueryEmbedding>> {
    let Some(lease) = db.active_embedding_write_lease()? else {
        return Ok(None);
    };
    let profile = db.embedding_generation_profile(lease.generation_id)?;
    check_format(&profile)?;
    AiService::authorize_embedding(db, &profile, corpus)?;
    let session = AiService::embedding_session(db.clone(), &profile).await?;
    let vector = session.embed_for(corpus, input).await?;
    let current = db.active_embedding_write_lease()?;
    if !current.is_some_and(|c| {
        c.generation_id == lease.generation_id
            && c.epoch == lease.epoch
            && c.fingerprint == lease.fingerprint
            && c.config_revision == lease.config_revision
    }) {
        return Err(AppError::InvalidInput(
            "Active embedding index changed during inference; use text search and retry".into(),
        ));
    }
    Ok(Some(TypedQueryEmbedding::new(lease, &profile, input, vector)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ai::provider::FakeAiProvider, db::embedding_generations::DimensionPolicy};

    fn profile(dimensions: usize) -> EmbeddingProfile {
        EmbeddingProfile {
            version: 1,
            backend: "synthetic".into(),
            endpoint: None,
            model: "fixture".into(),
            model_revision: Some("fixture-v1".into()),
            dimension_policy: DimensionPolicy::Explicit(dimensions),
            dimensions,
            query_format: QUERY_FORMAT.into(),
            document_format: DOCUMENT_FORMAT.into(),
            preprocessing: PREPROCESSING.into(),
            normalization: "none".into(),
            encoding: "float32-le".into(),
            distance_metric: "cosine".into(),
        }
    }

    fn fixture() -> Arc<Database> {
        let db = Arc::new(Database::new_for_testing().unwrap());
        db.connection().execute_batch("INSERT INTO accounts(id,provider,email,name,created_at) VALUES('a','gmail','fixture@example.test','Fixture',0);
            INSERT INTO emails(id,account_id,thread_id,subject,sender,sender_email,recipients_json,snippet,timestamp,created_at) VALUES('email','a','thread','Synthetic invoice','Synthetic sender','sender@example.test','[]','Invoice snippet',1,1);
            INSERT INTO email_bodies(email_id,body) VALUES('email','Complete synthetic invoice body');
            INSERT INTO memory_facts(id,account_id,subject_kind,subject_key,fact,created_at,updated_at) VALUES('fact','a','user','self','Synthetic workshop preference',1,1);
            INSERT INTO help_doc_chunks(chunk_id,lang,page,section_index,page_title,heading,content,created_at) VALUES('help','en','fixture',0,'Fixture guide','Fixture heading','Synthetic instructions',1);").unwrap();
        db.set_preference("memory_enabled", "true").unwrap();
        db
    }

    #[tokio::test]
    async fn managed_build_and_retrieval_cover_all_corpora_and_dimensions() {
        for dimensions in [768, 2560, 4096] {
            let db = fixture();
            let profile = profile(dimensions);
            let lease = db
                .prepare_embedding_generation(
                    &profile,
                    &Corpus::ALL,
                    db.embedding_generation_control().unwrap().revision,
                )
                .unwrap();
            let fake = Arc::new(FakeAiProvider::new().with_embedding_dim(dimensions));
            let session = EmbeddingSession::new_for_testing(db.clone(), profile.clone(), fake.clone()).unwrap();
            assert_eq!(run_build_with_session(&db, &lease, &session).await.unwrap(), 3);
            let calls = fake.embed_calls();
            assert!(calls
                .iter()
                .any(|s| s.contains("Complete synthetic invoice body") && s.contains("Synthetic sender")));
            assert!(calls.iter().any(|s| s == "Synthetic workshop preference"));
            assert!(calls
                .iter()
                .any(|s| s == "Fixture guide › Fixture heading\nSynthetic instructions"));
            let active = db.active_embedding_write_lease().unwrap().unwrap();
            let vector = session.embed("invoice").await.unwrap();
            let query = TypedQueryEmbedding::new(active, &profile, "invoice", vector).unwrap();
            for corpus in Corpus::ALL {
                let hits = db
                    .search_generation_vectors(
                        &query,
                        "invoice",
                        corpus,
                        if corpus == Corpus::HelpDocs { None } else { Some("a") },
                        &[],
                        None,
                        5,
                    )
                    .unwrap();
                assert_eq!(hits.len(), 1, "{dimensions} {corpus:?}");
                assert!(hits[0].similarity.is_finite());
            }
            assert!(db
                .embedding_source_snapshots(lease.generation_id, Corpus::Emails, Some("a"), 10)
                .unwrap()
                .is_empty());
        }
    }

    #[tokio::test]
    async fn source_change_during_inference_rejects_result_and_keeps_active_generation() {
        let db = fixture();
        let profile = profile(768);
        let lease = db
            .prepare_embedding_generation(
                &profile,
                &Corpus::ALL,
                db.embedding_generation_control().unwrap().revision,
            )
            .unwrap();
        let fake = Arc::new(FakeAiProvider::new().with_embedding_dim(768));
        let db_for_hook = db.clone();
        fake.on_embed(move |_| {
            db_for_hook
                .connection()
                .execute(
                    "UPDATE email_bodies SET body='Changed during inference' WHERE email_id='email'",
                    [],
                )
                .unwrap();
        });
        let session = EmbeddingSession::new_for_testing(db.clone(), profile, fake).unwrap();
        assert!(run_build_with_session(&db, &lease, &session).await.is_err());
        assert_eq!(
            db.embedding_generation_status(lease.generation_id).unwrap().completed,
            0
        );
        assert_eq!(db.embedding_generation_control().unwrap().active_id, None);
    }

    #[tokio::test]
    async fn no_managed_generation_never_calls_mutable_embedding_provider() {
        let db = fixture();
        assert!(embed_query(&db, "invoice").await.unwrap().is_none());
        for corpus in Corpus::ALL {
            assert_eq!(update_active(&db, corpus, None, 10).await.unwrap(), 0);
        }
    }

    #[tokio::test]
    async fn explicit_pause_during_request_preserves_completed_sources_and_fences_response() {
        let db = fixture();
        let profile = profile(2560);
        let lease = db
            .prepare_embedding_generation(
                &profile,
                &Corpus::ALL,
                db.embedding_generation_control().unwrap().revision,
            )
            .unwrap();
        let fake = Arc::new(FakeAiProvider::new().with_embedding_dim(2560));
        let db_for_hook = db.clone();
        let generation = lease.generation_id;
        fake.on_embed(move |calls| {
            if calls == 2 {
                db_for_hook.stop_embedding_generation(generation, false, None).unwrap();
            }
        });
        let session = EmbeddingSession::new_for_testing(db.clone(), profile, fake.clone()).unwrap();
        assert!(run_build_with_session(&db, &lease, &session).await.is_err());
        assert_eq!(fake.embed_calls().len(), 2, "no subsequent help request after stop");
        let status = db.embedding_generation_status(generation).unwrap();
        assert_eq!(status.state, "paused");
        assert_eq!(status.completed, 1, "email completed before pause is retained");
        assert_eq!(db.embedding_generation_control().unwrap().active_id, None);
    }

    #[tokio::test]
    async fn same_profile_account_rebuild_copies_other_corpora_without_resending_them() {
        let db = fixture();
        let profile = profile(768);
        let first = db
            .prepare_embedding_generation(
                &profile,
                &Corpus::ALL,
                db.embedding_generation_control().unwrap().revision,
            )
            .unwrap();
        let fake = Arc::new(FakeAiProvider::new().with_embedding_dim(768));
        let session = EmbeddingSession::new_for_testing(db.clone(), profile.clone(), fake.clone()).unwrap();
        run_build_with_session(&db, &first, &session).await.unwrap();
        let next = db
            .prepare_embedding_generation(
                &profile,
                &Corpus::ALL,
                db.embedding_generation_control().unwrap().revision,
            )
            .unwrap();
        assert_eq!(
            run_build_with_session_scoped(&db, &next, &session, Some("a"))
                .await
                .unwrap(),
            3
        );
        assert_eq!(
            fake.embed_calls().len(),
            4,
            "three initial sources plus one scoped email"
        );
        assert_eq!(
            db.embedding_generation_control().unwrap().active_id,
            Some(next.generation_id)
        );
        assert_eq!(db.embedding_generation_status(next.generation_id).unwrap().completed, 3);
    }

    #[tokio::test]
    async fn proposed_server_preview_and_atomic_commit_do_not_save_before_build_job_exists() {
        use wiremock::{
            matchers::{method, path},
            Mock, MockServer, ResponseTemplate,
        };
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/embeddings"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"data":[{"index":0,"embedding":vec![0.1_f32;4096]}]})),
            )
            .mount(&server)
            .await;
        let db = fixture();
        let input = super::super::ai::AiSettingsInput {
            provider: "openai_compatible".into(),
            model: "fixture-chat".into(),
            embedding_model: Some("fixture-new-embed".into()),
            server: Some(super::super::ai_server::ServerSettingsInput {
                base_url: format!("{}/v1", server.uri()),
                api_key: Some("fixture-proposed-key".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let preview = preview_settings_build(&db, input.clone(), DimensionPolicy::Native, &[Corpus::Emails])
            .await
            .unwrap();
        assert_eq!(preview.profile.dimensions, 4096);
        assert!(db.get_preference("openai_compatible_base_url").unwrap().is_none());
        assert!(db.get_preference("openai_compatible_api_key_id").unwrap().is_none());
        assert!(db.get_preference("ai_provider").unwrap().is_none());
        assert!(db.embedding_generation_control().unwrap().pending_id.is_none());
        let mut changed_settings = input.clone();
        changed_settings.monthly_budget_usd = 1.0;
        assert!(
            commit_settings_build(&db, changed_settings, DimensionPolicy::Native, &preview)
                .await
                .is_err(),
            "a preview binds proposed non-secret settings too"
        );
        assert!(db.get_preference("ai_monthly_budget").unwrap().is_none());
        db.connection().execute_batch("CREATE TEMP TRIGGER reject_settings_candidate BEFORE INSERT ON embedding_generations BEGIN SELECT RAISE(ABORT,'synthetic job failure'); END;").unwrap();
        assert!(
            commit_settings_build(&db, input.clone(), DimensionPolicy::Native, &preview)
                .await
                .is_err()
        );
        assert!(db.get_preference("openai_compatible_base_url").unwrap().is_none());
        assert!(db.get_preference("openai_compatible_api_key_id").unwrap().is_none());
        assert!(db.get_preference("ai_provider").unwrap().is_none());
        assert!(db.embedding_generation_control().unwrap().pending_id.is_none());
        db.connection()
            .execute_batch("DROP TRIGGER reject_settings_candidate;")
            .unwrap();
        let lease = commit_settings_build(&db, input, DimensionPolicy::Native, &preview)
            .await
            .unwrap();
        assert_eq!(
            db.get_preference("openai_compatible_base_url").unwrap().as_deref(),
            Some(format!("{}/v1", server.uri()).as_str())
        );
        assert!(db.get_preference("openai_compatible_api_key_id").unwrap().is_some());
        assert_eq!(
            db.embedding_generation_control().unwrap().pending_id,
            Some(lease.generation_id)
        );
        assert!(db.embedding_generation_control().unwrap().active_id.is_none());
        assert_eq!(run_build(&db, &lease).await.unwrap(), 1);
        assert_eq!(
            db.embedding_generation_control().unwrap().active_id,
            Some(lease.generation_id)
        );
        let requests = server.received_requests().await.unwrap();
        assert!(requests
            .iter()
            .all(|r| serde_json::from_slice::<serde_json::Value>(&r.body)
                .unwrap()
                .get("dimensions")
                .is_none()));
    }
    #[tokio::test(start_paused = true)]
    async fn queued_lease_survives_more_than_its_original_deadline_without_inference() {
        use crate::services::clock::{Clock, FixedClock};
        let db = fixture();
        let profile = profile(768);
        let lease = db
            .prepare_embedding_generation(
                &profile,
                &[Corpus::Emails],
                db.embedding_generation_control().unwrap().revision,
            )
            .unwrap();
        let clock = Arc::new(FixedClock::new(chrono::Utc::now().timestamp()));
        assert!(
            db.heartbeat_embedding_generation_at(&lease, clock.now_secs() + 61)
                .is_err(),
            "the original un-heartbeated queue lease expires after60s"
        );
        let queue = task_queue::TaskQueue::new(1, "embedding_queue_test");
        let (release, blocked) = tokio::sync::oneshot::channel::<()>();
        let (started_tx, started) = tokio::sync::oneshot::channel();
        queue
            .submit_named("blocker", async move {
                let _ = started_tx.send(());
                let _ = blocked.await;
            })
            .await;
        started.await.unwrap();
        let (ran_tx, ran) = tokio::sync::oneshot::channel();
        let run_clock = clock.clone();
        enqueue_build_with(
            db.clone(),
            &queue,
            lease.clone(),
            clock.clone(),
            move |db, lease| async move {
                let current = db
                    .heartbeat_embedding_generation_at(&lease, run_clock.now_secs())
                    .is_ok();
                let _ = ran_tx.send(current);
                Ok(0)
            },
        )
        .await
        .unwrap();
        tokio::task::yield_now().await;
        for _ in 0..8 {
            clock.advance_secs(10);
            tokio::time::advance(Duration::from_secs(10)).await;
            tokio::task::yield_now().await;
        }
        assert_eq!(
            db.embedding_generation_status(lease.generation_id).unwrap().state,
            "building"
        );
        let deadline: i64 = db
            .reader()
            .query_row(
                "SELECT lease_until FROM embedding_generations WHERE id=?1",
                [lease.generation_id],
                |r| r.get(0),
            )
            .unwrap();
        assert!(deadline > clock.now_secs());
        release.send(()).unwrap();
        assert!(
            ran.await.unwrap(),
            "ownership is valid after80s of queued waiting without loading a provider"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cancelled_queue_entry_cannot_start_or_stop_a_resumed_epoch() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let db = fixture();
        let profile = profile(768);
        let lease = db
            .prepare_embedding_generation(
                &profile,
                &[Corpus::Emails],
                db.embedding_generation_control().unwrap().revision,
            )
            .unwrap();
        let queue = task_queue::TaskQueue::new(1, "embedding_queue_cancel_test");
        let (release, blocked) = tokio::sync::oneshot::channel::<()>();
        let (started_tx, started) = tokio::sync::oneshot::channel();
        queue
            .submit_named("blocker", async move {
                let _ = started_tx.send(());
                let _ = blocked.await;
            })
            .await;
        started.await.unwrap();
        let calls = Arc::new(AtomicU32::new(0));
        let calls_in_run = calls.clone();
        enqueue_build_with(
            db.clone(),
            &queue,
            lease.clone(),
            Arc::new(super::super::clock::SystemClock),
            move |_, _| async move {
                calls_in_run.fetch_add(1, Ordering::Relaxed);
                Ok(0)
            },
        )
        .await
        .unwrap();
        queue.cancel_matching(|name| name.starts_with("embeddings:rebuild:"));
        tokio::time::advance(Duration::from_millis(400)).await;
        tokio::task::yield_now().await;
        assert_eq!(
            db.embedding_generation_status(lease.generation_id).unwrap().state,
            "paused"
        );
        let resumed = db
            .resume_embedding_generation(
                lease.generation_id,
                &lease.fingerprint,
                db.embedding_generation_control().unwrap().revision,
            )
            .unwrap();
        release.send(()).unwrap();
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        db.embedding_build_is_current(&resumed).unwrap();
        assert_eq!(
            db.embedding_generation_status(resumed.generation_id).unwrap().state,
            "building"
        );
    }

    #[tokio::test]
    async fn failed_enqueue_pauses_only_its_owned_generation() {
        let db = fixture();
        let profile = profile(768);
        let lease = db
            .prepare_embedding_generation(
                &profile,
                &[Corpus::Emails],
                db.embedding_generation_control().unwrap().revision,
            )
            .unwrap();
        let queue = task_queue::TaskQueue::new(1, "embedding_queue_failed_test");
        queue.close_for_testing().await;
        assert!(enqueue_build(db.clone(), &queue, lease.clone()).await.is_err());
        assert_eq!(
            db.embedding_generation_status(lease.generation_id).unwrap().state,
            "paused"
        );
    }
}
