//! Build and check a managed index in a fresh synthetic eval database.
//! Takes a data directory; refuses an existing generation. Never points this at user data.
use emailops_lib::{
    db::embedding_generations::{Corpus, DimensionPolicy},
    services::{
        ai::{AiService, AiSettingsInput},
        ai_server::{EmbeddingSource, ServerSettingsInput},
        embedding_runtime, help_docs,
    },
    Database,
};
use std::{path::PathBuf, sync::Arc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(std::env::args_os().nth(1).ok_or("synthetic data directory required")?);
    // A caller must create this explicit sentinel beside the public synthetic fixture.
    if !dir.join("SYNTHETIC_EVAL_ONLY").is_file() {
        return Err("synthetic fixture sentinel required".into());
    }
    let db = Arc::new(Database::new(dir)?);
    let control = db.embedding_generation_control()?;
    if control.active_id.is_some() || control.pending_id.is_some() {
        return Err("fresh synthetic DB required".into());
    }
    db.set_preference("memory_enabled", "true")?;
    db.set_preference("ai_enabled", "true")?;
    help_docs::ensure_text_index(&db)?;
    let settings = || AiSettingsInput {
        provider: "openai_compatible".into(),
        model: "google/gemma-4-e4b".into(),
        embedding_model: Some("text-embedding-qwen3-embedding-8b".into()),
        thinking_enabled: Some(false),
        max_email_count: Some(1000),
        max_email_age_days: Some(365),
        server: Some(ServerSettingsInput {
            base_url: "http://localhost:1234/v1".into(),
            embedding_source: EmbeddingSource::Server,
            ..Default::default()
        }),
        ..Default::default()
    };
    // LM Studio ignores explicit dimensions. The production adapter must reject,
    // and the failed proposal must leave preferences/generation references intact.
    let before = db.embedding_generation_control()?;
    if embedding_runtime::preview_settings_build(&db, settings(), DimensionPolicy::Explicit(768), &Corpus::ALL)
        .await
        .is_ok()
    {
        return Err("server ignored explicit dimensions but production adapter accepted them".into());
    }
    let after = db.embedding_generation_control()?;
    if before.revision != after.revision || before.active_id != after.active_id || before.pending_id != after.pending_id
    {
        return Err("failed explicit probe changed control state".into());
    }
    let preview =
        embedding_runtime::preview_settings_build(&db, settings(), DimensionPolicy::Native, &Corpus::ALL).await?;
    if preview.profile.dimensions != 4096 {
        return Err("expected observed Qwen8b native4096".into());
    }
    let lease = embedding_runtime::commit_settings_build(&db, settings(), DimensionPolicy::Native, &preview).await?;
    let start = std::time::Instant::now();
    let count = embedding_runtime::run_build(&db, &lease).await?;
    let active = db
        .active_embedding_write_lease()?
        .ok_or("generation did not activate")?;
    if active.generation_id != lease.generation_id {
        return Err("wrong active generation".into());
    }
    let account: String = db.reader().query_row(
        "SELECT id FROM accounts WHERE email='ulises@emailopslabs.dev'",
        [],
        |r| r.get(0),
    )?;
    for corpus in Corpus::ALL {
        let input = match corpus {
            Corpus::Emails => "renewal quote",
            Corpus::MemoryFacts => "customer number",
            Corpus::HelpDocs => "Ollama backend",
        };
        let query = embedding_runtime::embed_query_for(&db, corpus, input)
            .await?
            .ok_or("missing typed query")?;
        let hits = db.search_generation_vectors(
            &query,
            input,
            corpus,
            if corpus == Corpus::HelpDocs {
                None
            } else {
                Some(&account)
            },
            &[],
            None,
            5,
        )?;
        if hits.is_empty() {
            return Err(format!("no live managed vector hits for {corpus:?}").into());
        }
        println!("corpus={corpus:?} typed_vector_hits={}", hits.len());
    }
    // Constructing through AiService is also the production permission boundary.
    let _ = AiService::embedding_session(db.clone(), &preview.profile).await?;
    println!(
        "dimensions={} generation={} sources={} elapsed_s={:.3} explicit_mismatch_rejected=true",
        preview.profile.dimensions,
        lease.generation_id,
        count,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
