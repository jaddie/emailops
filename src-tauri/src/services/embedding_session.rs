//! Immutable embedding identity with authorization checked before every request.
//! Clients and credentials are private; live HTTP clients are rebuilt from current
//! credentials so sessions never retain a revoked key. Neutral canaries detect
//! observable alias drift, but cannot prove opaque remote model provenance.

use std::sync::Arc;

use super::ai::AiService;
use crate::ai::openrouter::validate_embedding_vector;
#[cfg(test)]
use crate::ai::provider::AIProvider;
use crate::db::embedding_generations::{Corpus, EmbeddingProfile};
use crate::db::Database;
use crate::models::error::{AppError, Result};

pub(crate) const CANARY_TEXT: &str =
    "EmailOps embedding identity check version 1: a neutral sentence about shapes and colours.";

const OPAQUE_PROBE_PREFIX: &str = "opaque-observation-v1:";

/// A probe discriminator is observation evidence, never an advertised revision.
/// UI must keep these profiles visibly opaque, even though the field is populated.
pub fn has_opaque_identity(profile: &EmbeddingProfile) -> bool {
    profile
        .model_revision
        .as_deref()
        .is_none_or(|revision| revision.starts_with(OPAQUE_PROBE_PREFIX))
}

pub(crate) fn opaque_probe_revision(db: &Database, profile: &EmbeddingProfile, vector: &[f32]) -> Result<String> {
    let control = db.embedding_generation_control()?;
    for generation in [control.active_id, control.pending_id]
        .into_iter()
        .flatten()
        .filter(|id| *id > 0)
    {
        let prior = db.embedding_generation_profile(generation)?;
        if !has_opaque_identity(&prior) {
            continue;
        }
        let mut same_contract = profile.clone();
        same_contract.model_revision = prior.model_revision.clone();
        if same_contract.canonicalized()? != prior.canonicalized()? {
            continue;
        }
        if let (Some(revision), Some(baseline)) = (prior.model_revision.as_ref(), load_canary(db, &prior)?) {
            if compatible_canaries(&baseline, vector) {
                return Ok(revision.clone());
            }
        }
    }
    Ok(format!("{OPAQUE_PROBE_PREFIX}{}", uuid::Uuid::new_v4()))
}

pub struct EmbeddingSession {
    profile: EmbeddingProfile,
    db: Arc<Database>,
    #[cfg(test)]
    provider: Option<Arc<dyn AIProvider>>,
}

impl EmbeddingSession {
    pub(crate) fn new(db: Arc<Database>, profile: EmbeddingProfile) -> Result<Self> {
        Ok(Self {
            profile: profile.canonicalized()?,
            db,
            #[cfg(test)]
            provider: None,
        })
    }

    pub fn profile(&self) -> &EmbeddingProfile {
        &self.profile
    }

    #[cfg(test)]
    pub(crate) fn new_for_testing(
        db: Arc<Database>,
        profile: EmbeddingProfile,
        provider: Arc<dyn AIProvider>,
    ) -> Result<Self> {
        let mut session = Self::new(db, profile)?;
        session.provider = Some(provider);
        Ok(session)
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_for(Corpus::Emails, text).await
    }

    pub async fn embed_for(&self, corpus: Corpus, text: &str) -> Result<Vec<f32>> {
        AiService::authorize_embedding(&self.db, &self.profile, corpus)?;
        #[cfg(test)]
        if let Some(provider) = &self.provider {
            let result = provider.embed(text).await?;
            validate_embedding_vector(&result.embedding, Some(self.profile.dimensions))?;
            return Ok(result.embedding);
        }
        let result = AiService::embed_profile_request(&self.db, &self.profile, text, false).await?;
        Ok(result.embedding)
    }

    /// Compare a neutral, versioned canary with the first observed result for
    /// this fingerprint. This catches observable alias drift across resumes;
    /// it cannot establish identity for an opaque provider that changes only
    /// other inputs. The limitation also applies to advertised model revisions.
    pub async fn verify_identity(&self) -> Result<()> {
        let vector = self.embed(CANARY_TEXT).await?;
        #[cfg(test)]
        if self.provider.is_some() {
            return remember_canary(&self.db, &self.profile, &vector);
        }
        verify_canary(&self.db, &self.profile, &vector)
    }
}

pub(crate) fn remember_canary(db: &Database, profile: &EmbeddingProfile, vector: &[f32]) -> Result<()> {
    validate_embedding_vector(vector, Some(profile.dimensions))?;
    let key = format!("embedding_canary:{}", profile.fingerprint()?);
    let encoded = serde_json::to_string(vector)?;
    {
        let conn = db.connection();
        conn.execute(
            "INSERT OR IGNORE INTO user_preferences(key,value) VALUES(?1,?2)",
            rusqlite::params![key, encoded],
        )?;
    }
    verify_canary(db, profile, vector)
}

fn load_canary(db: &Database, profile: &EmbeddingProfile) -> Result<Option<Vec<f32>>> {
    let key = format!("embedding_canary:{}", profile.fingerprint()?);
    let Some(saved) = db.get_preference(&key)? else {
        return Ok(None);
    };
    let prior: Vec<f32> = serde_json::from_str(&saved)?;
    validate_embedding_vector(&prior, Some(profile.dimensions))?;
    Ok(Some(prior))
}

fn compatible_canaries(prior: &[f32], vector: &[f32]) -> bool {
    if prior.len() != vector.len() {
        return false;
    }
    let dot: f64 = prior.iter().zip(vector).map(|(a, b)| *a as f64 * *b as f64).sum();
    let norm_a: f64 = prior.iter().map(|v| (*v as f64).powi(2)).sum();
    let norm_b: f64 = vector.iter().map(|v| (*v as f64).powi(2)).sum();
    dot / (norm_a * norm_b).sqrt() >= 0.99999
}

pub(crate) fn verify_canary(db: &Database, profile: &EmbeddingProfile, vector: &[f32]) -> Result<()> {
    validate_embedding_vector(vector, Some(profile.dimensions))?;
    let prior = load_canary(db, profile)?.ok_or_else(|| AppError::AiError("This embedding identity has no verified neutral baseline. Probe a fresh candidate and build a replacement generation.".into()))?;
    if !compatible_canaries(&prior, vector) {
        return Err(AppError::AiError("The embedding model's neutral identity probe changed. Build a new generation after reviewing the provider; existing vectors cannot safely be reused. Model aliases are opaque and a matching probe cannot guarantee unchanged model identity.".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::provider::FakeAiProvider;
    use crate::db::embedding_generations::DimensionPolicy;

    fn profile() -> EmbeddingProfile {
        EmbeddingProfile {
            version: 1,
            backend: "ollama".into(),
            endpoint: Some("http://localhost:11434".into()),
            model: "synthetic-model".into(),
            model_revision: None,
            dimension_policy: DimensionPolicy::Explicit(8),
            dimensions: 8,
            query_format: "raw-query-v1".into(),
            document_format: "emailops-documents-v1".into(),
            preprocessing: "emailops-chunk-v1".into(),
            normalization: "none".into(),
            encoding: "float32-le".into(),
            distance_metric: "cosine".into(),
        }
    }

    #[tokio::test]
    async fn frozen_session_rechecks_master_and_feature_permissions_before_every_request() {
        let db = Arc::new(Database::new_for_testing().unwrap());
        let fake = Arc::new(FakeAiProvider::new());
        let session = EmbeddingSession::new_for_testing(db.clone(), profile(), fake.clone()).unwrap();
        assert_eq!(session.embed("synthetic input").await.unwrap().len(), 8);
        db.set_preference("ai_enabled", "false").unwrap();
        assert!(session.embed("blocked").await.is_err());
        db.set_preference("ai_enabled", "true").unwrap();
        assert!(session.embed_for(Corpus::MemoryFacts, "disabled memory").await.is_err());
        db.set_preference("memory_enabled", "true").unwrap();
        assert!(session.embed_for(Corpus::MemoryFacts, "enabled memory").await.is_ok());
        db.set_preference("memory_enabled", "false").unwrap();
        assert!(session.embed_for(Corpus::MemoryFacts, "blocked").await.is_err());
        assert_eq!(fake.embed_calls().len(), 2);
    }

    #[tokio::test]
    async fn new_opaque_probe_can_build_replacement_while_old_identity_refuses_drift() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        let db = Arc::new(Database::new_for_testing().unwrap());
        db.set_preference("openai_compatible_base_url", &server.uri()).unwrap();
        let mut candidate = profile();
        candidate.backend = "openai_compatible".into();
        candidate.endpoint = Some(server.uri());
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"data":[{"embedding":vec![0.5;8]}]})),
            )
            .mount(&server)
            .await;
        let first = AiService::probe_embedding_profile(db.clone(), &candidate)
            .await
            .unwrap();
        assert!(has_opaque_identity(&first));
        let revision = db.embedding_generation_control().unwrap().revision;
        db.prepare_embedding_generation(&first, &[Corpus::Emails], revision)
            .unwrap();
        server.reset().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"data":[{"embedding":[0.5000001,0.5,0.5,0.5,0.5,0.5,0.5,0.5]}]})),
            )
            .mount(&server)
            .await;
        let jittered = AiService::probe_embedding_profile(db.clone(), &candidate)
            .await
            .unwrap();
        assert_eq!(
            first.fingerprint().unwrap(),
            jittered.fingerprint().unwrap(),
            "numeric noise reuses the pending observed identity"
        );
        server.reset().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"data":[{"embedding":[-0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]}]})),
            )
            .mount(&server)
            .await;
        assert!(AiService::probe_embedding_profile(db.clone(), &first).await.is_err());
        assert!(AiService::embedding_session(db.clone(), &first).await.is_err());
        let replacement = AiService::probe_embedding_profile(db.clone(), &candidate)
            .await
            .unwrap();
        assert!(has_opaque_identity(&replacement));
        assert_ne!(first.fingerprint().unwrap(), replacement.fingerprint().unwrap());
        assert!(AiService::embedding_session(db, &replacement).await.is_ok());
    }

    #[tokio::test]
    async fn missing_baseline_never_silently_certifies_a_saved_profile() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"data":[{"embedding":vec![0.5;8]}]})),
            )
            .mount(&server)
            .await;
        let db = Arc::new(Database::new_for_testing().unwrap());
        db.set_preference("openai_compatible_base_url", &server.uri()).unwrap();
        let mut saved = profile();
        saved.backend = "openai_compatible".into();
        saved.endpoint = Some(server.uri());
        saved.model_revision = Some(format!("{OPAQUE_PROBE_PREFIX}{}", uuid::Uuid::new_v4()));
        assert!(AiService::embedding_session(db.clone(), &saved).await.is_err());
        assert!(AiService::probe_embedding_profile(db.clone(), &saved).await.is_err());
        assert!(load_canary(&db, &saved).unwrap().is_none());
    }

    #[tokio::test]
    async fn live_session_uses_rotated_credentials_and_refuses_retargeted_endpoint() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"data":[{"index":0,"embedding":vec![0.5;8]}]})),
            )
            .mount(&server)
            .await;
        let db = Arc::new(Database::new_for_testing().unwrap());
        db.set_preference("openai_compatible_base_url", &server.uri()).unwrap();
        super::super::ai_server::apply_key_change(
            &db,
            &super::super::ai_server::KeyChange::Set("synthetic-key-a".into()),
            true,
        )
        .unwrap();
        let mut saved = profile();
        saved.backend = "openai_compatible".into();
        saved.endpoint = Some(server.uri());
        remember_canary(&db, &saved, &[0.5; 8]).unwrap();
        let session = AiService::embedding_session(db.clone(), &saved).await.unwrap();
        db.set_preference("ai_embedding_model", "unrelated-new-model").unwrap();
        super::super::ai_server::apply_key_change(
            &db,
            &super::super::ai_server::KeyChange::Set("synthetic-key-b".into()),
            true,
        )
        .unwrap();
        session.embed("synthetic input").await.unwrap();
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0].headers.get("authorization").unwrap(),
            "Bearer synthetic-key-a"
        );
        assert_eq!(
            requests[1].headers.get("authorization").unwrap(),
            "Bearer synthetic-key-b"
        );
        let body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
        assert_eq!(body["model"], "synthetic-model");
        super::super::ai_server::apply_key_change(&db, &super::super::ai_server::KeyChange::Clear, true).unwrap();
        session.embed("keyless server").await.unwrap();
        let requests = server.received_requests().await.unwrap();
        assert!(
            !requests[2].headers.contains_key("authorization"),
            "revoked key is never reused"
        );
        db.set_preference("openai_compatible_base_url", "http://127.0.0.1:1")
            .unwrap();
        assert!(session.embed("must not leave").await.is_err());
        assert_eq!(server.received_requests().await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn remote_permission_and_help_gate_are_live_even_for_frozen_sessions() {
        let db = Arc::new(Database::new_for_testing().unwrap());
        let fake = Arc::new(FakeAiProvider::new());
        let mut remote = profile();
        remote.backend = "openai_compatible".into();
        remote.endpoint = Some("https://synthetic.invalid/v1".into());
        db.set_preference("ai_provider", "openai_compatible").unwrap();
        db.set_preference("openai_compatible_base_url", "https://synthetic.invalid/v1")
            .unwrap();
        db.set_preference("openai_compatible_disclosed_host", "synthetic.invalid")
            .unwrap();
        let session = EmbeddingSession::new_for_testing(db.clone(), remote, fake.clone()).unwrap();
        session.embed("consented synthetic input").await.unwrap();
        db.delete_preference("openai_compatible_disclosed_host").unwrap();
        assert!(session.embed("blocked remote").await.is_err());
        db.set_preference("openai_compatible_disclosed_host", "synthetic.invalid")
            .unwrap();
        db.set_preference("ai_provider", "ollama").unwrap();
        assert!(session.embed("old remote source no longer authorized").await.is_err());
        let local = EmbeddingSession::new_for_testing(db.clone(), profile(), fake.clone()).unwrap();
        db.set_preference("help_docs_enabled", "false").unwrap();
        assert!(local.embed_for(Corpus::HelpDocs, "blocked help").await.is_err());
        assert_eq!(fake.embed_calls().len(), 1);
    }

    #[tokio::test]
    async fn paid_frozen_session_rechecks_budget_before_each_request() {
        let db = Arc::new(Database::new_for_testing().unwrap());
        db.set_preference("ai_monthly_budget", "0.5").unwrap();
        db.set_preference("ai_provider", "openrouter").unwrap();
        let fake = Arc::new(FakeAiProvider::new());
        let mut paid = profile();
        paid.backend = "openrouter".into();
        paid.endpoint = Some("https://openrouter.ai/api/v1".into());
        let session = EmbeddingSession::new_for_testing(db.clone(), paid, fake.clone()).unwrap();
        session.embed("within budget").await.unwrap();
        db.connection().execute("INSERT INTO ai_usage(provider,model,operation,prompt_tokens,completion_tokens,cost_usd,timestamp) VALUES('openrouter','synthetic','embed',1,0,1,?1)",rusqlite::params![chrono::Utc::now().timestamp()]).unwrap();
        assert!(matches!(
            session.embed("past budget").await,
            Err(AppError::BudgetExceeded(_))
        ));
        assert_eq!(fake.embed_calls().len(), 1);
    }

    #[tokio::test]
    async fn frozen_session_never_accepts_a_different_actual_dimension() {
        let db = Arc::new(Database::new_for_testing().unwrap());
        let fake = Arc::new(FakeAiProvider::new().with_embedding_dim(9));
        let session = EmbeddingSession::new_for_testing(db, profile(), fake).unwrap();
        assert!(session.embed("input").await.is_err());
    }

    #[tokio::test]
    async fn durable_canary_detects_same_dimension_provider_drift() {
        let db = Arc::new(Database::new_for_testing().unwrap());
        let session =
            EmbeddingSession::new_for_testing(db.clone(), profile(), Arc::new(FakeAiProvider::new())).unwrap();
        session.verify_identity().await.unwrap();
        let key = format!("embedding_canary:{}", profile().fingerprint().unwrap());
        db.set_preference(&key, &serde_json::to_string(&[1.0f32; 8]).unwrap())
            .unwrap();
        assert!(session.verify_identity().await.is_err());
    }
}
