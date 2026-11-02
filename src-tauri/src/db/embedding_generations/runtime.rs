//! Snapshot-based inference inputs and typed, active-only retrieval.
use super::*;

#[derive(Clone, Debug)]
pub enum SourceContent {
    Email {
        subject: String,
        sender: String,
        body: String,
    },
    Text(String),
}

#[derive(Clone, Debug)]
pub struct EmbeddingSourceSnapshot {
    pub source: SourceRevision,
    pub content: SourceContent,
}

/// Constructed only after inference with the frozen profile and exact input.
/// Raw vectors cannot cross the managed search boundary.
#[derive(Clone, Debug)]
pub struct TypedQueryEmbedding {
    lease: ActiveGenerationLease,
    query_format: String,
    input_hash: String,
    vector: Vec<f32>,
}

impl TypedQueryEmbedding {
    pub fn matches_input(&self, input: &str) -> bool {
        self.input_hash == input_hash(input.as_bytes())
    }
    pub fn new(
        lease: ActiveGenerationLease,
        profile: &EmbeddingProfile,
        input: &str,
        vector: Vec<f32>,
    ) -> Result<Self> {
        if profile.fingerprint()? != lease.fingerprint
            || input.is_empty()
            || vector.len() != profile.dimensions
            || vector.iter().any(|v| !v.is_finite())
            || vector.iter().all(|v| *v == 0.0)
        {
            return invalid("Query embedding does not match the frozen active profile or input");
        }
        Ok(Self {
            lease,
            query_format: profile.query_format.clone(),
            input_hash: input_hash(input.as_bytes()),
            vector,
        })
    }
}

#[derive(Clone, Debug)]
pub struct GenerationVectorHit {
    pub source_id: String,
    /// Help hydration uses its existing stable text rowid, never vector rowids.
    pub help_rowid: Option<i64>,
    pub similarity: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(
    feature = "ts",
    derive(ts_rs::TS),
    ts(export, export_to = "../../src/types/generated/")
)]
#[serde(rename_all = "camelCase")]
pub struct GenerationCorpusStatus {
    pub corpus: Corpus,
    pub enabled: bool,
    pub requested_enabled: bool,
    #[cfg_attr(feature = "ts", ts(type = "number"))]
    pub completed: i64,
    #[cfg_attr(feature = "ts", ts(type = "number"))]
    pub failed: i64,
    #[cfg_attr(feature = "ts", ts(type = "number"))]
    pub pending: i64,
}

impl Database {
    pub(crate) fn apply_fenced_ai_preference_writes(
        &self,
        expected_revision: i64,
        preferences: &[(String, Option<String>)],
    ) -> Result<()> {
        let mut conn = self.connection();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if control(&tx)?.revision != expected_revision {
            return invalid("AI settings changed while preparing the save; retry or use the atomic embedding workflow");
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
        tx.commit()?;
        Ok(())
    }
    /// Disable is a settings transition, not an index rebuild. Its transaction
    /// fences pending workers without deleting their completed work or any
    /// active vector data, and publishes credential references with settings.
    pub(crate) fn apply_embedding_opt_out_preferences(
        &self,
        expected_revision: i64,
        preferences: &[(String, Option<String>)],
    ) -> Result<()> {
        if !preferences
            .iter()
            .any(|(key, value)| key == "ai_embedding_model" && value.as_deref() == Some(""))
        {
            return invalid("An explicit empty embedding selection is required");
        }
        let mut conn = self.connection();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if control(&tx)?.revision != expected_revision {
            return invalid("AI settings changed; refresh before disabling Embeddings");
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
        tx.execute("UPDATE embedding_generations SET state='paused',epoch=epoch+1,owner=NULL,lease_until=NULL,validation_json=NULL,error='Embeddings disabled; explicit resume requires an enabled selection' WHERE id=(SELECT pending_id FROM embedding_control) AND state IN ('preparing','building','validating','failed','interrupted')",[])?;
        tx.commit()?;
        Ok(())
    }

    /// Cleanup from a queued/dropped worker must never stop a newer epoch.
    pub(crate) fn finish_embedding_build_if_owned(
        &self,
        lease: &BuildLease,
        failed: bool,
        error: &str,
    ) -> Result<bool> {
        Ok(self.connection().execute("UPDATE embedding_generations SET state=?5,epoch=epoch+1,owner=NULL,lease_until=NULL,validation_json=NULL,error=?6 WHERE id=?1 AND epoch=?2 AND owner=?3 AND fingerprint=?4 AND id=(SELECT pending_id FROM embedding_control) AND state IN ('preparing','building','validating')",params![lease.generation_id,lease.epoch,lease.owner,lease.fingerprint,if failed {"failed"} else {"paused"},error])?==1)
    }
    pub fn embedding_generation_corpus_status(&self, generation: i64) -> Result<Vec<GenerationCorpusStatus>> {
        let conn = self.reader();
        let mut stmt=conn.prepare("SELECT c.enabled,c.requested_enabled,(SELECT count(*) FROM embedding_generation_applicable_sources s JOIN embedding_generation_items i ON i.generation_id=s.generation_id AND i.corpus=s.corpus AND i.source_id=s.source_id AND i.source_revision=s.revision WHERE s.generation_id=c.generation_id AND s.corpus=c.corpus AND i.status='complete'),(SELECT count(*) FROM embedding_generation_applicable_sources s JOIN embedding_generation_items i ON i.generation_id=s.generation_id AND i.corpus=s.corpus AND i.source_id=s.source_id AND i.source_revision=s.revision WHERE s.generation_id=c.generation_id AND s.corpus=c.corpus AND i.status='failed'),(SELECT count(*) FROM embedding_generation_applicable_sources s WHERE s.generation_id=c.generation_id AND s.corpus=c.corpus AND NOT EXISTS(SELECT 1 FROM embedding_generation_items i WHERE i.generation_id=s.generation_id AND i.corpus=s.corpus AND i.source_id=s.source_id AND i.source_revision=s.revision AND i.status='complete')) FROM embedding_generation_corpora c WHERE generation_id=?1 AND corpus=?2")?;
        let mut statuses = Vec::new();
        for corpus in Corpus::ALL {
            if let Some(status) = stmt
                .query_row(params![generation, corpus.as_str()], |r| {
                    Ok(GenerationCorpusStatus {
                        corpus,
                        enabled: r.get(0)?,
                        requested_enabled: r.get(1)?,
                        completed: r.get(2)?,
                        failed: r.get(3)?,
                        pending: r.get(4)?,
                    })
                })
                .optional()?
            {
                statuses.push(status);
            }
        }
        Ok(statuses)
    }
    pub fn embedding_pending_count(&self, generation: i64, corpus: Corpus, account: Option<&str>) -> Result<i64> {
        Ok(self.reader().query_row("SELECT count(*) FROM embedding_generation_applicable_sources s WHERE s.generation_id=?1 AND s.corpus=?2 AND (?3 IS NULL OR s.account_id=?3) AND NOT EXISTS(SELECT 1 FROM embedding_generation_items i WHERE i.generation_id=s.generation_id AND i.corpus=s.corpus AND i.source_id=s.source_id AND i.source_revision=s.revision AND i.status='complete')",params![generation,corpus.as_str(),account],|r|r.get(0))?)
    }
    pub fn embedding_source_complete(&self, generation: i64, source: &SourceRevision) -> Result<bool> {
        Ok(self.reader().query_row("SELECT EXISTS(SELECT 1 FROM embedding_generation_items WHERE generation_id=?1 AND corpus=?2 AND source_id=?3 AND source_revision=?4 AND status='complete')",params![generation,source.corpus.as_str(),source.source_id,source.revision],|r|r.get(0))?)
    }
    pub fn embedding_build_is_current(&self, lease: &BuildLease) -> Result<()> {
        check_lease(&self.reader(), lease).map(|_| ())
    }
    pub fn embedding_source_is_current(&self, generation: i64, source: &SourceRevision) -> Result<()> {
        check_source(&self.reader(), source, generation)
    }

    pub fn managed_memory_facts(&self, ids: &[String], account: &str) -> Result<Vec<crate::models::MemoryFact>> {
        let conn = self.reader();
        let mut rows = Vec::new();
        let mut stmt = conn.prepare(&format!(
            "{} WHERE id=?1 AND account_id=?2 AND status<>'retired'",
            super::super::memory::FACT_SELECT
        ))?;
        for id in ids {
            if let Some(fact) = stmt
                .query_row(params![id, account], super::super::memory::row_to_fact)
                .optional()?
            {
                rows.push(fact);
            }
        }
        Ok(rows)
    }
    /// Content and revision come from the SAME reader transaction. A source edit
    /// after this snapshot is rejected by the existing fenced write API.
    pub fn embedding_source_snapshots(
        &self,
        generation: i64,
        corpus: Corpus,
        account: Option<&str>,
        limit: usize,
    ) -> Result<Vec<EmbeddingSourceSnapshot>> {
        let mut conn = self.reader();
        let tx = conn.transaction()?;
        let sources = {
            let mut stmt = tx.prepare("SELECT s.source_id,s.account_id,s.revision FROM embedding_generation_applicable_sources s WHERE s.generation_id=?1 AND s.corpus=?2 AND (?3 IS NULL OR s.account_id=?3) AND NOT EXISTS(SELECT 1 FROM embedding_generation_items i WHERE i.generation_id=s.generation_id AND i.corpus=s.corpus AND i.source_id=s.source_id AND i.source_revision=s.revision AND i.status='complete') ORDER BY s.source_id LIMIT ?4")?;
            let rows = stmt
                .query_map(params![generation, corpus.as_str(), account, limit as i64], |r| {
                    Ok(SourceRevision {
                        corpus,
                        source_id: r.get(0)?,
                        account_id: r.get(1)?,
                        revision: r.get(2)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows
        };
        let mut result = Vec::with_capacity(sources.len());
        for source in sources {
            let content = match corpus {
                Corpus::Emails => tx.query_row("SELECT e.subject,e.sender,CASE WHEN trim(coalesce(b.body,''))='' THEN e.snippet ELSE b.body END FROM emails e LEFT JOIN email_bodies b ON b.email_id=e.id WHERE e.id=?1",[&source.source_id],|r| Ok(SourceContent::Email {subject:r.get(0)?,sender:r.get(1)?,body:r.get(2)?}))?,
                Corpus::MemoryFacts => SourceContent::Text(tx.query_row("SELECT fact FROM memory_facts WHERE id=?1",[&source.source_id],|r|r.get(0))?),
                Corpus::HelpDocs => SourceContent::Text(tx.query_row("SELECT CASE WHEN page_title=heading THEN page_title ELSE page_title||' › '||heading END||char(10)||content FROM help_doc_chunks WHERE chunk_id=?1",[&source.source_id],|r|r.get(0))?),
            };
            result.push(EmbeddingSourceSnapshot { source, content });
        }
        tx.commit()?;
        Ok(result)
    }

    /// Active pointer, profile, source validity, filtering and vector→source
    /// mapping share one SQLite snapshot. Filtering precedes ORDER BY/LIMIT.
    /// vec0 receives eligible rowids before KNN, so excluded neighbours cannot
    /// consume the candidate budget. No corpus-wide vector materialization.
    #[allow(clippy::too_many_arguments)]
    pub fn search_generation_vectors(
        &self,
        query: &TypedQueryEmbedding,
        exact_input: &str,
        corpus: Corpus,
        account: Option<&str>,
        categories: &[String],
        help_page: Option<&str>,
        limit: usize,
    ) -> Result<Vec<GenerationVectorHit>> {
        let mut conn = self.reader();
        let tx = conn.transaction()?;
        let enabled: bool = tx.query_row(
            "SELECT coalesce((SELECT lower(value) FROM user_preferences WHERE key='ai_enabled'),'true')='true' AND coalesce((SELECT trim(value) FROM user_preferences WHERE key='ai_embedding_model'),'configured')<>''",
            [],
            |r| r.get(0),
        )?;
        if !enabled {
            return invalid("Embedding retrieval is disabled; use text search");
        }
        let dimensions = check_write_target(&tx, &WriteTarget::Active(&query.lease))?;
        let json: String = tx.query_row(
            "SELECT profile_json FROM embedding_generations WHERE id=?1",
            [query.lease.generation_id],
            |r| r.get(0),
        )?;
        let profile: EmbeddingProfile = serde_json::from_str(&json)?;
        if profile.fingerprint()? != query.lease.fingerprint
            || profile.query_format != query.query_format
            || dimensions != query.vector.len()
            || input_hash(exact_input.as_bytes()) != query.input_hash
        {
            return invalid("Query identity or input changed; retry retrieval with the active profile");
        }
        let table = vector_table(query.lease.generation_id, corpus)?;
        let categories = serde_json::to_string(categories)?;
        let blob: Vec<u8> = query.vector.iter().flat_map(|v| v.to_le_bytes()).collect();
        let sql = format!("WITH nearest AS (SELECT rowid,distance FROM {table} WHERE embedding MATCH ?1 AND rowid IN (SELECT c.id FROM embedding_generation_chunks c JOIN embedding_generation_applicable_sources s ON s.generation_id=c.generation_id AND s.corpus=c.corpus AND s.source_id=c.source_id AND s.revision=c.source_revision JOIN embedding_generation_items i ON i.generation_id=c.generation_id AND i.corpus=c.corpus AND i.source_id=c.source_id AND i.source_revision=c.source_revision AND i.status='complete' LEFT JOIN emails e ON c.corpus='emails' AND e.id=c.source_id LEFT JOIN help_doc_chunks h ON c.corpus='help_docs' AND h.chunk_id=c.source_id WHERE c.generation_id=?2 AND c.corpus=?3 AND (?4 IS NULL OR s.account_id=?4) AND (c.corpus<>'emails' OR json_array_length(?5)=0 OR EXISTS(SELECT 1 FROM json_each(?5) j WHERE lower(j.value)=lower(e.category))) AND (c.corpus<>'help_docs' OR ?6 IS NULL OR h.page=?6) AND (c.corpus<>'memory_facts' OR coalesce((SELECT value FROM user_preferences WHERE key='memory_enabled'),'false')='true') AND (c.corpus<>'help_docs' OR coalesce((SELECT value FROM user_preferences WHERE key='help_docs_enabled'),'true')='true')) ORDER BY distance LIMIT ?7) SELECT c.source_id,h.rowid,MAX(1.0-n.distance) similarity FROM nearest n JOIN embedding_generation_chunks c ON c.id=n.rowid LEFT JOIN help_doc_chunks h ON c.corpus='help_docs' AND h.chunk_id=c.source_id GROUP BY c.source_id,h.rowid ORDER BY similarity DESC,c.source_id");
        let rows = {
            let mut stmt = tx.prepare(&sql)?;
            let rows = stmt
                .query_map(
                    params![
                        blob,
                        query.lease.generation_id,
                        corpus.as_str(),
                        account,
                        categories,
                        help_page,
                        limit.saturating_mul(5).clamp(1, 4096) as i64
                    ],
                    |r| {
                        Ok(GenerationVectorHit {
                            source_id: r.get(0)?,
                            help_rowid: r.get(1)?,
                            similarity: r.get(2)?,
                        })
                    },
                )?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows
        };
        tx.commit()?;
        Ok(rows.into_iter().take(limit).collect())
    }
}
