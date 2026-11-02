-- Additive: legacy metadata/rowids/vector bytes/preferences are deliberately untouched.
-- Legacy generation zero has no provable profile and can never be activated.
CREATE TABLE IF NOT EXISTS embedding_generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_json TEXT,
    fingerprint TEXT,
    dimensions INTEGER NOT NULL CHECK(dimensions BETWEEN 1 AND 8192),
    state TEXT NOT NULL CHECK(state IN ('legacy_unverified','preparing','building','validating','active','retained','paused','interrupted','failed','repair')),
    epoch INTEGER NOT NULL DEFAULT 0,
    owner TEXT,
    lease_until INTEGER,
    config_revision INTEGER NOT NULL,
    scope_revision INTEGER NOT NULL DEFAULT 0,
    validation_json TEXT,
    mutation_version INTEGER NOT NULL DEFAULT 0,
    validated_source_version INTEGER,
    validated_mutation_version INTEGER,
    validated_schema_version INTEGER,
    error TEXT,
    created_at INTEGER NOT NULL,
    CHECK(id <> 0 OR state = 'legacy_unverified')
);
INSERT OR IGNORE INTO embedding_generations(id, dimensions, state, config_revision, created_at)
VALUES(0, 768, 'legacy_unverified', 0, unixepoch());
CREATE TABLE IF NOT EXISTS embedding_control (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    revision INTEGER NOT NULL DEFAULT 0,
    source_version INTEGER NOT NULL DEFAULT 0,
    active_id INTEGER REFERENCES embedding_generations(id),
    pending_id INTEGER REFERENCES embedding_generations(id),
    retained_id INTEGER REFERENCES embedding_generations(id)
);
INSERT OR IGNORE INTO embedding_control(singleton) VALUES(1);
CREATE TABLE IF NOT EXISTS embedding_generation_corpora (
    generation_id INTEGER NOT NULL REFERENCES embedding_generations(id) ON DELETE CASCADE,
    corpus TEXT NOT NULL CHECK(corpus IN ('emails','memory_facts','help_docs')),
    enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
    requested_enabled INTEGER NOT NULL CHECK(requested_enabled IN (0,1)),
    PRIMARY KEY(generation_id, corpus)
);
-- Tombstones keep monotonically increasing revisions when source IDs are reused.
CREATE TABLE IF NOT EXISTS embedding_sources (
    corpus TEXT NOT NULL CHECK(corpus IN ('emails','memory_facts','help_docs')),
    source_id TEXT NOT NULL,
    account_id TEXT,
    revision INTEGER NOT NULL,
    eligible INTEGER NOT NULL CHECK(eligible IN (0,1)),
    PRIMARY KEY(corpus, source_id)
);
CREATE INDEX IF NOT EXISTS embedding_sources_account ON embedding_sources(account_id);
CREATE TABLE IF NOT EXISTS embedding_generation_items (
    generation_id INTEGER NOT NULL REFERENCES embedding_generations(id) ON DELETE CASCADE,
    corpus TEXT NOT NULL CHECK(corpus IN ('emails','memory_facts','help_docs')),
    source_id TEXT NOT NULL,
    source_revision INTEGER NOT NULL,
    input_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('complete','failed')),
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK(chunk_count >= 0),
    error TEXT,
    PRIMARY KEY(generation_id, corpus, source_id)
);
CREATE TABLE IF NOT EXISTS embedding_generation_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL REFERENCES embedding_generations(id) ON DELETE CASCADE,
    corpus TEXT NOT NULL CHECK(corpus IN ('emails','memory_facts','help_docs')),
    source_id TEXT NOT NULL,
    source_revision INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    input_hash TEXT NOT NULL,
    UNIQUE(generation_id, corpus, source_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS embedding_generation_chunks_source ON embedding_generation_chunks(corpus, source_id, generation_id);
CREATE VIEW IF NOT EXISTS embedding_eligible_emails AS
SELECT e.id FROM emails e JOIN accounts a ON a.id=e.account_id
WHERE e.is_deleted=0 AND e.pending_sync=0 AND e.mailbox NOT IN ('spam','trash') AND a.enabled=1
  AND NOT EXISTS(SELECT 1 FROM email_junk j WHERE j.email_id=e.id
    AND (j.user_override='junk' OR (j.band='junk' AND j.primary_kind IN ('spam','phishing')
      AND (j.user_override IS NULL OR j.user_override<>'not_junk'))));
-- Eligibility here is structural. A frozen build may disable a whole corpus;
-- Stage 3 must additionally enforce opt-in/remote permissions before inference.
CREATE TRIGGER IF NOT EXISTS embedding_source_emails_insert AFTER INSERT ON emails BEGIN
    INSERT INTO embedding_sources(corpus,source_id,account_id,revision,eligible)
    VALUES('emails',new.id,new.account_id,1,EXISTS(SELECT 1 FROM embedding_eligible_emails WHERE id=new.id))
    ON CONFLICT(corpus,source_id) DO UPDATE SET account_id=excluded.account_id,
      revision=embedding_sources.revision+1,eligible=excluded.eligible;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_emails_update AFTER UPDATE OF id,subject,sender,sender_email,recipients_json,cc_json,snippet,raw_json,account_id,is_deleted,pending_sync,mailbox,category,timestamp ON emails
WHEN old.id IS NOT new.id OR old.subject IS NOT new.subject OR old.sender IS NOT new.sender OR old.sender_email IS NOT new.sender_email OR old.recipients_json IS NOT new.recipients_json OR old.cc_json IS NOT new.cc_json OR old.snippet IS NOT new.snippet OR old.raw_json IS NOT new.raw_json OR old.account_id IS NOT new.account_id OR old.is_deleted IS NOT new.is_deleted OR old.pending_sync IS NOT new.pending_sync OR old.mailbox IS NOT new.mailbox OR old.category IS NOT new.category OR old.timestamp IS NOT new.timestamp BEGIN
    INSERT INTO embedding_sources(corpus,source_id,account_id,revision,eligible)
    VALUES('emails',new.id,new.account_id,1,EXISTS(SELECT 1 FROM embedding_eligible_emails WHERE id=new.id))
    ON CONFLICT(corpus,source_id) DO UPDATE SET account_id=excluded.account_id,
      revision=embedding_sources.revision+1,eligible=excluded.eligible;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_emails_delete AFTER DELETE ON emails BEGIN
    UPDATE embedding_sources SET revision=revision+1,eligible=0 WHERE corpus='emails' AND source_id=old.id;
END;
INSERT INTO embedding_sources(corpus,source_id,account_id,revision,eligible) SELECT 'emails',new.id,new.account_id,1,EXISTS(SELECT 1 FROM embedding_eligible_emails WHERE id=new.id) FROM emails new;
CREATE TRIGGER IF NOT EXISTS embedding_source_memory_facts_insert AFTER INSERT ON memory_facts BEGIN
    INSERT INTO embedding_sources(corpus,source_id,account_id,revision,eligible)
    VALUES('memory_facts',new.id,new.account_id,1,(new.status <> 'retired' AND EXISTS(SELECT 1 FROM accounts WHERE id=new.account_id AND enabled=1)))
    ON CONFLICT(corpus,source_id) DO UPDATE SET account_id=excluded.account_id,
      revision=embedding_sources.revision+1,eligible=excluded.eligible;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_memory_facts_update AFTER UPDATE OF id,fact,status,account_id,subject_kind,subject_key ON memory_facts
WHEN old.id IS NOT new.id OR old.fact IS NOT new.fact OR old.status IS NOT new.status OR old.account_id IS NOT new.account_id OR old.subject_kind IS NOT new.subject_kind OR old.subject_key IS NOT new.subject_key BEGIN
    INSERT INTO embedding_sources(corpus,source_id,account_id,revision,eligible)
    VALUES('memory_facts',new.id,new.account_id,1,(new.status <> 'retired' AND EXISTS(SELECT 1 FROM accounts WHERE id=new.account_id AND enabled=1)))
    ON CONFLICT(corpus,source_id) DO UPDATE SET account_id=excluded.account_id,
      revision=embedding_sources.revision+1,eligible=excluded.eligible;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_memory_facts_delete AFTER DELETE ON memory_facts BEGIN
    UPDATE embedding_sources SET revision=revision+1,eligible=0 WHERE corpus='memory_facts' AND source_id=old.id;
END;
INSERT INTO embedding_sources(corpus,source_id,account_id,revision,eligible) SELECT 'memory_facts',new.id,new.account_id,1,(new.status <> 'retired' AND EXISTS(SELECT 1 FROM accounts WHERE id=new.account_id AND enabled=1)) FROM memory_facts new;
CREATE TRIGGER IF NOT EXISTS embedding_source_help_docs_insert AFTER INSERT ON help_doc_chunks BEGIN
    INSERT INTO embedding_sources(corpus,source_id,account_id,revision,eligible)
    VALUES('help_docs',new.chunk_id,NULL,1,1)
    ON CONFLICT(corpus,source_id) DO UPDATE SET account_id=excluded.account_id,
      revision=embedding_sources.revision+1,eligible=excluded.eligible;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_help_docs_update AFTER UPDATE OF chunk_id,lang,page,section_index,part,anchor,page_title,heading,content,nav_target ON help_doc_chunks
WHEN old.chunk_id IS NOT new.chunk_id OR old.lang IS NOT new.lang OR old.page IS NOT new.page OR old.section_index IS NOT new.section_index OR old.part IS NOT new.part OR old.anchor IS NOT new.anchor OR old.page_title IS NOT new.page_title OR old.heading IS NOT new.heading OR old.content IS NOT new.content OR old.nav_target IS NOT new.nav_target BEGIN
    INSERT INTO embedding_sources(corpus,source_id,account_id,revision,eligible)
    VALUES('help_docs',new.chunk_id,NULL,1,1)
    ON CONFLICT(corpus,source_id) DO UPDATE SET account_id=excluded.account_id,
      revision=embedding_sources.revision+1,eligible=excluded.eligible;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_help_docs_delete AFTER DELETE ON help_doc_chunks BEGIN
    UPDATE embedding_sources SET revision=revision+1,eligible=0 WHERE corpus='help_docs' AND source_id=old.chunk_id;
END;
INSERT INTO embedding_sources(corpus,source_id,account_id,revision,eligible) SELECT 'help_docs',new.chunk_id,NULL,1,1 FROM help_doc_chunks new;
CREATE TRIGGER IF NOT EXISTS embedding_source_body_insert AFTER INSERT ON email_bodies BEGIN
    UPDATE embedding_sources SET revision=revision+1,eligible=EXISTS(SELECT 1 FROM embedding_eligible_emails WHERE id=new.email_id) WHERE corpus='emails' AND source_id=new.email_id;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_body_update AFTER UPDATE ON email_bodies
WHEN old.body IS NOT new.body OR old.email_id IS NOT new.email_id BEGIN
    UPDATE embedding_sources SET revision=revision+1,eligible=EXISTS(SELECT 1 FROM embedding_eligible_emails WHERE id=new.email_id) WHERE corpus='emails' AND source_id=new.email_id;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_body_delete AFTER DELETE ON email_bodies BEGIN
    UPDATE embedding_sources SET revision=revision+1,eligible=EXISTS(SELECT 1 FROM embedding_eligible_emails WHERE id=old.email_id) WHERE corpus='emails' AND source_id=old.email_id;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_junk_insert AFTER INSERT ON email_junk BEGIN
    UPDATE embedding_sources SET revision=revision+1,eligible=EXISTS(SELECT 1 FROM embedding_eligible_emails WHERE id=new.email_id) WHERE corpus='emails' AND source_id=new.email_id;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_junk_update AFTER UPDATE ON email_junk
WHEN old.user_override IS NOT new.user_override OR old.band IS NOT new.band OR old.primary_kind IS NOT new.primary_kind BEGIN
    UPDATE embedding_sources SET revision=revision+1,eligible=EXISTS(SELECT 1 FROM embedding_eligible_emails WHERE id=new.email_id) WHERE corpus='emails' AND source_id=new.email_id;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_junk_delete AFTER DELETE ON email_junk BEGIN
    UPDATE embedding_sources SET revision=revision+1,eligible=EXISTS(SELECT 1 FROM embedding_eligible_emails WHERE id=old.email_id) WHERE corpus='emails' AND source_id=old.email_id;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_account_enabled AFTER UPDATE OF enabled ON accounts
WHEN old.enabled IS NOT new.enabled BEGIN
    UPDATE embedding_sources SET revision=revision+1,
      eligible=CASE corpus
        WHEN 'emails' THEN EXISTS(SELECT 1 FROM embedding_eligible_emails WHERE id=source_id)
        WHEN 'memory_facts' THEN EXISTS(SELECT 1 FROM memory_facts f WHERE f.id=source_id AND f.status<>'retired' AND new.enabled=1)
        ELSE eligible END
    WHERE account_id=new.id;
END;
-- Relevant mutable preferences participate in the optimistic configuration fence.
CREATE TRIGGER IF NOT EXISTS embedding_preferences_insert AFTER INSERT ON user_preferences
WHEN new.key IN ('ai_enabled','ai_provider','ai_embedding_model','openai_compatible_base_url','openai_compatible_embedding_source','openai_compatible_embedding_validated','openai_compatible_embedding_dimensions','openrouter_embedding_validated_model','openrouter_embedding_dimensions','openrouter_zdr','openai_compatible_disclosed_host','openai_compatible_api_key_id','openai_compatible_api_key_dev','openrouter_api_key_id','openrouter_api_key_dev','ai_max_email_count','ai_max_email_age_days','memory_enabled','help_docs_enabled') OR new.key GLOB 'embeddings_categories:*' BEGIN
    UPDATE embedding_control SET revision=revision+1;
END;
CREATE TRIGGER IF NOT EXISTS embedding_preferences_update AFTER UPDATE ON user_preferences
WHEN (new.key IN ('ai_enabled','ai_provider','ai_embedding_model','openai_compatible_base_url','openai_compatible_embedding_source','openai_compatible_embedding_validated','openai_compatible_embedding_dimensions','openrouter_embedding_validated_model','openrouter_embedding_dimensions','openrouter_zdr','openai_compatible_disclosed_host','openai_compatible_api_key_id','openai_compatible_api_key_dev','openrouter_api_key_id','openrouter_api_key_dev','ai_max_email_count','ai_max_email_age_days','memory_enabled','help_docs_enabled') OR new.key GLOB 'embeddings_categories:*') AND old.value IS NOT new.value BEGIN
    UPDATE embedding_control SET revision=revision+1;
END;
CREATE TRIGGER IF NOT EXISTS embedding_preferences_delete AFTER DELETE ON user_preferences
WHEN old.key IN ('ai_enabled','ai_provider','ai_embedding_model','openai_compatible_base_url','openai_compatible_embedding_source','openai_compatible_embedding_validated','openai_compatible_embedding_dimensions','openrouter_embedding_validated_model','openrouter_embedding_dimensions','openrouter_zdr','openai_compatible_disclosed_host','openai_compatible_api_key_id','openai_compatible_api_key_dev','openrouter_api_key_id','openrouter_api_key_dev','ai_max_email_count','ai_max_email_age_days','memory_enabled','help_docs_enabled') OR old.key GLOB 'embeddings_categories:*' BEGIN
    UPDATE embedding_control SET revision=revision+1;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_emails_rekey BEFORE UPDATE OF id ON emails
WHEN old.id IS NOT new.id BEGIN
    UPDATE embedding_sources SET revision=revision+1,eligible=0 WHERE corpus='emails' AND source_id=old.id;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_memory_facts_rekey BEFORE UPDATE OF id ON memory_facts
WHEN old.id IS NOT new.id BEGIN
    UPDATE embedding_sources SET revision=revision+1,eligible=0 WHERE corpus='memory_facts' AND source_id=old.id;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_help_docs_rekey BEFORE UPDATE OF chunk_id ON help_doc_chunks
WHEN old.chunk_id IS NOT new.chunk_id BEGIN
    UPDATE embedding_sources SET revision=revision+1,eligible=0 WHERE corpus='help_docs' AND source_id=old.chunk_id;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_watermark_insert AFTER INSERT ON embedding_sources BEGIN
    UPDATE embedding_control SET source_version=source_version+1;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_watermark_update AFTER UPDATE ON embedding_sources BEGIN
    UPDATE embedding_control SET source_version=source_version+1;
END;
CREATE TABLE IF NOT EXISTS embedding_generation_email_scopes (
    generation_id INTEGER NOT NULL REFERENCES embedding_generations(id) ON DELETE CASCADE,
    account_id TEXT NOT NULL,
    categories_json TEXT NOT NULL,
    min_timestamp INTEGER,
    PRIMARY KEY(generation_id,account_id)
);
CREATE VIEW IF NOT EXISTS embedding_generation_applicable_sources AS
SELECT c.generation_id,s.corpus,s.source_id,s.account_id,s.revision,s.eligible
FROM embedding_sources s JOIN embedding_generation_corpora c ON c.corpus=s.corpus
WHERE c.enabled=1 AND s.eligible=1
  AND (s.corpus<>'emails' OR EXISTS(
    SELECT 1 FROM emails e JOIN embedding_generation_email_scopes p
      ON p.generation_id=c.generation_id AND p.account_id=e.account_id
    WHERE e.id=s.source_id AND (p.min_timestamp IS NULL OR e.timestamp>=p.min_timestamp)
      AND (json_array_length(p.categories_json)=0 OR EXISTS(
        SELECT 1 FROM json_each(p.categories_json) category WHERE lower(category.value)=lower(e.category)))));
CREATE TRIGGER IF NOT EXISTS embedding_account_insert AFTER INSERT ON accounts BEGIN
    UPDATE embedding_control SET revision=revision+1;
END;
CREATE TRIGGER IF NOT EXISTS embedding_source_body_rekey BEFORE UPDATE OF email_id ON email_bodies
WHEN old.email_id IS NOT new.email_id BEGIN
    UPDATE embedding_sources SET revision=revision+1 WHERE corpus='emails' AND source_id=old.email_id;
END;
CREATE TRIGGER IF NOT EXISTS embedding_generation_profile_immutable
BEFORE UPDATE OF profile_json,fingerprint,dimensions ON embedding_generations
WHEN old.profile_json IS NOT new.profile_json OR old.fingerprint IS NOT new.fingerprint OR old.dimensions IS NOT new.dimensions
BEGIN SELECT RAISE(ABORT,'Embedding generation profiles are immutable'); END;
