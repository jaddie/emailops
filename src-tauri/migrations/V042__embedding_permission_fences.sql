-- Detected embedding dimensions participate in the settings revision fence.
DROP TRIGGER IF EXISTS embedding_preferences_insert;
DROP TRIGGER IF EXISTS embedding_preferences_update;
DROP TRIGGER IF EXISTS embedding_preferences_delete;
CREATE TRIGGER IF NOT EXISTS embedding_preferences_insert AFTER INSERT ON user_preferences
WHEN new.key IN ('ai_enabled','ai_provider','ai_embedding_model','openai_compatible_base_url','openai_compatible_embedding_source','openai_compatible_embedding_validated','openai_compatible_embedding_dimensions','openai_compatible_embedding_actual_dimensions','openrouter_embedding_validated_model','openrouter_embedding_dimensions','openrouter_embedding_actual_dimensions','openrouter_zdr','openai_compatible_disclosed_host','openai_compatible_api_key_id','openai_compatible_api_key_dev','openrouter_api_key_id','openrouter_api_key_dev','ai_max_email_count','ai_max_email_age_days','memory_enabled','help_docs_enabled') OR new.key GLOB 'embeddings_categories:*' BEGIN
    UPDATE embedding_control SET revision=revision+1;
END;
CREATE TRIGGER IF NOT EXISTS embedding_preferences_update AFTER UPDATE ON user_preferences
WHEN (new.key IN ('ai_enabled','ai_provider','ai_embedding_model','openai_compatible_base_url','openai_compatible_embedding_source','openai_compatible_embedding_validated','openai_compatible_embedding_dimensions','openai_compatible_embedding_actual_dimensions','openrouter_embedding_validated_model','openrouter_embedding_dimensions','openrouter_embedding_actual_dimensions','openrouter_zdr','openai_compatible_disclosed_host','openai_compatible_api_key_id','openai_compatible_api_key_dev','openrouter_api_key_id','openrouter_api_key_dev','ai_max_email_count','ai_max_email_age_days','memory_enabled','help_docs_enabled') OR new.key GLOB 'embeddings_categories:*') AND old.value IS NOT new.value BEGIN
    UPDATE embedding_control SET revision=revision+1;
END;
CREATE TRIGGER IF NOT EXISTS embedding_preferences_delete AFTER DELETE ON user_preferences
WHEN old.key IN ('ai_enabled','ai_provider','ai_embedding_model','openai_compatible_base_url','openai_compatible_embedding_source','openai_compatible_embedding_validated','openai_compatible_embedding_dimensions','openai_compatible_embedding_actual_dimensions','openrouter_embedding_validated_model','openrouter_embedding_dimensions','openrouter_embedding_actual_dimensions','openrouter_zdr','openai_compatible_disclosed_host','openai_compatible_api_key_id','openai_compatible_api_key_dev','openrouter_api_key_id','openrouter_api_key_dev','ai_max_email_count','ai_max_email_age_days','memory_enabled','help_docs_enabled') OR old.key GLOB 'embeddings_categories:*' BEGIN
    UPDATE embedding_control SET revision=revision+1;
END;
