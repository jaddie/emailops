# services/retrieval

Shared FTS candidate fetches, typed managed vector fetches, Reciprocal Rank
Fusion and thread deduplication. Chat, draft precedent search, agent search and
research use the same managed embedding identity; SearchBar keeps its existing
FTS-only switch.

`fetch_fts(db, FtsRequest)` works without AI. `fetch_vector(db, VectorRequest)`
requires a `TypedQueryEmbedding` plus its exact formatted input; raw float slices
are not accepted. The database rechecks the active generation, profile, dimensions,
format and input hash on one reader snapshot. Its vec0 `rowid IN` subquery applies
account, generation scope, category, current source revision and eligibility
before KNN's candidate budget, and maps vector rows to source IDs in that snapshot.

`services/embedding_runtime::embed_query` creates query vectors through a frozen
active-profile session, rechecks the active pointer after inference, and returns
`None` without any provider call when no verified active index exists. Provider
errors, stale activation/configuration and missing indexes fall back to FTS at
call sites. Help reuses a typed query only when its exact input also matches;
rewritten mailbox questions otherwise require their own help query.

`services/embedding_runtime` owns revision-checked preview/commit, explicit
resume/re-probe and rollback, cross-corpus builds, per-request authorization,
source snapshots, cancellation and leases. `db/embedding_generations` owns
transactions and atomic activation. Keep prompts, chat assembly and frontend
lifecycle presentation in their existing owning modules.

Settings changes enter through `embedding_runtime::preview_settings_build` and
`commit_settings_build`, passing proposed `AiSettingsInput`, dimension policy and
the returned preview. They validate unsaved configuration, stage versioned
credentials, then atomically publish preference writes plus the candidate job.
`preview_build`/`commit_build` remain the lower-level profile-only path when the
trusted provider settings already exist. See the database module's commit-boundary
and orphan-secret rules; frontend code must not compose separate save/rebuild calls.

AI Backend & Models exposes a credential-free lifecycle snapshot (active, pending
and retained generations), actual dimensions, opaque identity warning and
per-corpus progress. Settings preview the unsaved configuration with neutral
input and confirm before the atomic settings/build commit; stop, resume, discard
and rollback remain explicit. Account-scoped rebuilds use the managed generation workflow. Existing unverified legacy
indexes remain intact and retrieval falls back to FTS until activation.
