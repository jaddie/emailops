# Embedding generation database boundary

This module owns additive V041 storage and index lifecycle. It has no provider,
network, Tauri or UI dependency. Existing fixed-name vectors and metadata remain
legacy generation zero, preserved and unverified. Public legacy vector searches
return no candidates; FTS remains available. Legacy writers cannot populate a
managed generation. Stage 2 is a development checkpoint, not a standalone release:
provider request suppression and managed retrieval arrive in Stage 3.

## Invariants

- A generation's versioned, credential-free profile/fingerprint and actual
  dimensions are immutable. Profiles cover backend/endpoint, model/revision,
  requested dimensions, formatting/preprocessing, normalization, encoding and
  cosine distance. Missing model revision is an opaque identity, not a guarantee
  that a remote model cannot change. Input hashes cover complete formatted text.
- Candidate configuration, three vector tables and ownership are created in one
  top-level transaction. Table names derive only from internal integer IDs.
  sqlite-vec 0.1.9 has no rename/savepoint callbacks: no renaming or nested
  savepoints are used. File-backed tests exercise create/write/drop rollback and
  reopen. All metadata/vector replacements and source cleanup commit together.
- Source ledger revisions and tombstones survive source-ID reuse. SQL source
  triggers cover email input/body/eligibility, junk overrides, facts, help text,
  account enablement and deletions, including FK cascades. Each generation has
  cleanup triggers, so source mutations physically remove its old vectors and
  metadata before row mappings disappear. Unchanged sync writes do not invalidate.
- Email scope is persisted per generation/account, separately from embedding
  identity: category selection and the existing count/age cutoff. Corpus opt-out
  is explicit; memory/help settings can further disable requested corpora. Scope
  snapshots are captured under the configuration write lock. New source IDs in
  that scope join coverage; out-of-scope sources do not block activation.
- Workers carry generation, epoch, random owner token, fingerprint and source
  revision. Writes reject expired/stopped workers, configuration changes, changed
  sources, wrong batch association, dimensions, nonfinite and zero-norm vectors.
  Active writes also pin the active pointer/epoch and current scope revision.
- Full coverage/consistency/KNN validation runs on a reader snapshot. Publishing
  validation rechecks source, generation, configuration and schema watermarks.
  Activation compares those fences in a short write transaction and switches all
  corpora together, retaining the previous generation. Active source mutations
  invalidate old rows; they do not make an incompatible index eligible for reuse.
- Pausing preserves completed work and fences late results. A 60-second durable
  lease covers app/CLI overlap; startup interrupts only expired owners, never a
  healthy process. Restart does not restart inference. Heartbeat while a provider
  request is in flight; explicit resume obtains a new owner/epoch after identity
  and permission revalidation. An immediate restart can display building until
  the prior lease expires; startup cannot prove another process is dead.
- Rollback reopens the retained generation as a candidate, snapshots current
  selection policy and requires catch-up/validation/activation. Its profile is
  restored by switching the active pointer; credentials/consent/preferences are
  never restored. Discard cannot drop generation zero or the active generation.
  Copies require equal fingerprints and current, complete source chunks.
- Startup checks generated schema and at most one stored vector per table.
  Missing/invalid tables enter repair state and are not recreated. Broken cleanup
  triggers are removed so source data stays editable. Full validation remains a
  candidate operation, not a startup corpus scan.

## Public workflow and Stage 3 integration

1. Probe/authorize a provider and construct an immutable `EmbeddingProfile`.
   Read `embedding_generation_control`; `prepare_embedding_generation` requires
   its expected revision. Requested/detected dimensions must already agree.
2. Enumerate `embedding_generation_sources`; fetch source content and its revision
   under one reader snapshot, then format exact inputs with that profile. Do not
   mix a revision from one snapshot with content from another. Perform inference
   outside database transactions and heartbeat the build lease independently.
3. Store whole-source batches, or persist sanitized per-source errors. Validate,
   then activate. On stop/failure use `stop_embedding_generation`; for recovery
   re-probe the expected fingerprint and obtain a new lease with explicit resume.
4. Incremental indexing uses `active_embedding_write_lease` and
   `store_active_generation_embeddings`. Before enumerating work after a policy
   or account/settings change, explicitly `refresh_active_embedding_scope` with
   the current lease. This reconciles categories/cutoff/opt-outs and purges newly
   excluded vectors. Profile changes require a replacement generation.
5. The service must enforce AI enablement, current per-feature/remote consent,
   endpoint disclosure, key availability/revocation and provider identity before
   *every request*. Database checks fence stale commits; they cannot prevent a
   request already sent. Never derive credentials from saved profile metadata.
   Probe opaque identities and surface their limits; a matching fingerprint is
   caller-asserted provider identity, not a database-performed network check. The
   immutable request must carry its profile/session, source revision and exact
   input hash together; enforce provider response association before storing.
   The database checks batch cardinality and storage invariants, but cannot prove
   which semantic input/provider produced raw floats. No extra durable request
   ticket would establish that provenance.
6. Stage 3 must implement typed query embeddings, active-pointer/profile checks
   after inference, account/scope filtering, vector/metadata mapping in one read
   snapshot and retry/FTS fallback. These raw legacy search APIs do not query
   managed generations. Keep semantic SearchBar deliberately disabled.

Use the frozen active profile, not mutable legacy provider preferences, for both
query and document sessions. Stage 3 owns dimension probing/native-policy UI
contracts, malformed provider response association, provider/network errors and
retrieval/eval behavior. Stage 4 owns commands/settings/progress presentation.
