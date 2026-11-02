# Embedding generation database boundary

This module owns additive V041 storage and index lifecycle. It has no provider,
network, Tauri or UI dependency. Existing fixed-name vectors and metadata remain
legacy generation zero, preserved and unverified. Public legacy vector searches
return no candidates; FTS remains available. Legacy writers cannot populate a
managed generation. Managed service execution now uses `services/embedding_runtime` and frozen
`embedding_session` providers. Legacy installations send no semantic inference
until an explicitly requested verified replacement is built.

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

## Stage 3 runtime boundary

`embedding_source_snapshots` reads content and its source revision together in
one transaction. `embedding_build_is_current` / `embedding_source_is_current`
fence requests before I/O, while the existing write transaction fences results.
`TypedQueryEmbedding::new` binds the active lease/profile, full exact query input
hash, query format and validated vector. `search_generation_vectors` requires
that exact input and checks the active pointer/configuration/profile in the same
reader snapshot as vec0 KNN and vector-row mapping. Eligible rowids are passed to
vec0 before top-K; filters include account, category, source revision, scope,
retirement, mailbox/junk/account eligibility and live feature gates. It never
falls back to legacy vectors. Pinned-extension tests place 24 closer excluded
neighbours ahead of one eligible neighbour to verify filtering precedes top-K.

Candidate creation, rollback preparation and discard carry a current active
scope forward across their lifecycle-only revision bump. A scope already stale
from a real settings change remains stale. This preserves active retrieval while
its replacement builds without blessing outdated account/category policy.
V042 adds detected-dimension metadata to the settings revision fence. Memory scope
matches the actual opt-in default (`false`); help remains on by default.

## Proposed settings and candidate commit

The settings UI uses `embedding_runtime::preview_settings_build(db, input,
policy, corpora)` and `commit_settings_build(db, input, policy, preview)`, followed
by a queued `run_build` with the returned lease. It never saves settings and then
independently asks for a rebuild. Preview validates the proposed endpoint,
disclosure, key change, source and model; only fixed neutral text is sent, with
no published settings or key references. Commit repeats validation/probing,
preserves and checks the preview's opaque observation identity, binds proposed
non-secret settings, and rechecks the optimistic revision.

`prepare_embedding_generation_with_preferences` publishes all planned settings,
current credential references, candidate profile, vector tables and durable job
inside one top-level SQLite transaction. A failed fence, settings write, DDL or
job insert rolls everything back. Active profile/pointer/vector rows are not
rewritten. A changed destination may temporarily make that old profile unavailable
until activation; it never acquires the new configuration by relabelling.

The OS vault cannot join a SQLite transaction. Release builds stage new secrets
under fresh UUID references before the transaction; staging never overwrites or
deletes the currently referenced key. A failed DB commit can leave an unreachable
vault entry, which grants no authority and is never resolved by retained profiles.
Debug key values and their references are written within the same SQLite
transaction. Current credential references are always resolved live; profile and
rollback metadata contain none. Clearing a reference revokes authority even if an
old vault entry remains. Legacy fixed references remain readable.

Simple rollback to a different previously configured endpoint is intentionally refused
until that destination/source is explicitly selected and authorized again with
current disclosure/key settings. The backend returns an actionable authorization
message; Stage 4 must display it rather than present a generic rebuild failure.
Retained profiles never restore old consent or credential references implicitly.
Same-destination model/dimension rollback verifies the retained neutral baseline
with current credentials, catches up sources and uses ordinary validation/activation.

## Queued ownership and explicit opt-out

Desktop commands hand a committed/resumed lease directly to
`embedding_runtime::enqueue_build`. Its guard starts a ten-second heartbeat
before enqueue, observes queued cancellation, and hands ownership to `run_build`
when picked up. `TaskQueue::try_submit_named` exposes enqueue failure plus the
existing shared task cancellation context. A dropped/failed/cancelled queued
entry pauses only its exact generation/epoch/owner; it cannot stop a later resume.
`run_build` failure cleanup uses the same ownership fence. No new lifecycle state
or startup inference was added. Process death stops heartbeats; ordinary expired
lease recovery marks interruption and still requires explicit re-probe/resume.
Tests advance a `FixedClock` and paused Tokio time through eighty queued seconds
without loading a provider, and cover cancellation followed by resume and enqueue
failure. Heartbeats never revive an expired lease.

`AiService::apply_embedding_opt_out` validates an explicitly empty model without
an embedding request, stages credentials as usual, then atomically saves settings
and pauses/fences pending work. It preserves active profile/vector bytes. Frozen
sessions and typed retrieval both check the live empty-model opt-out. Re-enabling
uses the explicit settings resume/build workflow; a matching paused generation
can retain its verified progress, and no paid work resumes automatically.

Legacy `AiService::save_config`, `prepare_ai_settings` and `apply_ai_settings`
share a model/backend/server-endpoint/dimension-policy guard whenever a managed
active or pending generation exists. Same-contract chat-model/budget/thinking
saves require no embedding probe; changed embedding contracts require the atomic
workflow, and disabling requires the explicit opt-out path. Legacy preference
writes use an optimistic revision fence and one SQLite transaction. Generic
`set_pref` and downloaded-model auto-selection call `set_legacy_ai_preference`;
unrelated preferences pass through normally, while current credential references,
disclosure and canary metadata remain backend-owned. These guards do not wrap
fixture/eval code that deliberately configures its own database directly.

Processing count/age limits travel with `AiSettingsInput` as optional
`maxEmailCount`/`maxEmailAgeDays` fields. Proposed values participate in the
non-secret preview fingerprint and are persisted before candidate scope is
snapshotted in the same transaction. Older callers that omit them preserve
existing preferences. Ordinary saves and explicit opt-out use the same planned
preference writes; the frontend never writes these limits after enqueueing.

## Polled lease recovery and retained-index collection

`embedding_generation_status` runs expired-owner recovery at the database status
boundary, using the current time, before reading status. A fast reopen can still
show a healthy unexpired lease, but a later status poll marks its expired owner
`interrupted`, increments its epoch and clears ownership without restarting any
inference. A provided-time helper makes this testable with `FixedClock`; healthy
other-process owners are unchanged and stale worker results remain fenced.

After a successful activation commits, `cleanup_obsolete_embedding_generations`
runs in a separate top-level transaction. It selects only unreferenced `retained`
generations above zero, protecting the active pointer, current rollback pointer
and any pending candidate. With any `pending_id` it performs no DDL; this preserves
schema-version validation. Cleanup drops each obsolete generation's invalidation
triggers/vector tables and cascades its metadata together, without changing the
control revision. Activation does not fail if cleanup fails: a warning is logged,
and the public cleanup method or the next activation can retry. Active and previous
rollback indexes remain usable after cleanup failure.

File-backed tests cover four activations with storage bounded to generation zero,
active and previous; injected DELETE and DROP failures with complete DDL/metadata
rollback; no schema change during pending validation; and a pinned sqlite-vec old
reader retaining its metadata and KNN snapshot while another connection collects
the obsolete table. A new reader sees the table removed after its snapshot ends.

`resume_embedding_generation_with_preferences` and
`prepare_embedding_rollback_with_preferences` are the atomic DB boundaries for
explicitly proposed settings plus retained progress/data. Their existing variants
delegate with no preference writes. The service must freshly validate provider
identity, current disclosure/credentials and the proposed requested-corpus set.
The DB transaction rechecks revision, pointer, resumable state and fingerprint,
publishes only the caller's current settings/credential references, snapshots the
resulting scope and obtains a new owner/epoch. It preserves complete current
sources that remain applicable and purges newly excluded sources transactionally.
A failed transition rolls back settings and scope changes together. No profile or
rollback record supplies a credential or restores consent. Explicitly re-authorized
cross-destination rollback can use this boundary; the simple rollback path still
refuses to infer historical destination/key authorization.
