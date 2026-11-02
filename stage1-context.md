# Stage 1: Configurable embedding dimensions and index lifecycle

Stage 1 was completed by a delegated **GPT-6 Astra agent at Extra High reasoning**. The user approved this design on **2026-11-02**, authorizing Stage 2 with GPT-6.1 Sol at High reasoning and its subsequent checkpoint. This file preserves the Stage 1 report; source links are repository-relative.

At completion of Stage 1, no files had been changed, tests run, migrations executed, or models loaded. The existing untracked `context.md` remained untouched.

## Recommendation and confirmed findings

Use **immutable embedding configurations with generation-based index replacement**, covering email, memory and help documents.

The source review confirmed several problems beyond hardcoded dimensions:

| Finding | Source |
|---|---|
| All three vector tables use 768 dimensions | [db/mod.rs](src-tauri/src/db/mod.rs), line 334 |
| Rebuilding deletes the existing index before creating its replacement | [embeddings.rs](src-tauri/src/services/embeddings.rs), line 651 |
| Settings save and rebuild are separate frontend operations | [AiSettings.tsx](src/components/Settings/AiSettings.tsx), line 539 |
| Email vector retrieval does not enforce embedding-model identity | [db/embeddings.rs](src-tauri/src/db/embeddings.rs), line 92 |
| Help retrieval reuses an untyped email-query vector | [help retrieval](src-tauri/src/services/help_docs/retrieval.rs), line 300 |
| Content hashes cover only the first 500 body bytes and omit the embedded sender | [embeddings.rs](src-tauri/src/services/embeddings.rs), line 104 |

Line references describe the source inspected during Stage 1 and may shift during implementation.

## Approved design

### 1. Separate requested dimensions from detected dimensions

Offer `Native` and `Explicit(n)`. Native omits the API parameter and records the actual response length. Explicit requests must return exactly the requested size. If LM Studio returns 4096 after requesting 768, report that mismatch and offer native 4096; never truncate, pad or silently change the selection.

Validate every document/query vector and batch response, including dimensions, finite values and result association. The pinned `sqlite-vec` 0.1.9 allows up to 8192 dimensions, so 768, 2560 and 4096 fit without upgrading it. [Pinned dependency source](https://docs.rs/crate/sqlite-vec/0.1.9/source/sqlite-vec.c)

### 2. Persist an immutable embedding profile and versioned fingerprint

Include backend/source, normalized endpoint, model and available revision/artifact identity, dimension policy and actual dimensions, query/document formatting, preprocessing versions, normalization, encoding and distance metric. Exclude credentials and unrelated chat settings.

Separately hash the complete, exact formatted embedding input. A matching vector length or model name alone must never establish compatibility. Indexing jobs and queries receive a frozen embedding session instead of rereading mutable preferences mid-operation.

### 3. Introduce durable generations through an additive migration

Store generation configuration, lifecycle state, progress, errors, ownership/fencing token and validation results. Maintain active, pending and retained-generation references. Scope embedding metadata to its generation and source revision.

Use tables such as `vec_emails_g42`, `vec_memory_facts_g42` and `vec_help_docs_g42`, with names generated exclusively from internal IDs. Preserve existing tables and metadata row IDs as legacy generation zero.

Activation changes ordinary SQLite metadata; it does not rename vector tables. The pinned extension has no virtual-table rename implementation. Transactional create/write/drop behaviour remains an explicit Stage 2 test gate. [Pinned dependency source](https://docs.rs/crate/sqlite-vec/0.1.9/source/sqlite-vec.c)

### 4. Build replacements alongside the active indexes

The lifecycle is `preparing → building → validating → active`, with paused, failed and retained states. Persist the proposed configuration and build job together in the backend; remove the frontend's separate save-then-rebuild responsibility.

Before activation, require complete applicable-corpus coverage, consistent metadata/vector rows, current source revisions and successful validation queries. Failed items prevent activation.

A short write transaction rechecks configuration revision, cancellation, ownership and source changes, then switches the active generation and retains the previous one. Provider requests never run inside that transaction.

### 5. Coordinate configuration changes across all three indexes

Model/dimension transitions activate together because current retrieval shares query embeddings. Corpus workers, progress and ordinary content updates remain independent. An explicitly disabled or unbuilt corpus stays on text search.

Same-profile scoped rebuilds may copy verified, unchanged vectors for unaffected accounts. Copying between different fingerprints is forbidden.

### 6. Capture source changes and fence stale work

Persist source edits, eligibility changes and deletions in the same transaction as their underlying changes. Each embedding request carries its generation, build epoch, source revision and content hash; all must still match when its result commits.

This prevents late responses from undoing cancellation, recreating deleted content or writing into a retired generation. Drain source changes before activation and recheck under the final write lock.

Account/folder/fact deletion must clean every generation. Existing fixed-table deletion paths need updating, including [account cleanup](src-tauri/src/db/accounts.rs), line 154. Durable ownership must also cover app/CLI overlap.

### 7. Enforce retrieval compatibility at the database boundary

Replace raw-vector handoffs with a typed query embedding carrying its profile fingerprint, generation, dimensions and formatting/input identity.

After inference, recheck the active generation before searching. If activation occurred meanwhile, retry once or use FTS. Run vector search and metadata mapping in one reader snapshot, retaining account isolation. Help/memory vector reuse requires matching identity.

Provider failures and mismatches produce actionable status plus existing text-search fallback. This work will not enable the deliberately disabled semantic SearchBar path.

### 8. Make interruption recovery and rollback explicit

Stop pauses the candidate and preserves completed work. Restart fences abandoned workers and marks the build interrupted. Resume revalidates provider identity and source revisions; it does not automatically restart paid remote processing.

Retain the previous generation for rollback. Restoring it also restores its matching embedding configuration, after checking provider availability and catching up source changes. Revoked credentials or remote-processing permission must remain revoked.

Missing or corrupt active tables produce a repair state and FTS fallback; startup must not silently recreate them at 768.

### 9. Preserve legacy data without inventing provenance

Existing settings, source data and vector bytes remain intact. Existing 768 configurations retain their requested dimensions.

However, legacy rows lack enough provenance to prove their embedding configuration. **The approved migration preserves them as unverified and uses text search until a verified replacement is built.** It does not automatically start a rebuild.

This temporarily reduces semantic functionality on existing installations, but avoids certifying potentially incompatible vectors. It was the principal compatibility trade-off presented for approval.

## UI requirements

The UI should show active and pending configurations, requested/detected dimensions, verification status, per-corpus progress, failures and interrupted builds. Actions should include probe, build-and-activate, stop, resume, retry, discard and rollback. Configuration previews must be revision-checked by the backend so another settings writer cannot bypass rebuild requirements.

## Risks and validation gates

- **Opaque model changes:** use available model revisions and neutral probe comparisons to detect drift. An API that silently replaces a model under the same identity cannot provide a strict provenance guarantee; this limitation must remain visible.
- **Storage:** raw float vectors grow from 3 KiB at 768 dimensions to 16 KiB at 4096. Rebuilding temporarily retains both generations, plus SQLite overhead.
- **SQLite correctness:** Stage 2 must use file-backed tests covering populated legacy migration, preserved row IDs/settings, vector/metadata transaction rollback, failed activation and restart recovery.
- **Concurrency:** test cancellation during inference/commit, competing jobs/settings changes, source deletion, retrieval during activation and stale-worker rejection.
- **Provider/retrieval:** exercise all three dimensions and corpora, ignored/rejected dimension parameters, same-dimension incompatible models, malformed batches and focused retrieval evals.
- **Performance:** measure throughput, query latency, contention, cancellation and LM Studio GPU/VRAM behaviour before changing batching or concurrency. No throughput or residency claims were verified during Stage 1.

## Execution stages and checkpoints

| Stage | Model | Reasoning | Gate |
|---|---|---|---|
| 1. Architecture/design | GPT-6 Astra (`gpt-6-astra`) | Extra High | Approved on 2026-11-02 |
| 2. Database migrations/index lifecycle | GPT-6.1 Sol (`gpt-6.1-sol`) | High | Verified database/migration/recovery changes; stop and report before Stage 3 |
| 3. Provider integration/RAG retrieval | GPT-6.1 Sol (`gpt-6.1-sol`) | High | Dynamic dimensions, fingerprints and retrieval isolation |
| 4. UI/configuration | GPT-6.1 Terra (`gpt-6.1-terra`) | Medium | Existing UI conventions and lifecycle visibility |
| 5. Tests/fixes/performance | GPT-6.1 Sol (`gpt-6.1-sol`) | Medium | Actual validation outcomes and measurements before optimization |
| 6. Independent final review | GPT-6 Astra (`gpt-6-astra`) | Extra High | Investigate safety/concurrency issues; resolve critical/high findings |

Use separate verified-stage commits where appropriate. Preserve user data, settings and existing changes. Do not push, install or deploy without authorization. Stage 2 may use synthetic vectors to verify the database contract; provider/retrieval and UI integration remain subsequent stages.
