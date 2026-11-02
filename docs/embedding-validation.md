# Embedding lifecycle validation — 2026-11-02

Scope: additive managed index generations, frozen provider identity, typed retrieval,
and the Settings lifecycle. See [database boundary](../src-tauri/src/db/embedding_generations/MODULE.md)
and [approved design](../stage1-context.md). No installed database was opened;
no app installation, deployment or push was performed.

## Live LM Studio observations

Loopback endpoint: `http://localhost:1234/v1`. Only neutral synthetic text and the
repository's public synthetic demo fixture were sent. Ordinary inference can
cause LM Studio to auto-load/evict a model; no management load/unload requests,
settings changes or downloads were made.

`/api/v0/models` reported Qwen's embedding models as GGUF, `Q4_K_M`, architecture
`qwen3`. The primary alias was `text-embedding-qwen3-embedding-8b`; its actual
native output was **4096 floats**. Quantization is server-reported metadata,
not an independently verified artifact hash. Qwen 4b produced 2560 and nomic 1.5
produced 768. All three ignored explicit `dimensions` values 768, 2560 and 4096,
returning their native length. The runtime must reject mismatched explicit
requests rather than truncate/pad or write incompatible vectors.

All five advertised models initially reported `not-loaded`. After benchmarking,
only nomic reported `loaded`: listing does not establish residency. Observational
`rocm-smi` snapshots showed 4.63 GB VRAM before and 4.91 GB after out of 25.75 GB, with
5% GPU busy in both samples and a low-power warning. These snapshots do not prove
inference ran on the GPU, simultaneous residency or peak VRAM consumption.

## Synthetic throughput benchmark

Each configuration used 16 bounded neutral inputs (about 150 words each), one
unmeasured warm-up request and three measured repetitions. Values below are
median inputs/second; every measured configuration had zero request errors.
HTTP request latency includes JSON parsing; elapsed throughput includes the
thread-pool overhead. Cold load time is excluded. This is one local server and
small workload, not a production throughput guarantee.

| Model | Serial 1 | Batch 2 | Batch 4 | Batch 8 | Serial requests, concurrency 2 |
|---|---:|---:|---:|---:|---:|
| Qwen embedding 4b, 2560 |15.65|17.18|17.32|21.47|22.23|
| Qwen embedding 8b, 4096 |12.24|12.59|12.60|12.70|13.11|
| nomic 1.5, 768 |88.03|90.32|100.52|103.64|131.14|

Median single-request latency for the primary 8b was 80.7 ms serial and 150.3 ms
with concurrency 2. Batch 8 requests took 629.5 ms median. The primary model's
throughput improvement was small (~4% for batch 8, ~7% for concurrency 2), while
individual-request latency increased. Production remains serial; preserving
source/cancellation fencing has more value than adding concurrency for this
measurement. Other models showed larger workload-specific gains.

Raw reports: `/tmp/emailops-stage5-benchmark.json` and `.jsonl` (local only).

## Reproduction

Set normal local Cargo/Node tooling as needed; this session used writable caches
`CARGO_HOME=/tmp/emailops-cargo-home`, `CARGO_TARGET_DIR=/tmp/emailops-stage2-target`,
`CARGO_INCREMENTAL=0`, and Node 22. Final checks used
`CARGO_BUILD_JOBS=16`, `RUST_TEST_THREADS=16`; the primary target path became a
symlink to the ignored NVMe `src-tauri/target`, with a separate check lane.

```sh
make benchmark-embeddings ARGS="--model text-embedding-qwen3-embedding-8b --model text-embedding-qwen3-embedding-4b --model text-embedding-nomic-embed-text-v1.5 --out /tmp/embedding-benchmark.json"
make eval-embedding-lifecycle
```

The eval target makes a fresh temporary schema with the app's migration runner,
seeds the existing public demo fixture, builds a managed Native 4096 generation
across all three corpora, verifies typed vector hits, then invokes the mandatory
`examples/chat_eval.rs` harness with explicit `--prod-db <synthetic path>`. Each case receives a separate
SQLite backup under the script's new temporary directory. Because the harness's
conservative five-page/250ms snapshot makes vector-index copies take minutes,
the script uses `--in-place-dangerous` **only for its own disposable per-case
copy**, after checking the synthetic sentinel and resolved path containment.
The populated fixture and installed app database are never the in-place target.
Judge mode remains off. The canonical embedded Qwen 3.5 4b chat model was not
advertised by LM Studio; `google/gemma-4-e4b` is an explicitly documented local
alternative, so these evals do not establish canonical-model parity.

## Fixes and tests

Independent review found and resolved these lifecycle integration defects:

- Processing count/age limits were saved after candidate enqueue, invalidating
  its revision and capturing old scope. They now participate in the settings DTO,
  preview fingerprint and atomic preferences/candidate transaction. Missing
  fields preserve older callers' settings; no later frontend scope writes remain.
- Re-enabling after opt-out now explicitly reauthorizes and resumes the same
  paused generation, preserving completed verified vectors. A disabled live
  memory/help switch does not erase its requested corpus set or force a restart;
  actual requests remain gated off. Resume cannot expand the original request.
- Retained rollback accepts proposed current settings, disclosure and credentials
  through a concrete preview/confirmation. A→B→reauthorized A reuses A without
  creating C, never sends B's key to A, and never restores historical secrets.
- Status polling recovers expired owners after a quick restart. Activation also
  attempts bounded cleanup of obsolete, unreferenced retained generations;
  active/pending/retained/legacy references and pending builds are protected.

Tests that drove the settings regressions first failed on the old behavior.
Confirmed checks:

| Check | Outcome |
|---|---|
| Final full Rust gate, default features | **4657 passed, 0 failed, 12 ignored** |
| Final full frontend Vitest gate | **3452 passed** |
| Final Clippy with and without desktop, rustfmt, Biome, TypeScript | Passed |
| Eval server-context configuration regressions | 15 passed |
| Default-feature embedding tests before review fixes | 124 passed |
| Final `services::embedding_runtime::tests`, no default features | **13 passed** |
| Final file-backed lifecycle/wrapper suite, separate check lane | **41 passed** |
| Final scoped Settings/lifecycle Vitest suites | **51 passed** |
| `export_bindings_aisettingsinput` | 1 passed |
| Final `export_bindings_embedding` | 4 passed |
| Final `make build` (TypeScript + Vite production output) | Passed |
| Both eval examples, `--no-default-features --features eval` | Built |
| Final default-feature desktop `cargo build --bin emailops` | **Passed** |

The 13 service tests use trait fakes or a loopback mock provider; the 41 lifecycle
checks use synthetic file-backed SQLite with the pinned sqlite-vec extension.
These are distinct from live LM Studio measurements. Tests cover 768/2560/4096,
all corpora, populated legacy migration/row IDs/settings, transaction rollback,
source edits/deletion, cancellation/restart/stale workers, retrieval during
activation, missing tables, credential revocation and retained rollback.

Migration tests preserve legacy vectors/settings as **unverified**, with text
search until an explicit verified replacement is built. No installed user
migration, interactive app session or physical-device acceptance was tested.
The existing dependency-freshness finding (`calamine` 0.37→0.38) was not changed.
Final gate logs are under `/tmp/emailops-final-gates/{commit,push,frontend}`.
The push gate exits unsuccessfully solely because of that dependency-freshness
check; no Git push was performed. The final independent source review is recorded
in [embedding-review.md](embedding-review.md).

## Live managed runtime and reply evals

The actual Qwen 8b Native 4096 workflow activated **529 sources** together:
110 emails, 7 memory facts and 412 help chunks. Typed queries returned five
managed hits from each corpus. A mismatched Explicit 768 proposal was rejected
without changing lifecycle control state. The scope-fixed run took **62.649 s**
including post-build query/identity checks; an earlier run took 64.599 s. These
measurements preceded the final resume/rollback recovery fixes, which have the
focused regression coverage above.

The mandatory `examples/chat_eval.rs` gate ran on isolated synthetic copies with
Gemma as the documented alternative chat model and Qwen 8b embeddings; no judge.
LM Studio reported Gemma's loaded context as **8192**, despite its advertised
maximum 131072. The initial override left the client budget automatic. An explicit
`EMAILOPS_EVAL_CONTEXT_WINDOW=8192` rerun used the supported server-settings
planner and **did not resolve all failures**:

| Case | Final checks | Result |
|---|---:|---|
| `retrieval_trashed_quote_not_cited` | 3/4 | Failed: answer omitted the live 4200 quote; forbidden 3100 check passed |
| `retrieval_excludes_junk_lookalike` | 3/4 | Failed only `prompt_fits_window`; correct price and junk exclusions passed |
| `app_help_ollama_en` | 7/8 | Failed only `prompt_fits_window`; help grounding/citation and no-mailbox checks passed |
| `mem_borgbase_customer_number` | 3/3 | Passed actual memory-fact recall |
| `demo_memory_pending_tasks` | 4/4 | Passed route/tool/reply checks |

The two budget failures reported estimated prompts of **6989** and **7274** tokens
in an 8192-token window. The existing planner reserves 1024 reply tokens plus
256 safety tokens, leaving a **6912-token prompt allowance**. The heuristic fails
when `BudgetTrace.fits` is false; this is not proof the server truncated input.
The exact origin of the remaining estimate/budget mismatch was not established.
No prompt was changed and no assertion was weakened to obtain a passing result.

For the quote failure, the trace showed the alternative model requesting
`from="Larkspur Freight"` with semantic query `renewal quote`; it got no matches.
The synthetic fixture contains the live quote, but that literal sender filter
matches neither its sender display name nor its hyphenated email domain. One
isolated comparison with embeddings explicitly disabled returned the same zero
matches and failed the same 4200 assertion (3/4). This confirms the failure also
occurs without managed vectors; it does not establish canonical-model parity
or a complete diagnosis of the planner. Further unrelated prompt work was left
out of this lifecycle change. **The scoped reply-eval gate is not fully green.**

Initial reports: `/tmp/emailops-embedding-eval.MG6GHA/reports/`.
Explicit-context rerun: `reports-8192/` under that same directory.
Text-search comparison: `reports-fts/` under that directory. Raw reports and
synthetic databases remain local; only this sanitized summary is tracked.

Fixture limitation discovered and resolved: `generate_demo_db.py --prod-db`
clones DDL/history without migration seed rows, so it cannot produce this
managed-index fixture. The eval target uses the generator's normal fresh
migration path, which backs up initialized schema and seed rows. The script
returns failure on heuristic errors even though the example harness itself can
exit successfully after rendering a failed-case report.
