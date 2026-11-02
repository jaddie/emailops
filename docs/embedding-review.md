# Independent embedding lifecycle review

Reviewed on 2026-11-02 by a delegated **GPT-6 Astra agent at Extra High reasoning**, initially in a fresh context. Scope: baseline `d22f7f44` through the implementation commits and final Stage 5 working diff, including validation tooling. The reviewer performed an initial review and a follow-up review of the fixes.

**Final source verdict: no remaining actionable findings in the inspected diff.** This is a bounded change review, not a whole-application security assessment or deployment approval. Executed test, build and live-provider outcomes are recorded separately in [embedding-validation.md](embedding-validation.md).

## Findings and resolutions

References identify the final reviewed source; line numbers may move in later changes.

| Severity | Finding | Resolution and source |
| --- | --- | --- |
| P1 | Settings could invalidate a newly queued build by writing email count/age limits afterward. | Limits now participate in the preview fingerprint and atomic preference/candidate transaction. Separate frontend writes were removed. `src-tauri/src/services/ai.rs:702` |
| P2 | A restart before lease expiry could leave abandoned work permanently displayed as building. | Status polling performs expiry-fenced recovery, requiring explicit resume. `src-tauri/src/db/embedding_generations.rs:797` |
| P2 | Repeated activation accumulated inaccessible older retained indexes. | Separate best-effort cleanup preserves legacy zero, active and the previous generation; it skips pending builds and does not change configuration revision. `src-tauri/src/db/embedding_generations.rs:942` |
| P2 | Opt-out preserved completed work but offered no supported way to re-enable and resume it. | A verified settings proposal can atomically reauthorize and resume the same generation with a fresh ownership fence. `src-tauri/src/services/embedding_runtime.rs:117`, `src-tauri/src/db/embedding_generations.rs:821` |
| P2 | Cross-destination rollback required starting another replacement build, potentially losing the desired rollback target. | Explicit rollback preview accepts freshly authorized settings and credentials; the transition publishes those settings atomically without creating another generation. `src-tauri/src/services/embedding_runtime.rs:192`, `src-tauri/src/db/embedding_generations.rs:984` |
| P2 | Resuming after disabling memory/help incorrectly demanded discarding the candidate. | Resume preserves its original requested corpora, intersects them with live feature permissions, and rejects expansion. `src-tauri/src/services/embedding_runtime.rs:117` |

## Review evidence and boundaries

The reviewer inspected migration preservation, transaction boundaries, source invalidation, pre-KNN account/privacy filtering, profile identity, revision and ownership fences, credential publication, failure rollback, cleanup and interruption recovery. Regression coverage includes an old file-backed reader surviving concurrent sqlite-vec cleanup, injected DROP/DELETE failures, unchanged settings after failed transitions, and no disabled-corpus document requests after resume.

The reviewer directly ran `git diff --check` and shell syntax validation. Cargo suites, frontend checks and live inference were run by the implementation/validation workers and parent; the reviewer inspected those tests and their reported outcomes without independently rerunning them.

The validation script was checked for explicit disposable per-case database paths, sentinel validation and resolved-path containment. It never defaults to installed application data. No application installation, deployment or push was performed.
