#!/usr/bin/env bash
# Explicit, disposable public-fixture eval. Never opens the installed app DB.
set -euo pipefail
cd "$(dirname "$0")/.."
task_dir=$(mktemp -d /tmp/emailops-embedding-eval.XXXXXX)
printf 'Synthetic eval workspace: %s\n' "$task_dir"
python3 scripts/generate_demo_db.py --demo-db "$task_dir/fixture/emailops.db" > "$task_dir/fixture.log"
touch "$task_dir/fixture/SYNTHETIC_EVAL_ONLY"
cargo build --no-default-features --features eval --manifest-path src-tauri/Cargo.toml --example embedding_lifecycle_eval --example chat_eval
cargo run --quiet --no-default-features --features eval --manifest-path src-tauri/Cargo.toml --example embedding_lifecycle_eval -- "$task_dir/fixture" | tee "$task_dir/lifecycle.log"
export EMAILOPS_EVAL_PROVIDER=openai_compatible
export EMAILOPS_EVAL_MODEL=google/gemma-4-e4b
export EMAILOPS_EVAL_BASE_URL=http://localhost:1234/v1
export EMAILOPS_EVAL_EMBEDDING_SOURCE=server
export EMAILOPS_EVAL_CONTEXT_WINDOW="${EMAILOPS_EVAL_CONTEXT_WINDOW:-8192}"
failed=0
for case_id in retrieval_trashed_quote_not_cited retrieval_excludes_junk_lookalike app_help_ollama_en mem_borgbase_customer_number demo_memory_pending_tasks; do
  # The harness's conservative page-at-a-time snapshot takes minutes for vectors.
  # Make our own isolated SQLite snapshot, then open only that disposable copy.
  python3 - "$task_dir" "$case_id" <<'PYCASE'
import pathlib, sqlite3, sys
root = pathlib.Path(sys.argv[1]).resolve()
source = (root / 'fixture' / 'emailops.db').resolve()
if not (source.parent / 'SYNTHETIC_EVAL_ONLY').is_file() or not source.is_relative_to(root):
    raise SystemExit('source must be the sentinel-marked synthetic fixture under this workspace')
destination = (root / 'cases' / sys.argv[2] / 'emailops.db').resolve()
if not destination.is_relative_to(root):
    raise SystemExit('destination must stay inside the disposable workspace')
destination.parent.mkdir(parents=True)
with sqlite3.connect(f'file:{source}?mode=ro', uri=True) as src, sqlite3.connect(destination) as dst:
    src.backup(dst)
PYCASE
  cargo run --quiet --no-default-features --features eval --manifest-path src-tauri/Cargo.toml --example chat_eval -- --tier all --case "$case_id" --in-place-dangerous --prod-db "$task_dir/cases/$case_id/emailops.db" --cases-dir src-tauri/evals/chat/cases --out "$task_dir/reports/$case_id" 2>&1 | tee "$task_dir/$case_id.log"
  if rg -q '\[eval\]    (FAIL|ERROR)' "$task_dir/$case_id.log"; then failed=1; fi
done
printf 'Reports retained at %s\n' "$task_dir"
exit "$failed"
