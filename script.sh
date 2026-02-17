#!/usr/bin/env bash
set -euo pipefail

# ========= Config =========
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"   # override if needed
MODEL_ID="${MODEL_ID:-hicache-persist-demo}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-30000}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-1800}"       # 30 min default
HICACHE_DIR="${HICACHE_DIR:-/tmp/sglang_hicache_persist_test}"
LOG1="${LOG1:-/tmp/sglang_hicache_run1.log}"
LOG2="${LOG2:-/tmp/sglang_hicache_run2.log}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"
SGLANG_IS_FLASHINFER_AVAILABLE="${SGLANG_IS_FLASHINFER_AVAILABLE:-false}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.45}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-2048}"
HICACHE_RATIO="${HICACHE_RATIO:-1.10}"
HICACHE_STORAGE_PREFETCH_POLICY="${HICACHE_STORAGE_PREFETCH_POLICY:-wait_complete}"
STORAGE_SYNC_TIMEOUT_SEC="${STORAGE_SYNC_TIMEOUT_SEC:-180}"
STORAGE_STABLE_SEC="${STORAGE_STABLE_SEC:-8}"
PAGE_SIZE="${PAGE_SIZE:-1}"

PID=""
TAIL_PID=""

# ========= Prereq =========
command -v curl >/dev/null || { echo "curl not found"; exit 1; }
command -v jq >/dev/null || { echo "jq not found"; exit 1; }
[[ "$PAGE_SIZE" =~ ^[0-9]+$ ]] || { echo "PAGE_SIZE must be a positive integer"; exit 1; }
(( PAGE_SIZE > 0 )) || { echo "PAGE_SIZE must be > 0"; exit 1; }

fail() {
  echo "FAIL: $*"
  exit 1
}

stop_log_stream() {
  if [[ -n "${TAIL_PID:-}" ]]; then
    kill "$TAIL_PID" >/dev/null 2>&1 || true
    wait "$TAIL_PID" >/dev/null 2>&1 || true
    TAIL_PID=""
  fi
}

cleanup_server() {
  stop_log_stream
  if [[ -n "${PID:-}" ]]; then
    kill "$PID" >/dev/null 2>&1 || true
    wait "$PID" >/dev/null 2>&1 || true
    PID=""
  fi
}

trap cleanup_server EXIT

start_log_stream() {
  local log_file="$1"
  echo "----- Live startup logs: $log_file -----"
  tail -n 0 -f "$log_file" &
  TAIL_PID=$!
}

start_server() {
  local log_file="$1"
  : > "$log_file"

  SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$HICACHE_DIR" \
  SGLANG_IS_FLASHINFER_AVAILABLE="$SGLANG_IS_FLASHINFER_AVAILABLE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$MODEL_ID" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --tp-size 1 \
    --page-size "$PAGE_SIZE" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    --enable-hierarchical-cache \
    --hicache-ratio "$HICACHE_RATIO" \
    --attention-backend "$ATTENTION_BACKEND" \
    --sampling-backend "$SAMPLING_BACKEND" \
    --log-level warning \
    --enable-cache-report \
    --hicache-storage-backend file \
    --hicache-storage-prefetch-policy "$HICACHE_STORAGE_PREFETCH_POLICY" \
    --hicache-write-policy write_through \
    --hicache-storage-backend-extra-config '{"prefetch_threshold":1}' \
    >"$log_file" 2>&1 &
  PID=$!

  echo "Server PID: $PID"
  start_log_stream "$log_file"
}

wait_ready() {
  local log_file="$1"
  local elapsed=0

  echo "Waiting for server http://$SERVER_HOST:$SERVER_PORT to be ready..."
  while true; do
    if curl -fsS "http://$SERVER_HOST:$SERVER_PORT/v1/models" >/dev/null 2>&1; then
      echo "Server is ready."
      stop_log_stream
      return 0
    fi

    if ! kill -0 "$PID" >/dev/null 2>&1; then
      stop_log_stream
      echo "Server exited during startup. Last logs:"
      tail -n 200 "$log_file" || true
      if grep -q "cannot find -lcuda" "$log_file"; then
        echo "Hint: FlashInfer JIT link failed with missing -lcuda."
        echo "Current backends: attention=$ATTENTION_BACKEND sampling=$SAMPLING_BACKEND"
      fi
      if grep -q "Not enough host memory available" "$log_file"; then
        echo "Hint: HiCache host memory is too large for this machine."
        echo "Try smaller pools, for example:"
        echo "  MEM_FRACTION_STATIC=0.35 MAX_TOTAL_TOKENS=1024 HICACHE_RATIO=1.05 bash test.sh"
      fi
      fail "server process died before becoming ready"
    fi

    sleep 2
    elapsed=$((elapsed + 2))

    if (( elapsed % 20 == 0 )); then
      echo "Still waiting... (${elapsed}s)"
    fi

    if (( elapsed >= STARTUP_TIMEOUT_SEC )); then
      stop_log_stream
      echo "Startup timed out. Last logs:"
      tail -n 200 "$log_file" || true
      fail "server did not become ready within ${STARTUP_TIMEOUT_SEC}s"
    fi
  done
}

cached_tokens() {
  jq -r '.usage.prompt_tokens_details.cached_tokens // 0' "$1"
}

count_hicache_files() {
  find "$HICACHE_DIR" -type f | wc -l | tr -d ' '
}

wait_for_storage_flush() {
  local min_files="$1"
  local elapsed=0
  local stable=0
  local last_count=-1
  local cur_count=0

  echo "Waiting for HiCache storage flush (min_files=$min_files, timeout=${STORAGE_SYNC_TIMEOUT_SEC}s)..."
  while (( elapsed < STORAGE_SYNC_TIMEOUT_SEC )); do
    cur_count="$(count_hicache_files)"

    if [[ "$cur_count" == "$last_count" ]] && (( cur_count >= min_files )); then
      stable=$((stable + 1))
    else
      stable=0
    fi

    if (( stable >= STORAGE_STABLE_SEC )); then
      echo "HiCache storage appears stable: file_count=$cur_count"
      return 0
    fi

    last_count="$cur_count"
    sleep 1
    elapsed=$((elapsed + 1))
    if (( elapsed % 10 == 0 )); then
      echo "  storage progress: file_count=$cur_count elapsed=${elapsed}s"
    fi
  done

  echo "Timed out waiting for storage flush. file_count=$cur_count"
  return 1
}

check_response_reasonable() {
  local file="$1"
  local label="$2"

  if jq -e '.error != null' "$file" >/dev/null; then
    echo "$label API error:"
    jq '.error' "$file"
    fail "$label returned error"
  fi

  local text
  text="$(jq -r '.choices[0].text // ""' "$file" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
  local completion_tokens
  completion_tokens="$(jq -r '.usage.completion_tokens // 0' "$file")"

  [[ ${#text} -ge 80 ]] || fail "$label response too short"
  [[ "$completion_tokens" -gt 0 ]] || fail "$label completion_tokens is 0"
  echo "$text" | grep -Eq '[A-Za-z]' || fail "$label looks invalid"

  echo "$label preview: ${text:0:180}"
}

PROMPT="$(cat <<'EOF'
You are an LLM assistant helping a support engineer draft a customer-facing response.

Company context:
- Product: NimbusDB Cloud (managed Postgres-compatible service)
- Customer: Ridgeway Analytics (enterprise account, SOC2-required, 24x7 ETL + BI workloads)
- Region: us-east-1 primary, us-west-2 disaster-recovery replica
- Contractual SLO: 99.95% monthly availability
- Current plan: 8 vCPU / 64 GB RAM primary, read replica in same region

Incident timeline (all times UTC):
- 2026-02-12 01:07: Alerts fired for elevated query latency (p95 from 120ms to 1.9s)
- 2026-02-12 01:15: Replica lag increased from <2s to 4m 20s
- 2026-02-12 01:24: Customer ETL job "daily_fact_rollup" started timing out
- 2026-02-12 01:40: On-call noticed burst of long-running queries from BI dashboard refresh
- 2026-02-12 02:05: Emergency change applied: max_connections reduced 1200 -> 800; queueing enabled
- 2026-02-12 02:22: p95 latency improved to 430ms
- 2026-02-12 03:10: Replica lag returned to <10s

Observed signals:
- CPU saturation on primary at 92-98% for ~70 minutes
- IO wait spikes on data volume (gp3 baseline exceeded, burst credits dropped)
- Top wait events: Lock:transactionid, IO:DataFileRead
- Slow query sample:
  SELECT customer_id, sum(amount)
  FROM fact_orders
  WHERE created_at >= now() - interval '30 days'
  GROUP BY customer_id
  ORDER BY sum(amount) DESC
  LIMIT 1000;
- Query pattern came from repeated dashboard auto-refresh at 30s interval across 200 users
- No evidence of data corruption
- No failed failover events
- Backup jobs completed successfully

Customer asks:
1) "Was this an outage or performance degradation?"
2) "Did we lose data or risk replication inconsistency?"
3) "What permanent fixes will prevent recurrence?"
4) "What should we change in our dashboard/query patterns?"

Your task:
Write a clear customer-facing response with these sections:
- Executive summary (non-alarmist, plain English)
- Root cause analysis
- Data integrity assessment
- Immediate mitigations already taken
- Permanent corrective actions (platform side + customer side)
- Next 7-day validation plan with measurable checkpoints

Keep tone professional and concise. Avoid blame. Include concrete recommendations.
EOF
)"

make_request() {
  local out="$1"
  jq -n \
    --arg model "$MODEL_ID" \
    --arg prompt "$PROMPT" \
    '{model:$model,prompt:$prompt,max_tokens:256,temperature:0}' \
  | curl -fsS "http://$SERVER_HOST:$SERVER_PORT/v1/completions" \
      -H "Content-Type: application/json" \
      -d @- >"$out"
}

echo "Cleaning storage dir: $HICACHE_DIR"
# rm -rf "$HICACHE_DIR"
mkdir -p "$HICACHE_DIR"

echo "Launch config:"
echo "  model_path=$MODEL_PATH"
echo "  mem_fraction_static=$MEM_FRACTION_STATIC"
echo "  max_total_tokens=$MAX_TOTAL_TOKENS"
echo "  hicache_ratio=$HICACHE_RATIO"
echo "  hicache_storage_prefetch_policy=$HICACHE_STORAGE_PREFETCH_POLICY"
echo "  attention_backend=$ATTENTION_BACKEND"
echo "  sampling_backend=$SAMPLING_BACKEND"
echo "  page_size=$PAGE_SIZE"

echo "=== Run #1 ==="
start_server "$LOG1"
wait_ready "$LOG1"

make_request /tmp/resp_run1_first.json
make_request /tmp/resp_run1_second.json

check_response_reasonable /tmp/resp_run1_first.json "run1_first"
check_response_reasonable /tmp/resp_run1_second.json "run1_second"

C1="$(cached_tokens /tmp/resp_run1_first.json)"
C2="$(cached_tokens /tmp/resp_run1_second.json)"
echo "run1_first_cached_tokens=$C1"
echo "run1_second_cached_tokens=$C2"

[[ "$C2" -gt 10 ]] || fail "run1_second cached_tokens must be > 10"

sleep 2
FILE_COUNT="$(find "$HICACHE_DIR" -type f | wc -l | tr -d ' ')"
echo "hicache_file_count=$FILE_COUNT"
[[ "$FILE_COUNT" -gt 0 ]] || fail "no HiCache files written to disk"

# Ensure storage writes are actually persisted before restart.
# File count scales by page count, not raw token count.
# A 25% floor handles radix dedupe and non-page-aligned tails.
C2_PAGE_COUNT=$((C2 / PAGE_SIZE))
MIN_EXPECTED_FILES=$((C2_PAGE_COUNT / 4))
if (( MIN_EXPECTED_FILES < 1 )); then
  MIN_EXPECTED_FILES=1
fi
wait_for_storage_flush "$MIN_EXPECTED_FILES" || fail "HiCache files did not flush to expected level"

echo "Restarting server..."
cleanup_server
sleep 2

echo "=== Run #2 (after restart) ==="
start_server "$LOG2"
wait_ready "$LOG2"

make_request /tmp/resp_run2_first.json
check_response_reasonable /tmp/resp_run2_first.json "run2_first_after_restart"

C3="$(cached_tokens /tmp/resp_run2_first.json)"
echo "run2_first_after_restart_cached_tokens=$C3"
[[ "$C3" -gt 10 ]] || fail "after restart first request cached_tokens must be > 10 (disk reuse proof)"

echo "PASS: responses are reasonable and cached_tokens > 10 after restart"
echo "Logs: $LOG1 $LOG2"