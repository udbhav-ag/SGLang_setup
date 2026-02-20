#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d)}"

METRICS_DIR="${METRICS_DIR:-$SCRIPT_DIR/metrics}"
if [[ -z "${BATCH_SIZES:-}" && -n "${BATCH_SIZE:-}" ]]; then
  BATCH_SIZES_STR="${BATCH_SIZE}"
else
  BATCH_SIZES_STR="${BATCH_SIZES:-1 2 4 8 16 32}"
fi
if [[ -z "${INPUT_LENS:-}" && -n "${INPUT_LEN:-}" ]]; then
  INPUT_LENS_STR="${INPUT_LEN}"
else
  INPUT_LENS_STR="${INPUT_LENS:-4096 8192 16384 32768}"
fi

BATCH_SIZES_STR="${BATCH_SIZES_STR//,/ }"
INPUT_LENS_STR="${INPUT_LENS_STR//,/ }"

read -r -a BATCH_SIZES_ARR <<< "$BATCH_SIZES_STR"
read -r -a INPUT_LENS_ARR <<< "$INPUT_LENS_STR"

if [[ "${#BATCH_SIZES_ARR[@]}" -eq 0 ]]; then
  echo "No batch sizes configured. Set BATCH_SIZES or BATCH_SIZE."
  exit 1
fi
if [[ "${#INPUT_LENS_ARR[@]}" -eq 0 ]]; then
  echo "No input lengths configured. Set INPUT_LENS or INPUT_LEN."
  exit 1
fi

FILE_PREFIX="${FILE_PREFIX:-metrics_sweep}"
OUT_FILE="${OUT_FILE:-$METRICS_DIR/${FILE_PREFIX}_${RUN_ID}.txt}"
PYTHON_BIN="${PYTHON_BIN:-/Udbhav/miniconda3/envs/venv/bin/python}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-$SCRIPT_DIR/prefill_gpu_decode_cpu.py}"
FAIL_ON_CASE_ERROR="${FAIL_ON_CASE_ERROR:-0}"

METRIC_REGEX='DECODE_MAX_TOTAL_TOKENS=|PREFILL_MODEL_LOAD_S=|PREFILL_TOKENIZER_LOAD_S=|PREFILL_TOTAL_LOAD_S=|PREFILL_FORWARD_FIRST_HALF_S=|PREFILL_FORWARD_S=|PREFILL_KV_SAVE_S=|DECODE_MODEL_LOAD_S=|DECODE_TOKENIZER_LOAD_S=|DECODE_TOTAL_LOAD_S=|DECODE_TTFT_S=|GPU prefill time:|KV load timing:'

mkdir -p "$METRICS_DIR"
if [[ ! -w "$METRICS_DIR" ]]; then
  echo "Metrics directory is not writable: $METRICS_DIR"
  exit 1
fi

TOTAL_CASES=$((${#BATCH_SIZES_ARR[@]} * ${#INPUT_LENS_ARR[@]}))
CASE_INDEX=0
FAILED_CASES=0

rm -f "$OUT_FILE"
{
  echo "run_id=$RUN_ID"
  echo "batch_sizes=${BATCH_SIZES_ARR[*]}"
  echo "input_lens=${INPUT_LENS_ARR[*]}"
  echo "total_cases=$TOTAL_CASES"
  echo
} > "$OUT_FILE"

echo "Running metrics sweep..."
echo "Total cases: $TOTAL_CASES"
echo "Combined metrics file: $OUT_FILE"

for batch_size in "${BATCH_SIZES_ARR[@]}"; do
  for input_len in "${INPUT_LENS_ARR[@]}"; do
    CASE_INDEX=$((CASE_INDEX + 1))
    CASE_TAG="bs${batch_size}_in${input_len}"
    CASE_LOG_FILE="${METRICS_DIR}/${FILE_PREFIX}_${CASE_TAG}_${RUN_ID}.log"
    CASE_METRICS_FILE="${METRICS_DIR}/${FILE_PREFIX}_${CASE_TAG}_${RUN_ID}.txt"

    echo "[$CASE_INDEX/$TOTAL_CASES] Running case: batch_size=$batch_size input_len=$input_len"
    if "$PYTHON_BIN" "$PIPELINE_SCRIPT" "$@" --batch-size "$batch_size" --input-len "$input_len" >"$CASE_LOG_FILE" 2>&1; then
      STATUS="SUCCESS"
    else
      STATUS="FAILED"
      FAILED_CASES=$((FAILED_CASES + 1))
      echo "  Case failed. See log: $CASE_LOG_FILE"
    fi

    grep -E "$METRIC_REGEX" "$CASE_LOG_FILE" > "$CASE_METRICS_FILE" || true

    {
      echo "=== case=$CASE_TAG status=$STATUS ==="
      echo "log_file=$CASE_LOG_FILE"
      echo "metrics_file=$CASE_METRICS_FILE"
      cat "$CASE_METRICS_FILE"
      echo
    } >> "$OUT_FILE"
  done
done

echo
echo "Done."
echo "Extracted metrics (all cases): $OUT_FILE"
if [[ "$FAILED_CASES" -gt 0 ]]; then
  echo "Completed with failures: $FAILED_CASES/$TOTAL_CASES cases failed."
  if [[ "$FAIL_ON_CASE_ERROR" == "1" ]]; then
    exit 1
  fi
else
  echo "All cases succeeded: $TOTAL_CASES/$TOTAL_CASES."
fi
