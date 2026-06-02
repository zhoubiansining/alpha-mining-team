#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_API_PORT="${DATA_API_PORT:-18001}"
BACK_TEST_PORT="${BACK_TEST_PORT:-18000}"
OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
LEADER_MODEL="${LEADER_MODEL:-GLM-4.7}"
PROPOSER_MODEL="${PROPOSER_MODEL:-GLM-4.7}"
CRITIC_MODEL="${CRITIC_MODEL:-GLM-4.7}"
EVALUATOR_ENDPOINT="${EVALUATOR_ENDPOINT:-http://127.0.0.1:${BACK_TEST_PORT}/evaluate}"
EVALUATOR_TIMEOUT="${EVALUATOR_TIMEOUT:-300}"
ALPHA_MINING_LOG_LEVEL="${ALPHA_MINING_LOG_LEVEL:-INFO}"
ALPHA_MINING_DEBUG_PROMPTS="${ALPHA_MINING_DEBUG_PROMPTS:-1}"

pids=()
cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  for pid in "${pids[@]:-}"; do
    wait "$pid" >/dev/null 2>&1 || true
  done
  popd >/dev/null 2>&1 || true
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for real LLM evaluation." >&2
  exit 1
fi

export DATA_API_PORT
export BACK_TEST_PORT
export DATA_API_BASE="${DATA_API_BASE:-http://127.0.0.1:${DATA_API_PORT}}"
export OPENAI_API_BASE LEADER_MODEL PROPOSER_MODEL CRITIC_MODEL EVALUATOR_ENDPOINT EVALUATOR_TIMEOUT ALPHA_MINING_LOG_LEVEL ALPHA_MINING_DEBUG_PROMPTS

pushd "${ROOT_DIR}" >/dev/null

nohup "$PYTHON_BIN" -m data_api.main > /tmp/data_api.eval.log 2>&1 &
pids+=("$!")
nohup "$PYTHON_BIN" -m back_test.main > /tmp/back_test.eval.log 2>&1 &
pids+=("$!")

printf 'Waiting for services...\n'
last_ready_error=""
for i in $(seq 1 60); do
  for pid in "${pids[@]:-}"; do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "Service process exited before readiness: pid=$pid" >&2
      echo "--- data_api log ---" >&2
      tail -n 80 /tmp/data_api.eval.log >&2 || true
      echo "--- back_test log ---" >&2
      tail -n 80 /tmp/back_test.eval.log >&2 || true
      exit 1
    fi
  done
  ready_output="$(DATA_API_PORT="$DATA_API_PORT" BACK_TEST_PORT="$BACK_TEST_PORT" "$PYTHON_BIN" - <<'PY' 2>&1 || true
import os
import json
import urllib.request
data_port = os.environ["DATA_API_PORT"]
back_port = os.environ["BACK_TEST_PORT"]
for url in (f"http://127.0.0.1:{data_port}/health", f"http://127.0.0.1:{back_port}/health"):
    with urllib.request.urlopen(url, timeout=2) as resp:
        resp.read(1)
with urllib.request.urlopen(f"http://127.0.0.1:{back_port}/openapi.json", timeout=2) as resp:
    schema = json.load(resp)
if "/evaluate" not in schema.get("paths", {}):
    raise RuntimeError("back_test service does not expose /evaluate")

# Verify data_api returns non-empty bars (avoid shadow data race condition)
try:
    univ_url = f"http://127.0.0.1:{data_port}/universe?name=HS300&market=cn_stock"
    univ_resp = urllib.request.urlopen(univ_url, timeout=5)
    univ_data = json.loads(univ_resp.read())
    symbols = univ_data.get("symbols", [])
    if not symbols:
        raise RuntimeError("data_api returned empty universe for HS300")
    # Try fetching bars for first symbol
    bars_url = f"http://127.0.0.1:{data_port}/bars/daily?symbol={symbols[0]}&start=2018-01-01&end=2023-12-31&market=cn_stock&adjust=qfq"
    bars_resp = urllib.request.urlopen(bars_url, timeout=5)
    bars_data = json.loads(bars_resp.read())
    if not bars_data.get("bars"):
        raise RuntimeError(f"data_api returned empty bars for symbol {symbols[0]}")
    print(f"data_api ready: universe={len(symbols)} symbols, first symbol has {len(bars_data.get('bars', []))} bars")
except Exception as e:
    raise RuntimeError(f"data_api not ready (data check failed): {e}")

print("ready")
PY
  )"
  if [[ "$ready_output" == *"ready"* ]]; then
    break
  fi
  last_ready_error="$ready_output"
  sleep 1
  if [[ "$i" -eq 60 ]]; then
    echo "Services did not become ready." >&2
    echo "--- readiness error ---" >&2
    echo "$last_ready_error" >&2
    echo "--- data_api log ---" >&2
    tail -n 80 /tmp/data_api.eval.log >&2 || true
    echo "--- back_test log ---" >&2
    tail -n 80 /tmp/back_test.eval.log >&2 || true
    exit 1
  fi
done

printf 'Running final evaluation pipeline...\n'
"$PYTHON_BIN" scripts/final_eval_pipeline.py \
  --baseline "${1:-momentum_20d}" \
  --max-iter "${MAX_ITERATIONS:-10}" \
  --universe "${UNIVERSE:-HS300}" \
  --start "${START_DATE:-2018-01-01}" \
  --end "${END_DATE:-2023-12-31}"

popd >/dev/null
trap - EXIT INT TERM
cleanup
