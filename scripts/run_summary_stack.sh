#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="$ROOT_DIR/.summary-stack-pids"
LOG_DIR="$ROOT_DIR/artifacts/summary_backend/run"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-summary-backend/bin/python}"
DRAMATIQ_BIN="${DRAMATIQ_BIN:-$ROOT_DIR/.venv-summary-backend/bin/dramatiq}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi
if [[ ! -x "$DRAMATIQ_BIN" ]]; then
  DRAMATIQ_BIN="dramatiq"
fi

mkdir -p "$PID_DIR" "$LOG_DIR"

export SUMMARY_BACKEND_DRY_RUN="${SUMMARY_BACKEND_DRY_RUN:-true}"
export SUMMARY_BACKEND_BROKER_URL="${SUMMARY_BACKEND_BROKER_URL:-redis://localhost:6379/0}"
export SUMMARY_BACKEND_API_HOST="${SUMMARY_BACKEND_API_HOST:-0.0.0.0}"
export SUMMARY_BACKEND_API_PORT="${SUMMARY_BACKEND_API_PORT:-8088}"
export SUMMARY_BACKEND_CORS_ORIGINS="${SUMMARY_BACKEND_CORS_ORIGINS:-http://localhost:8090,http://127.0.0.1:8090}"
export SUMMARY_BACKEND_WORKER_PROCESSES="${SUMMARY_BACKEND_WORKER_PROCESSES:-1}"
export SUMMARY_BACKEND_WORKER_THREADS="${SUMMARY_BACKEND_WORKER_THREADS:-4}"
export SUMMARY_FRONTEND_HOST="${SUMMARY_FRONTEND_HOST:-0.0.0.0}"
export SUMMARY_FRONTEND_PORT="${SUMMARY_FRONTEND_PORT:-8090}"
export SUMMARY_FRONTEND_BACKEND_HTTP_URL="${SUMMARY_FRONTEND_BACKEND_HTTP_URL:-http://localhost:${SUMMARY_BACKEND_API_PORT}}"
export SUMMARY_FRONTEND_BACKEND_WS_URL="${SUMMARY_FRONTEND_BACKEND_WS_URL:-ws://localhost:${SUMMARY_BACKEND_API_PORT}}"

start_process() {
  local name="$1"
  shift
  local pid_file="$PID_DIR/$name.pid"
  local log_file="$LOG_DIR/$name.log"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name already running pid=$(cat "$pid_file")"
    return
  fi
  cd "$ROOT_DIR"
  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" >"$log_file" 2>&1 </dev/null &
  else
    nohup "$@" >"$log_file" 2>&1 </dev/null &
  fi
  echo $! >"$pid_file"
  echo "$name started pid=$(cat "$pid_file") log=$log_file"
}

start_process worker "$DRAMATIQ_BIN" summary_backend.tasks --processes "$SUMMARY_BACKEND_WORKER_PROCESSES" --threads "$SUMMARY_BACKEND_WORKER_THREADS"
start_process backend "$PYTHON_BIN" -m summary_backend
start_process frontend "$PYTHON_BIN" -m summary_frontend

echo "backend:  http://localhost:${SUMMARY_BACKEND_API_PORT}"
echo "frontend: http://localhost:${SUMMARY_FRONTEND_PORT}"
