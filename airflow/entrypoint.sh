#!/bin/sh
set -e

if [ -n "$LOGS_SQL" ]; then
    export LOGS_SQL=$(echo "$LOGS_SQL" | base64 -d)
fi

if [ -n "$METRICS_SQL" ]; then
    export METRICS_SQL=$(echo "$METRICS_SQL" | base64 -d)
fi

exec python -m log_summarizer.main "$@"
